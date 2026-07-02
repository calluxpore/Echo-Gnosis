import threading
import logging
import numpy as np
import sounddevice as sd

logger = logging.getLogger("EchoGnosis.Audio")

class SonicMap:
    def __init__(self, sample_rate=44100, max_voices=4):
        self.sample_rate = sample_rate
        self.max_voices = max_voices
        self.lock = threading.Lock()
        
        # Initialize voices
        # Each voice has:
        # - label: name of the object tracked
        # - phase: phase accumulator for sine wave
        # - freq: current frequency (Hz)
        # - target_freq: target frequency (Hz)
        # - amp: current amplitude (0.0 to 1.0)
        # - target_amp: target amplitude (0.0 to 1.0)
        # - pan: current panning (-1.0 to 1.0)
        # - target_pan: target panning (-1.0 to 1.0)
        self.voices = []
        for _ in range(self.max_voices):
            self.voices.append({
                'label': None,
                'phase': 0.0,
                'freq': 440.0,
                'target_freq': 440.0,
                'amp': 0.0,
                'target_amp': 0.0,
                'pan': 0.0,
                'target_pan': 0.0
            })
            
        # Smoothing coefficient per sample
        # At 44.1kHz, 0.0005 gives a smoothing time constant of ~45ms,
        # which is fast enough to react but slow enough to prevent clicks.
        self.alpha = 0.0005
        
        self.stream = None
        logger.info("SonicMap initialized with %d max voices.", self.max_voices)

    def start(self):
        """Starts the non-blocking sounddevice output stream."""
        logger.info("Starting audio stream...")
        self.stream = sd.OutputStream(
            channels=2,
            callback=self._audio_callback,
            samplerate=self.sample_rate,
            dtype='float32'
        )
        self.stream.start()
        logger.info("Audio stream started.")

    def stop(self):
        """Stops the audio stream."""
        if self.stream:
            logger.info("Stopping audio stream...")
            self.stream.stop()
            self.stream.close()
            self.stream = None
            logger.info("Audio stream stopped.")

    def update_objects(self, objects):
        """
        Updates voice assignments and targets based on detected objects.
        Expected schema for each object:
        {
            "label": str,
            "position_x": float (-1.0 to 1.0),
            "position_y": float (-1.0 to 1.0),
            "depth": float (0.1 to 1.0),
            "size": float (0.1 to 1.0),
            ...
        }
        """
        if not isinstance(objects, list):
            logger.warning("update_objects received invalid type: %s", type(objects))
            return
            
        with self.lock:
            # We track which voices were matched to incoming objects
            matched_voice_indices = set()
            
            for obj in objects:
                label = obj.get('label', 'object')
                pos_x = float(obj.get('position_x', 0.0))
                pos_y = float(obj.get('position_y', 0.0))
                depth = float(obj.get('depth', 0.5))
                
                # Clamp values to safe boundaries
                pos_x = max(-1.0, min(1.0, pos_x))
                pos_y = max(-1.0, min(1.0, pos_y))
                depth = max(0.1, min(1.0, depth))
                
                # Calculate targets:
                # - position_x maps to panning: -1.0 (left) to 1.0 (right)
                target_pan = pos_x
                # - position_y maps to base frequency: bottom (<200Hz) to top (>800Hz)
                # Map [-1, 1] to [200, 800]
                target_freq = 500.0 + 300.0 * pos_y
                # - depth maps to amplitude: near = loud, far = quiet
                # Map depth [0.1, 1.0] to amplitude [0.2, 0.02]
                # Near (0.1) -> 0.2, Far (1.0) -> 0.02
                target_amp = 0.2 * (1.1 - depth)
                
                # Step 1: Find a voice already tracking this label
                found_voice_idx = None
                for idx, voice in enumerate(self.voices):
                    if idx in matched_voice_indices:
                        continue
                    if voice['label'] == label and voice['target_amp'] > 0:
                        found_voice_idx = idx
                        break
                        
                # Step 2: If not found, find an idle voice (target_amp == 0)
                if found_voice_idx is None:
                    for idx, voice in enumerate(self.voices):
                        if idx in matched_voice_indices:
                            continue
                        if voice['target_amp'] == 0:
                            found_voice_idx = idx
                            # Initialize frequency, amp, pan from defaults/current to prevent leaps
                            voice['label'] = label
                            voice['freq'] = target_freq
                            voice['pan'] = target_pan
                            voice['amp'] = 0.0 # start silent and fade in
                            break
                            
                # Step 3: If still not found, steal the quietest active voice
                if found_voice_idx is None:
                    min_amp = float('inf')
                    for idx, voice in enumerate(self.voices):
                        if idx in matched_voice_indices:
                            continue
                        if voice['target_amp'] < min_amp:
                            min_amp = voice['target_amp']
                            found_voice_idx = idx
                    if found_voice_idx is not None:
                        voice = self.voices[found_voice_idx]
                        voice['label'] = label
                        # Keep current frequency, amp, pan but update targets
                
                # Apply new targets
                if found_voice_idx is not None:
                    self.voices[found_voice_idx]['target_freq'] = target_freq
                    self.voices[found_voice_idx]['target_pan'] = target_pan
                    self.voices[found_voice_idx]['target_amp'] = target_amp
                    matched_voice_indices.add(found_voice_idx)
            
            # Step 4: Fade out any voices that were not matched to any detected object
            for idx, voice in enumerate(self.voices):
                if idx not in matched_voice_indices:
                    voice['target_amp'] = 0.0

    def _audio_callback(self, outdata, frames, time_info, status):
        """Audio callback method executed in PortAudio thread."""
        if status:
            logger.warning("Audio stream status warning: %s", status)
            
        # Initialize buffers to zero
        outdata.fill(0)
        
        with self.lock:
            for voice in self.voices:
                # If voice is silent and has no target amplitude, skip synthesis
                if voice['amp'] < 1e-4 and voice['target_amp'] < 1e-4:
                    voice['amp'] = 0.0
                    voice['target_amp'] = 0.0
                    voice['label'] = None
                    continue
                
                # Synthesize samples for this block
                for i in range(frames):
                    # Exponential smoothing of synthesis parameters
                    voice['freq'] += self.alpha * (voice['target_freq'] - voice['freq'])
                    voice['amp'] += self.alpha * (voice['target_amp'] - voice['amp'])
                    voice['pan'] += self.alpha * (voice['target_pan'] - voice['pan'])
                    
                    # Accumulate phase to keep the waveform continuous
                    voice['phase'] += 2 * np.pi * voice['freq'] / self.sample_rate
                    if voice['phase'] > 2 * np.pi:
                        # Wrap phase around 2*pi
                        voice['phase'] %= (2 * np.pi)
                        
                    # Calculate sample
                    sample = np.sin(voice['phase']) * voice['amp']
                    
                    # Constant-power panning
                    # Map pan [-1.0, 1.0] to [0.0, 1.0]
                    p = (voice['pan'] + 1.0) / 2.0
                    left_gain = np.sqrt(1.0 - p)
                    right_gain = np.sqrt(p)
                    
                    # Add to output channels
                    outdata[i, 0] += sample * left_gain
                    outdata[i, 1] += sample * right_gain
                    
            # Clip output to prevent digital distortion if cumulative amplitude exceeds 1.0
            np.clip(outdata, -1.0, 1.0, out = outdata)
