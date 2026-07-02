import os
import time
import queue
import threading
import subprocess
import logging
import winsound

logger = logging.getLogger("EchoGnosis.TTS")

class PiperAlerts:
    def __init__(self, piper_exe=r"c:\Users\GEU\Desktop\EG\piper\piper.exe", 
                 model_path=r"c:\Users\GEU\Desktop\EG\piper\en_US-lessac-medium.onnx",
                 cooldown_seconds=5.0):
        self.piper_exe = piper_exe
        self.model_path = model_path
        self.cooldown_seconds = cooldown_seconds
        
        # Dictionary to track last alert time for each label
        # format: {label: timestamp}
        self.cooldowns = {}
        
        # Thread-safe queue for alert texts
        self.alert_queue = queue.Queue()
        
        # Temp file path for generating WAV
        self.temp_wav = os.path.abspath(os.path.join(os.path.dirname(__file__), "temp_alert.wav"))
        
        # Worker thread status
        self.running = False
        self.worker_thread = None
        
        logger.info("PiperAlerts initialized.")
        logger.info(f"Piper executable: {self.piper_exe}")
        logger.info(f"Voice model: {self.model_path}")

    def start(self):
        """Starts the TTS speech worker thread."""
        if self.running:
            return
        
        self.running = True
        self.worker_thread = threading.Thread(target=self._speech_worker, daemon=True)
        self.worker_thread.start()
        logger.info("TTS speech worker thread started.")

    def stop(self):
        """Stops the speech worker thread."""
        self.running = False
        self.alert_queue.put(None) # Sentinel to wake up/stop worker
        if self.worker_thread:
            self.worker_thread.join(timeout=2.0)
            self.worker_thread = None
        
        # Cleanup temp WAV file if it exists
        if os.path.exists(self.temp_wav):
            try:
                os.remove(self.temp_wav)
            except Exception as e:
                logger.warning(f"Could not remove temp WAV file: {e}")
                
        logger.info("TTS speech worker thread stopped.")

    def process_hazards(self, objects):
        """Checks detected objects for hazards and queues speech alerts with cooldown."""
        if not isinstance(objects, list):
            return
            
        current_time = time.time()
        
        for obj in objects:
            if obj.get('is_hazard', False):
                label = obj.get('label', 'object').lower().strip()
                
                # Check cooldown for this specific hazard label
                last_alert_time = self.cooldowns.get(label, 0.0)
                if current_time - last_alert_time >= self.cooldown_seconds:
                    # Construct alert message
                    alert_text = f"{label} ahead."
                    logger.info(f"Hazard detected! Queueing alert: '{alert_text}'")
                    
                    self.alert_queue.put(alert_text)
                    self.cooldowns[label] = current_time

    def _speech_worker(self):
        """Sequential worker that processes the queue, calls Piper, and plays the alert."""
        while self.running:
            try:
                alert_text = self.alert_queue.get(timeout=0.5)
                if alert_text is None:
                    # Sentinel received, exit
                    break
                    
                self._generate_and_play(alert_text)
                self.alert_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in speech worker loop: {e}", exc_info=True)

    def _generate_and_play(self, text):
        """Runs Piper to write WAV, then plays it synchronously in this thread."""
        if not os.path.exists(self.piper_exe) or not os.path.exists(self.model_path):
            logger.error("Piper executable or voice model not found. Cannot speak.")
            return

        command = [
            self.piper_exe,
            "--model", self.model_path,
            "--output_file", self.temp_wav
        ]
        
        try:
            logger.debug(f"Synthesizing alert text: '{text}'")
            # Run piper process
            process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Send text to stdin and wait for completion
            stdout, stderr = process.communicate(input=text, timeout=5.0)
            
            if process.returncode != 0:
                logger.error(f"Piper exited with code {process.returncode}. Stderr: {stderr}")
                return
                
            if os.path.exists(self.temp_wav) and os.path.getsize(self.temp_wav) > 0:
                logger.debug(f"Playing WAV file: {self.temp_wav}")
                # Play audio synchronously inside this worker thread to avoid overlapping audio
                winsound.PlaySound(self.temp_wav, winsound.SND_FILENAME)
            else:
                logger.error("WAV file was not generated or is empty.")
                
        except subprocess.TimeoutExpired:
            logger.error("Piper synthesis timed out.")
            try:
                process.kill()
            except:
                pass
        except Exception as e:
            logger.error(f"Failed to generate or play speech alert: {e}", exc_info=True)
