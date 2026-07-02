import os
import sys
import time
import argparse
import logging
import threading
import cv2
from vision_processor import VisionAgent
from audio_synthesizer import SonicMap
from tts_alerter import PiperAlerts

def setup_logging(level_name):
    # Map string to logging levels
    levels = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR
    }
    level = levels.get(level_name.upper(), logging.INFO)
    
    # Configure root logger
    logging.basicConfig(
        level=level,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

def main():
    parser = argparse.ArgumentParser(description="Echo-Gnosis: Local Audio-Tactile Semiotics Assistant")
    parser.add_argument("--camera", type=int, default=0, help="Index of the USB camera (default: 0)")
    parser.add_argument("--piper-exe", type=str, default=r"c:\Users\GEU\Desktop\EG\piper\piper.exe", 
                        help="Path to the local Piper executable")
    parser.add_argument("--piper-model", type=str, default=r"c:\Users\GEU\Desktop\EG\piper\en_US-lessac-medium.onnx", 
                        help="Path to the Piper ONNX voice model")
    parser.add_argument("--cooldown", type=float, default=5.0, help="Cooldown in seconds for the same hazard alert (default: 5.0)")
    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], 
                        help="Set the logging level (default: INFO)")
    
    args = parser.parse_args()
    
    setup_logging(args.log_level)
    logger = logging.getLogger("EchoGnosis.Main")
    
    logger.info("Initializing Echo-Gnosis Live GUI Application...")
    
    sonic_map = None
    piper_alerts = None
    vision_agent = None
    cap = None
    analysis_thread = None
    
    # Shared variables for vision thread synchronization
    frame_lock = threading.Lock()
    latest_gui_frame = None
    has_new_frame = False
    
    detected_objects_lock = threading.Lock()
    detected_objects = []
    
    running = True
    
    def vision_analysis_loop():
        """Background thread executing Ollama analysis, updating SonicMap & PiperAlerts."""
        nonlocal latest_gui_frame, has_new_frame, detected_objects
        logger.info("Background Vision Analysis thread started.")
        
        while running:
            frame_to_analyze = None
            
            with frame_lock:
                if has_new_frame:
                    frame_to_analyze = latest_gui_frame.copy()
                    has_new_frame = False
            
            if frame_to_analyze is not None:
                # Query local Ollama vision model (blocks on network/inference)
                objects = vision_agent.analyze_frame(frame_to_analyze)
                
                if objects is not None:
                    logger.info(f"Ollama detected: {objects}")
                    
                    with detected_objects_lock:
                        detected_objects = objects
                        
                    # Update active sound parameters
                    sonic_map.update_objects(objects)
                    
                    # Run hazard speech alerts checking
                    piper_alerts.process_hazards(objects)
                else:
                    logger.warning("Ollama vision analysis returned invalid data. Visual state retained.")
            
            # Avoid high CPU usage in analysis loop sleep
            time.sleep(0.03)
            
        logger.info("Background Vision Analysis thread finished.")
        
    try:
        # 1. Initialize and start continuous audio synthesizer (SonicMap)
        sonic_map = SonicMap(sample_rate=44100, max_voices=4)
        sonic_map.start()
        
        # 2. Initialize and start the Piper alerts class
        piper_alerts = PiperAlerts(
            piper_exe=args.piper_exe,
            model_path=args.piper_model,
            cooldown_seconds=args.cooldown
        )
        piper_alerts.start()
        
        # 3. Initialize Vision Agent
        vision_agent = VisionAgent()
        
        # 4. Initialize USB Camera inside Main Thread (required for OpenCV UI)
        logger.info(f"Opening camera index {args.camera}...")
        cap = cv2.VideoCapture(args.camera)
        if not cap.isOpened():
            logger.error(f"Failed to open USB Camera at index {args.camera}")
            raise RuntimeError(f"Could not open video device at index {args.camera}")
        logger.info("USB Camera opened successfully.")
        
        # 5. Start the background analysis thread
        analysis_thread = threading.Thread(target=vision_analysis_loop, daemon=True)
        analysis_thread.start()
        
        logger.info("Echo-Gnosis GUI Running. Press 'q' or 'ESC' in the feed window to quit.")
        
        # 6. Main GUI Thread Loop (30 FPS)
        while running:
            ret, frame = cap.read()
            if not ret:
                logger.warning("Failed to capture frame from webcam.")
                time.sleep(0.01)
                continue
            
            # Send latest frame copy to background thread
            with frame_lock:
                latest_gui_frame = frame.copy()
                has_new_frame = True
                
            # Safely fetch the latest parsed visual targets
            with detected_objects_lock:
                current_objects = list(detected_objects)
                
            # Render Bounding Boxes & Warning Banner
            h, w, _ = frame.shape
            has_active_hazard = False
            active_hazards = []
            
            for obj in current_objects:
                label = obj.get('label', 'object')
                pos_x = float(obj.get('position_x', 0.0))
                pos_y = float(obj.get('position_y', 0.0))
                depth = float(obj.get('depth', 0.5))
                size = float(obj.get('size', 0.2))
                is_hazard = bool(obj.get('is_hazard', False))
                
                # Calculate pixel coordinates from relative positions
                # position_x: -1.0 (far left) to 1.0 (far right)
                cx = int((pos_x + 1.0) / 2.0 * w)
                # position_y: -1.0 (bottom) to 1.0 (top)
                cy = int((1.0 - pos_y) / 2.0 * h) # invert y for pixel coordinates
                
                # Bounding box dimensions scaled by relative size parameter
                bw = int(size * w)
                bh = int(size * h)
                
                x1 = max(0, int(cx - bw / 2.0))
                y1 = max(0, int(cy - bh / 2.0))
                x2 = min(w - 1, int(cx + bw / 2.0))
                y2 = min(h - 1, int(cy + bh / 2.0))
                
                # Select visual representation style
                if is_hazard:
                    color = (0, 0, 255) # BGR Red
                    thickness = 3
                    has_active_hazard = True
                    active_hazards.append(label.lower())
                    text = f"HAZARD: {label.upper()} (d:{depth:.1f})"
                else:
                    color = (0, 255, 0) # BGR Green
                    thickness = 2
                    text = f"{label} (d:{depth:.1f})"
                
                # Draw main rectangle
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                
                # Draw text background bar
                (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                text_y = max(th + 8, y1 - 4)
                cv2.rectangle(frame, (x1, text_y - th - 6), (x1 + tw + 4, text_y + baseline), color, cv2.FILLED)
                cv2.putText(frame, text, (x1 + 2, text_y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
            
            # Display prominent red hazard flashing banner at the top of the GUI
            if has_active_hazard:
                cv2.rectangle(frame, (0, 0), (w, 55), (0, 0, 255), cv2.FILLED)
                banner_text = f"WARNING: {', '.join(active_hazards).upper()} AHEAD!"
                (btw, bth), _ = cv2.getTextSize(banner_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                bx = max(10, int((w - btw) / 2))
                cv2.putText(frame, banner_text, (bx, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Render current frame with overlays in the OpenCV window
            cv2.imshow("Echo-Gnosis Live GUI", frame)
            
            # Wait for 30ms to maintain ~30 FPS frame display
            key = cv2.waitKey(30) & 0xFF
            if key == ord('q') or key == 27: # 'q' or Escape key
                logger.info("User requested exit.")
                running = False
                break
            
            # Gracefully handle if user clicks window 'X' button
            try:
                if cv2.getWindowProperty("Echo-Gnosis Live GUI", cv2.WND_PROP_VISIBLE) < 1:
                    logger.info("GUI window closed by user.")
                    running = False
                    break
            except cv2.error:
                break
                
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received.")
    except Exception as e:
        logger.critical(f"Fatal error in main orchestrator: {e}", exc_info=True)
    finally:
        logger.info("Shutting down and cleaning resources...")
        running = False
        
        # Stop background analysis thread
        if analysis_thread and analysis_thread.is_alive():
            analysis_thread.join(timeout=2.0)
            
        # Stop generative synthesizer
        if sonic_map:
            sonic_map.stop()
            
        # Stop Piper TTS alerter
        if piper_alerts:
            piper_alerts.stop()
            
        # Release Camera
        if cap and cap.isOpened():
            cap.release()
            logger.info("Webcam released.")
            
        cv2.destroyAllWindows()
        logger.info("Echo-Gnosis clean exit completed.")

if __name__ == "__main__":
    main()
