import os
import sys
import time
import argparse
import logging
import threading
import cv2
import numpy as np
from vision_processor import VisionAgent
from audio_synthesizer import SonicMap
from tts_alerter import PiperAlerts

try:
    import pyrealsense2 as rs
    HAS_REALSENSE = True
except ImportError:
    HAS_REALSENSE = False

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
        """Background thread executing Ollama analysis, updating detected_objects."""
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
        
        # 4. Initialize Camera (RealSense with WebCam fallback)
        use_realsense = False
        pipeline = None
        align = None
        depth_scale = 0.001
        cap = None

        if HAS_REALSENSE:
            try:
                logger.info("Attempting to initialize Intel RealSense D435 camera...")
                pipeline = rs.pipeline()
                config = rs.config()
                # Configure depth and color streams
                config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
                config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
                
                profile = pipeline.start(config)
                
                # Retrieve depth sensor and scale
                depth_sensor = profile.get_device().first_depth_sensor()
                depth_scale = depth_sensor.get_depth_scale()
                logger.info(f"Intel RealSense D435 initialized successfully. Depth scale: {depth_scale}")
                
                # Set up aligner
                align = rs.align(rs.stream.color)
                use_realsense = True
            except Exception as e:
                logger.warning(f"Failed to initialize Intel RealSense: {e}. Falling back to standard webcam.")
                use_realsense = False

        if not use_realsense:
            logger.info(f"Opening standard camera index {args.camera}...")
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
            depth_image = None
            depth_colormap = None
            frame = None
            collision_detected = False
            close_depth_m = 1.2

            if use_realsense:
                try:
                    frames = pipeline.wait_for_frames(timeout_ms=1000)
                    aligned_frames = align.process(frames)
                    
                    depth_frame = aligned_frames.get_depth_frame()
                    color_frame = aligned_frames.get_color_frame()
                    
                    if not depth_frame or not color_frame:
                        logger.warning("RealSense frame capture empty. Retrying...")
                        time.sleep(0.01)
                        continue
                    
                    depth_image = np.asanyarray(depth_frame.get_data())
                    frame = np.asanyarray(color_frame.get_data())
                except Exception as e:
                    logger.warning(f"Error capturing from RealSense: {e}. Retrying...")
                    time.sleep(0.01)
                    continue
            else:
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
                current_objects = [dict(obj) for obj in detected_objects]
                
            h, w, _ = frame.shape
            
            # Update object depth and is_hazard in real-time if depth data is available
            if use_realsense and depth_image is not None:
                # 1. Update existing objects with actual depth values
                for obj in current_objects:
                    pos_x = float(obj.get('position_x', 0.0))
                    pos_y = float(obj.get('position_y', 0.0))
                    size = float(obj.get('size', 0.2))
                    
                    # Convert to pixel coordinate bounding box
                    cx = int((pos_x + 1.0) / 2.0 * w)
                    cy = int((1.0 - pos_y) / 2.0 * h)
                    bw = int(size * w)
                    bh = int(size * h)
                    
                    x1 = max(0, int(cx - bw / 2.0))
                    y1 = max(0, int(cy - bh / 2.0))
                    x2 = min(w - 1, int(cx + bw / 2.0))
                    y2 = min(h - 1, int(cy + bh / 2.0))
                    
                    # Extract depth in bounding box
                    if x2 > x1 and y2 > y1:
                        bbox_depth = depth_image[y1:y2, x1:x2]
                        valid_depths = bbox_depth[bbox_depth > 0]
                        if len(valid_depths) > 0:
                            actual_depth_m = np.median(valid_depths) * depth_scale
                            # Map actual depth (0.3m to 4.0m) to synthesizer depth (0.1 to 1.0)
                            mapped_depth = max(0.1, min(1.0, 0.1 + (actual_depth_m - 0.3) * (0.9 / 3.7)))
                            obj['depth'] = mapped_depth
                            
                            # If the object is too close, mark it as hazard
                            if actual_depth_m < 1.5:
                                obj['is_hazard'] = True
                                
                # 2. Real-time Proximity / Collision Hazard Detection in front of user
                # Width: 35% to 65% of screen (middle sector)
                # Height: 40% to 95% of screen (usually ground and straight ahead)
                hx1, hx2 = int(w * 0.35), int(w * 0.65)
                hy1, hy2 = int(h * 0.40), int(h * 0.95)
                
                hazard_zone_depths = depth_image[hy1:hy2, hx1:hx2]
                # Filter valid depths within our safety range (e.g. 0.3m to 1.2m)
                min_safety_mm = 300
                max_safety_mm = 1200
                near_pixels = hazard_zone_depths[(hazard_zone_depths >= min_safety_mm) & (hazard_zone_depths <= max_safety_mm)]
                
                if len(near_pixels) > (hazard_zone_depths.size * 0.05): # more than 5% of the zone is blocked
                    collision_detected = True
                    close_depth_m = np.median(near_pixels) * depth_scale
                    mapped_collision_depth = max(0.1, min(1.0, 0.1 + (close_depth_m - 0.3) * (0.9 / 3.7)))
                    
                    # Create proximity hazard object
                    proximity_obj = {
                        'label': 'obstacle',
                        'position_x': 0.0, # dead center
                        'position_y': -0.2, # slightly low center
                        'depth': mapped_collision_depth,
                        'size': 0.5,
                        'is_hazard': True
                    }
                    current_objects.append(proximity_obj)

            # Update SonicMap and PiperAlerts at 30 FPS
            if sonic_map:
                sonic_map.update_objects(current_objects)
            if piper_alerts:
                piper_alerts.process_hazards(current_objects)

            # Render Bounding Boxes & Warning Banner
            has_active_hazard = False
            active_hazards = []
            
            for obj in current_objects:
                label = obj.get('label', 'object')
                pos_x = float(obj.get('position_x', 0.0))
                pos_y = float(obj.get('position_y', 0.0))
                depth = float(obj.get('depth', 0.5))
                size = float(obj.get('size', 0.2))
                is_hazard = bool(obj.get('is_hazard', False))
                
                # Convert back to physical depth for display (0.1-1.0 maps back to 0.3-4.0m)
                disp_depth_m = 0.3 + (depth - 0.1) * 4.11
                if not use_realsense:
                    # If not realsense, just display what Ollama estimated directly
                    disp_depth_m = depth
                
                # Calculate pixel coordinates from relative positions
                cx = int((pos_x + 1.0) / 2.0 * w)
                cy = int((1.0 - pos_y) / 2.0 * h)
                
                bw = int(size * w)
                bh = int(size * h)
                
                x1 = max(0, int(cx - bw / 2.0))
                y1 = max(0, int(cy - bh / 2.0))
                x2 = min(w - 1, int(cx + bw / 2.0))
                y2 = min(h - 1, int(cy + bh / 2.0))
                
                if is_hazard:
                    color = (0, 0, 255) # BGR Red
                    thickness = 3
                    has_active_hazard = True
                    active_hazards.append(label.lower())
                    unit = "m" if use_realsense else ""
                    text = f"HAZARD: {label.upper()} ({disp_depth_m:.1f}{unit})"
                else:
                    color = (0, 255, 0) # BGR Green
                    thickness = 2
                    unit = "m" if use_realsense else ""
                    text = f"{label} ({disp_depth_m:.1f}{unit})"
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                
                (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                text_y = max(th + 8, y1 - 4)
                cv2.rectangle(frame, (x1, text_y - th - 6), (x1 + tw + 4, text_y + baseline), color, cv2.FILLED)
                cv2.putText(frame, text, (x1 + 2, text_y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
            
            # Draw visual Hazard Detection Zone overlay
            if use_realsense and depth_image is not None:
                hz_color = (0, 0, 255) if collision_detected else (255, 255, 255)
                hz_thickness = 2 if collision_detected else 1
                cv2.rectangle(frame, (int(w * 0.35), int(h * 0.40)), (int(w * 0.65), int(h * 0.95)), hz_color, hz_thickness)
                hz_label = "HAZARD MONITOR ZONE"
                if collision_detected:
                    hz_label += f": OBSTACLE {close_depth_m:.1f}m!"
                cv2.putText(frame, hz_label, (int(w * 0.35), int(h * 0.38)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, hz_color, 1, cv2.LINE_AA)

            # Display prominent red hazard flashing banner at the top of the GUI
            if has_active_hazard:
                cv2.rectangle(frame, (0, 0), (w, 55), (0, 0, 255), cv2.FILLED)
                banner_text = f"WARNING: {', '.join(set(active_hazards)).upper()} AHEAD!"
                (btw, bth), _ = cv2.getTextSize(banner_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                bx = max(10, int((w - btw) / 2))
                cv2.putText(frame, banner_text, (bx, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Create side-by-side combined frame
            if use_realsense and depth_image is not None:
                # Map raw depth image to 8-bit color map for display
                depth_scaled = cv2.convertScaleAbs(depth_image, alpha=0.03)
                depth_colormap = cv2.applyColorMap(depth_scaled, cv2.COLORMAP_JET)
                combined_frame = np.hstack((frame, depth_colormap))
            else:
                # Create a placeholder depth map to keep UI layout identical
                placeholder_depth = np.zeros_like(frame)
                cv2.putText(placeholder_depth, "RealSense Not Detected", (40, 200),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.putText(placeholder_depth, "Using Fallback WebCam Feed", (40, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(placeholder_depth, "RealSense Depth Map Offline", (40, 280),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1, cv2.LINE_AA)
                
                # Draw the hazard zone box as disabled/dotted gray
                cv2.rectangle(placeholder_depth, (int(w * 0.35), int(h * 0.40)), (int(w * 0.65), int(h * 0.95)), (100, 100, 100), 1)
                combined_frame = np.hstack((frame, placeholder_depth))

            # Render combined frame
            cv2.imshow("Echo-Gnosis Live GUI", combined_frame)
            
            key = cv2.waitKey(30) & 0xFF
            if key == ord('q') or key == 27:
                logger.info("User requested exit.")
                running = False
                break
            
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
            
        # Release standard camera if opened
        if cap and cap.isOpened():
            cap.release()
            logger.info("Webcam released.")
            
        # Stop RealSense pipeline if active
        if use_realsense and pipeline:
            try:
                pipeline.stop()
                logger.info("RealSense pipeline stopped.")
            except Exception as e:
                logger.error(f"Error stopping RealSense pipeline: {e}")
            
        cv2.destroyAllWindows()
        logger.info("Echo-Gnosis clean exit completed.")

if __name__ == "__main__":
    main()
