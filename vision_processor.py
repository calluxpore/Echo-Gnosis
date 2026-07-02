import cv2
import base64
import json
import logging
import ollama
from prompts import SYSTEM_PROMPT

logger = logging.getLogger("EchoGnosis.Vision")

class VisionAgent:
    def __init__(self):
        logger.info("VisionAgent initialized.")

    def analyze_frame(self, frame):
        """Encodes frame to base64, sends to Ollama, and parses structured JSON response."""
        if frame is None:
            return None
        
        try:
            # Resize to 512x512 for faster Ollama vision processing
            resized_frame = cv2.resize(frame, (512, 512))
            
            # Encode frame to JPEG
            _, buffer = cv2.imencode('.jpg', resized_frame)
            # Encode to base64 string
            b64_string = base64.b64encode(buffer).decode('utf-8')
            
            logger.info("Sending frame to Ollama (llava-phi3)...")
            response = ollama.chat(
                model='llava-phi3',
                messages=[
                    {
                        'role': 'system',
                        'content': SYSTEM_PROMPT
                    },
                    {
                        'role': 'user',
                        'content': 'Analyze this frame and return the JSON array of objects.',
                        'images': [b64_string]
                    }
                ],
                options={
                    'temperature': 0.0
                }
            )
            
            content = response.get('message', {}).get('content', '').strip()
            logger.debug(f"Ollama raw response content: {content}")
            
            parsed_json = self._clean_and_parse_json(content)
            if isinstance(parsed_json, list):
                for obj in parsed_json:
                    if isinstance(obj, dict):
                        # Map x (0.0 to 1.0) -> position_x (-1.0 to 1.0)
                        if 'x' in obj:
                            try:
                                x_val = float(obj.get('x', 0.5))
                                obj['position_x'] = (x_val * 2.0) - 1.0
                            except (ValueError, TypeError):
                                obj['position_x'] = 0.0
                        # Map y (0.0 to 1.0) -> position_y (1.0 to -1.0)
                        if 'y' in obj:
                            try:
                                y_val = float(obj.get('y', 0.5))
                                obj['position_y'] = 1.0 - (y_val * 2.0)
                            except (ValueError, TypeError):
                                obj['position_y'] = 0.0
            return parsed_json
            
        except Exception as e:
            logger.error(f"Error during Ollama frame analysis: {e}", exc_info=True)
            return None

    def _clean_and_parse_json(self, text):
        """Clean codeblock markups or leading/trailing text and parse JSON array."""
        if not text:
            return None
        
        # Try direct parsing first
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        
        # Clean markdown code block wraps
        cleaned = text
        if cleaned.startswith("```"):
            # Strip lines starting with ```
            lines = cleaned.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            cleaned = "\n".join(lines).strip()
            
            try:
                return json.loads(cleaned)
            except json.JSONDecodeError:
                pass

        # Try searching for the outer-most JSON array boundaries [ ]
        start_idx = cleaned.find('[')
        end_idx = cleaned.rfind(']')
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            array_content = cleaned[start_idx:end_idx + 1]
            try:
                return json.loads(array_content)
            except json.JSONDecodeError as e:
                logger.warning(f"Failed parsing subarray substring: {e}")
                
        logger.warning(f"Could not parse response as JSON: {text}")
        return None
