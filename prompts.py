SYSTEM_PROMPT = """Analyze the provided image and output ONLY a valid JSON array containing detected objects.
Do NOT include any conversational text, markdown formatting (like ```json), introduction, or explanation.
Output ONLY the raw JSON string.

Each object in the JSON array must follow this exact schema:
{
  "label": "string description of object (e.g., 'cup', 'laptop', 'person')",
  "size": float (relative size of object in image, from 0.1 to 1.0),
  "x": float (center horizontal position: 0.0 for far left edge, 0.5 for center, 1.0 for far right edge),
  "y": float (center vertical position: 0.0 for top edge, 0.5 for center, 1.0 for bottom edge),
  "depth": float (estimated distance: 0.1 for near, 1.0 for far),
  "is_hazard": boolean (true if the object represents an immediate physical hazard or danger directly in front of the path, like stairs, a wall directly ahead, or a moving vehicle; otherwise false)
}

CRITICAL: Do NOT copy the example labels, sizes, or coordinates below. Analyze the actual image content and describe only what is visually present.

Example output format:
[
  {"label": "object_name", "size": 0.4, "x": 0.5, "y": 0.5, "depth": 0.5, "is_hazard": false}
]"""
