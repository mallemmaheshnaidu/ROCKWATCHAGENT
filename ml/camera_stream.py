"""
camera_stream.py - Dynamic Multi-Frame Pit Surveillance Camera Renderer.

Generates dynamic, realistic surveillance frames for open-pit sectors:
- Realistic geological rock face textures with bench terraces, tension cracks, and dust drift.
- Real-time particle motion simulation: rolling gravel, falling detachment blocks, and dust plumes.
- Dynamic optical flow vectors and AI bounding boxes with tracking IDs and confidence metrics.
- Industrial CCTV HUD watermark (Camera ID, UTC timestamp, FPS, Optical zoom, PTZ coordinates).
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import Dict, Any, List, Tuple


class DynamicPitCameraStream:
    """
    Renders dynamic surveillance video frames representing live pit cameras.
    """

    def __init__(self, width: int = 480, height: int = 320):
        self.w = width
        self.h = height

    def render_frame(
        self,
        zone_id: str,
        t_hour: float,
        rockfall_intensity: float = 0.0,
        crack_width_cm: float = 0.0,
        detachment_detected: bool = False,
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Renders a dynamic CCTV frame for the sector at simulation time t_hour.
        Returns: (numpy_image_array, detected_bounding_boxes)
        """
        # Base background: layered rock face texture
        np.random.seed(int((t_hour % 10.0) * 100))
        img = Image.new("RGB", (self.w, self.h), color=(38, 34, 30))
        draw = ImageDraw.Draw(img)

        # 1. Bench terraces / rock strata bands
        for y_bench in range(40, self.h, 70):
            # Horizontal bench safety berm
            draw.line([(0, y_bench), (self.w, y_bench)], fill=(75, 70, 62), width=3)
            # Geological joint striations (diagonal fissures)
            for x_stri in range(20, self.w, 45):
                draw.line(
                    [(x_stri, y_bench), (x_stri + 35, y_bench + 65)],
                    fill=(50, 46, 40),
                    width=2,
                )

        # 2. Dynamic atmospheric dust particles / light wind drift
        for _ in range(35):
            px = np.random.randint(0, self.w)
            py = np.random.randint(0, self.h)
            radius = np.random.randint(1, 3)
            draw.ellipse(
                [(px, py), (px + radius, py + radius)],
                fill=(140, 130, 115, 60),
            )

        bounding_boxes = []

        # 3. Dynamic Tension Crack (widens as scenario progresses)
        crack_w_px = max(1, int(crack_width_cm * 1.5))
        if crack_width_cm > 1.0:
            crack_x = int(self.w * 0.42)
            crack_y_start = 55
            crack_y_end = min(self.h - 50, 55 + int(crack_width_cm * 12))
            
            # Draw jagged tension crack
            points = [(crack_x, crack_y_start)]
            curr_y = crack_y_start
            curr_x = crack_x
            while curr_y < crack_y_end:
                curr_y += 18
                curr_x += np.random.randint(-4, 5)
                points.append((curr_x, curr_y))

            draw.line(points, fill=(15, 12, 10), width=crack_w_px)

            # Crack AI Detection Bounding Box
            bbox_crack = {
                "label": f"CRACK_DILATION [{crack_width_cm:.1f}cm]",
                "xmin": crack_x - 20,
                "ymin": crack_y_start - 10,
                "xmax": crack_x + 35,
                "ymax": crack_y_end + 10,
                "confidence": min(0.98, 0.72 + crack_width_cm * 0.02),
                "type": "TENSION_CRACK",
            }
            bounding_boxes.append(bbox_crack)

        # 4. Moving Rock Detachment Debris & Gravitational Cascade
        if detachment_detected or rockfall_intensity > 0.15:
            # Phase of falling rock depends on fractional hour
            cycle = (t_hour * 8.0) % 1.0  # Motion cycle
            rock_origin_x = int(self.w * 0.48)
            rock_origin_y = int(self.h * 0.35)

            # Falling boulder trajectory
            fall_y = int(rock_origin_y + cycle * 120)
            fall_x = int(rock_origin_x + np.sin(cycle * np.pi) * 25)

            # Draw tumbling boulder with shadow
            draw.ellipse([(fall_x - 12, fall_y - 10), (fall_x + 12, fall_y + 10)], fill=(85, 78, 68))
            draw.ellipse([(fall_x - 8, fall_y - 6), (fall_x + 8, fall_y + 6)], fill=(110, 102, 90))

            # Optical flow velocity vector trail
            draw.line([(fall_x, fall_y - 18), (fall_x, fall_y)], fill=(239, 68, 68), width=2)
            draw.polygon([(fall_x, fall_y + 4), (fall_x - 4, fall_y - 2), (fall_x + 4, fall_y - 2)], fill=(239, 68, 68))

            # Small trailing gravel particles
            for i in range(12):
                gx = fall_x + np.random.randint(-25, 25)
                gy = max(0, fall_y - np.random.randint(5, 45))
                draw.rectangle([(gx, gy), (gx + 3, gy + 3)], fill=(95, 88, 76))

            # AI Rockfall Bounding Box
            bbox_rock = {
                "label": f"DETACHMENT_CASCADE [{fall_y*0.05:.1f} m/s]",
                "xmin": max(0, fall_x - 30),
                "ymin": max(0, fall_y - 35),
                "xmax": min(self.w - 1, fall_x + 30),
                "ymax": min(self.h - 1, fall_y + 20),
                "confidence": round(0.88 + rockfall_intensity * 0.10, 2),
                "type": "ROCKFALL_DEBRIS",
            }
            bounding_boxes.append(bbox_rock)

        # 5. Industrial CCTV HUD Overlay Watermark
        draw.rectangle([(0, 0), (self.w, 24)], fill=(10, 14, 22, 180))
        draw.rectangle([(0, self.h - 22), (self.w, self.h)], fill=(10, 14, 22, 180))

        # Camera Header text
        cam_id = zone_id.replace("ZONE_", "CAM-")
        t_sec = int(t_hour * 3600)
        hrs = t_sec // 3600
        mins = (t_sec % 3600) // 60
        secs = t_sec % 60
        time_str = f"T+00:{hrs:02d}:{mins:02d}:{secs:02d} UTC"
        hud_text_top = f"● REC | {cam_id} | {time_str} | 25.0 FPS | HD PTZ"
        draw.text((10, 5), hud_text_top, fill=(241, 245, 249))

        hud_text_bot = f"SENSOR FUSION ACTIVE | OPTICAL DETECTOR v2.4 | TARGET: {zone_id}"
        draw.text((10, self.h - 18), hud_text_bot, fill=(148, 163, 184))

        return np.array(img), bounding_boxes
