"""
generate_scenario_videos.py - Generates realistic open-pit mine CCTV surveillance MP4 videos.

Generates multi-front normal operational feeds and progressive rockfall scenario feeds:
1. cam01_north_wall_normal.mp4: North Highwall with bench terraces and passing haul truck.
2. cam02_south_crest_normal.mp4: South Crest overburden slope with crest berm and bulldozer.
3. cam03_ramp3_normal.mp4: Main Haulage switchback ramp with ascending heavy haul truck.
4. cam04_haul_road_normal.mp4: West Pit access road with water tanker truck and light vehicle.
5. mild_rockfall.mp4: Crest tension crack + tumbling pebbles + yellow AI tracking bounding boxes.
6. moderate_rockfall.mp4: Widened scarp fissure + sliding rock slabs + orange AI bounding boxes + dust plume.
7. severe_rockfall.mp4: Major bench collapse + cascading boulders + red AI bounding boxes + emergency alert HUD.
"""

import os
import shutil
import subprocess
import numpy as np
from PIL import Image, ImageDraw

VIDEOS_DIR = "data/videos"
os.makedirs(VIDEOS_DIR, exist_ok=True)
TMP_FRAMES = os.path.join(VIDEOS_DIR, "tmp_frames")


def draw_highwall_base(draw: ImageDraw.Draw, w: int, h: int, seed: int = 42):
    """Draws stratified rock highwall texture."""
    np.random.seed(seed)
    strata_colors = [(52, 46, 38), (44, 40, 34), (58, 52, 44), (40, 36, 30)]
    for y in range(0, h, 20):
        c = strata_colors[(y // 20) % len(strata_colors)]
        draw.rectangle([(0, y), (w, y + 20)], fill=c)

    # Safety berm lines
    draw.line([(0, int(h * 0.28)), (w, int(h * 0.28))], fill=(88, 80, 70), width=6)
    draw.line([(0, int(h * 0.60)), (w, int(h * 0.60))], fill=(82, 75, 65), width=6)
    draw.line([(0, int(h * 0.90)), (w, int(h * 0.90))], fill=(75, 68, 60), width=8)

    # Joint sets
    for x in range(25, w, 40):
        offset = np.random.randint(-15, 15)
        draw.line([(x, 0), (x + 35 + offset, h)], fill=(28, 24, 20), width=2)


def render_cam01_north_wall(out_path: str, n_frames: int = 45):
    """CAM-01: North Highwall Bench 4 normal operations."""
    os.makedirs(TMP_FRAMES, exist_ok=True)
    w, h = 640, 360

    for i in range(n_frames):
        img = Image.new("RGB", (w, h), color=(44, 40, 34))
        draw = ImageDraw.Draw(img)
        draw_highwall_base(draw, w, h, seed=42)

        # Ambient dust
        np.random.seed(i * 10)
        for _ in range(35):
            dx = (np.random.randint(0, w) + i * 4) % w
            dy = np.random.randint(0, h)
            draw.ellipse([(dx, dy), (dx + 2, dy + 2)], fill=(160, 150, 135))

        # Haul truck on lower bench
        truck_x = int((i / n_frames) * (w + 120) - 60)
        truck_y = int(h * 0.58) - 16
        if -50 <= truck_x <= w + 50:
            draw.rectangle([(truck_x, truck_y), (truck_x + 36, truck_y + 16)], fill=(234, 179, 8))
            draw.rectangle([(truck_x + 24, truck_y - 6), (truck_x + 36, truck_y + 10)], fill=(202, 138, 4))
            draw.ellipse([(truck_x + 4, truck_y + 12), (truck_x + 12, truck_y + 20)], fill=(15, 23, 42))
            draw.ellipse([(truck_x + 24, truck_y + 12), (truck_x + 32, truck_y + 20)], fill=(15, 23, 42))

        # CCTV HUD
        draw.rectangle([(0, 0), (w, 24)], fill=(10, 15, 26))
        draw.rectangle([(0, h - 22), (w, h)], fill=(10, 15, 26))
        draw.text((12, 5), f"● REC | CAM-01 [NORTH HIGHWALL BENCH 4] | 2026-11-15 12:00:{i:02d} | 25 FPS | HD PTZ", fill=(56, 189, 248))
        draw.text((12, h - 17), "STATUS: NORMAL OPERATIONS | HIGHWALL BENCH STABLE | TARP: LEVEL 1 GREEN", fill=(52, 211, 153))

        img.save(os.path.join(TMP_FRAMES, f"frame_{i:03d}.png"))

    subprocess.run([
        "ffmpeg", "-y", "-framerate", "15",
        "-i", os.path.join(TMP_FRAMES, "frame_%03d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", out_path
    ], capture_output=True, check=True)
    shutil.rmtree(TMP_FRAMES, ignore_errors=True)
    print(f"[✓] Created: {out_path}")


def render_cam02_south_crest(out_path: str, n_frames: int = 45):
    """CAM-02: South Crest Overburden Slope with safety crest and bulldozer."""
    os.makedirs(TMP_FRAMES, exist_ok=True)
    w, h = 640, 360

    for i in range(n_frames):
        img = Image.new("RGB", (w, h), color=(38, 34, 28))
        draw = ImageDraw.Draw(img)

        # Sky at upper portion
        draw.rectangle([(0, 0), (w, int(h * 0.22))], fill=(70, 85, 105))
        # Crest ridge line
        draw.polygon([(0, int(h * 0.22)), (w, int(h * 0.18)), (w, h), (0, h)], fill=(55, 48, 38))
        # Overburden soil layer
        draw.polygon([(0, int(h * 0.22)), (w, int(h * 0.18)), (w, int(h * 0.45)), (0, int(h * 0.50))], fill=(78, 64, 48))
        # Weathered rock layer below
        draw.polygon([(0, int(h * 0.50)), (w, int(h * 0.45)), (w, h), (0, h)], fill=(45, 40, 32))

        # Dozer grading along crest berm
        dozer_x = int(w * 0.65 - (i / n_frames) * 70)
        dozer_y = int(h * 0.19)
        # Dozer body
        draw.rectangle([(dozer_x, dozer_y - 12), (dozer_x + 30, dozer_y + 8)], fill=(245, 158, 11))
        # Blade
        draw.rectangle([(dozer_x - 6, dozer_y - 4), (dozer_x, dozer_y + 8)], fill=(100, 100, 100))
        # Track
        draw.rectangle([(dozer_x - 2, dozer_y + 8), (dozer_x + 32, dozer_y + 14)], fill=(20, 20, 20))

        # Ambient dust
        np.random.seed(i * 12)
        for _ in range(25):
            dx = (np.random.randint(0, w) + i * 3) % w
            dy = np.random.randint(int(h * 0.2), h)
            draw.ellipse([(dx, dy), (dx + 2, dy + 2)], fill=(150, 140, 125))

        # CCTV HUD
        draw.rectangle([(0, 0), (w, 24)], fill=(10, 15, 26))
        draw.rectangle([(0, h - 22), (w, h)], fill=(10, 15, 26))
        draw.text((12, 5), f"● REC | CAM-02 [SOUTH CREST OVERBURDEN] | 2026-11-15 12:00:{i:02d} | 25 FPS | HD PTZ", fill=(56, 189, 248))
        draw.text((12, h - 17), "STATUS: NORMAL OPERATIONS | CREST BERM SOUND | TARP: LEVEL 1 GREEN", fill=(52, 211, 153))

        img.save(os.path.join(TMP_FRAMES, f"frame_{i:03d}.png"))

    subprocess.run([
        "ffmpeg", "-y", "-framerate", "15",
        "-i", os.path.join(TMP_FRAMES, "frame_%03d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", out_path
    ], capture_output=True, check=True)
    shutil.rmtree(TMP_FRAMES, ignore_errors=True)
    print(f"[✓] Created: {out_path}")


def render_cam03_ramp3(out_path: str, n_frames: int = 45):
    """CAM-03: Main Haulage Switchback Ramp 3 with ascending heavy haul truck."""
    os.makedirs(TMP_FRAMES, exist_ok=True)
    w, h = 640, 360

    for i in range(n_frames):
        img = Image.new("RGB", (w, h), color=(42, 38, 32))
        draw = ImageDraw.Draw(img)

        # Rock face backdrop
        draw_highwall_base(draw, w, h, seed=77)

        # Diagonal Ramp Cut (from bottom left (0, 310) to top right (640, 120))
        ramp_poly = [(0, 330), (w, 140), (w, 100), (0, 290)]
        draw.polygon(ramp_poly, fill=(75, 68, 58))
        # Safety windrow berm along ramp outer edge
        draw.line([(0, 330), (w, 140)], fill=(110, 100, 85), width=8)

        # Loaded Haul Truck ascending ramp (moving upwards from left to right)
        frac = (i / n_frames)
        truck_x = int(frac * (w * 0.8) + 40)
        # y interpolates from 300 to 115
        truck_y = int(300 - frac * 185)

        # Haul truck
        draw.rectangle([(truck_x - 18, truck_y - 20), (truck_x + 22, truck_y + 4)], fill=(234, 179, 8))
        # Loaded ore body in truck bed
        draw.ellipse([(truck_x - 16, truck_y - 26), (truck_x + 12, truck_y - 14)], fill=(65, 55, 45))
        # Heavy mining tires
        draw.ellipse([(truck_x - 14, truck_y + 2), (truck_x - 4, truck_y + 12)], fill=(15, 23, 42))
        draw.ellipse([(truck_x + 8, truck_y + 2), (truck_x + 18, truck_y + 12)], fill=(15, 23, 42))

        # Dust puff behind truck
        for d in range(12):
            dx = truck_x - 25 - np.random.randint(0, 25)
            dy = truck_y + np.random.randint(-5, 10)
            draw.ellipse([(dx, dy), (dx + 4, dy + 4)], fill=(140, 130, 115))

        # CCTV HUD
        draw.rectangle([(0, 0), (w, 24)], fill=(10, 15, 26))
        draw.rectangle([(0, h - 22), (w, h)], fill=(10, 15, 26))
        draw.text((12, 5), f"● REC | CAM-03 [MAIN HAULAGE SWITCHBACK RAMP 3] | 2026-11-15 12:00:{i:02d} | 25 FPS", fill=(56, 189, 248))
        draw.text((12, h - 17), "STATUS: NORMAL OPERATIONS | HAULAGE FLOW NOMINAL (14.2 km/h) | TARP: LEVEL 1 GREEN", fill=(52, 211, 153))

        img.save(os.path.join(TMP_FRAMES, f"frame_{i:03d}.png"))

    subprocess.run([
        "ffmpeg", "-y", "-framerate", "15",
        "-i", os.path.join(TMP_FRAMES, "frame_%03d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", out_path
    ], capture_output=True, check=True)
    shutil.rmtree(TMP_FRAMES, ignore_errors=True)
    print(f"[✓] Created: {out_path}")


def render_cam04_haul_road(out_path: str, n_frames: int = 45):
    """CAM-04: West Pit Access Road with water tanker truck spraying to suppress dust."""
    os.makedirs(TMP_FRAMES, exist_ok=True)
    w, h = 640, 360

    for i in range(n_frames):
        img = Image.new("RGB", (w, h), color=(48, 42, 34))
        draw = ImageDraw.Draw(img)

        # Low bench backdrop
        draw_highwall_base(draw, w, h, seed=99)

        # Wide dual-lane flat haul road at bottom half
        draw.rectangle([(0, int(h * 0.48)), (w, h)], fill=(60, 54, 46))
        # Centerline marker berm
        draw.line([(0, int(h * 0.74)), (w, int(h * 0.74))], fill=(85, 78, 68), width=3)

        # Water spraying truck moving left to right
        tanker_x = int((i / n_frames) * (w + 100) - 50)
        tanker_y = int(h * 0.60)
        if -40 <= tanker_x <= w + 40:
            # Tanker cylinder
            draw.rectangle([(tanker_x - 20, tanker_y - 16), (tanker_x + 15, tanker_y + 4)], fill=(225, 29, 72))
            draw.rectangle([(tanker_x + 15, tanker_y - 8), (tanker_x + 26, tanker_y + 4)], fill=(255, 255, 255))
            # Spray fan behind
            for s in range(20):
                sx = tanker_x - 24 - np.random.randint(0, 35)
                sy = tanker_y + np.random.randint(-10, 10)
                draw.line([(tanker_x - 20, tanker_y), (sx, sy)], fill=(186, 230, 253), width=1)
            # Wheels
            draw.ellipse([(tanker_x - 15, tanker_y + 4), (tanker_x - 5, tanker_y + 14)], fill=(20, 20, 20))
            draw.ellipse([(tanker_x + 12, tanker_y + 4), (tanker_x + 22, tanker_y + 14)], fill=(20, 20, 20))

        # CCTV HUD
        draw.rectangle([(0, 0), (w, 24)], fill=(10, 15, 26))
        draw.rectangle([(0, h - 22), (w, h)], fill=(10, 15, 26))
        draw.text((12, 5), f"● REC | CAM-04 [WEST PIT ACCESS HAUL ROAD] | 2026-11-15 12:00:{i:02d} | 25 FPS", fill=(56, 189, 248))
        draw.text((12, h - 17), "STATUS: NORMAL OPERATIONS | WATER SPRAY DUST SUPPRESSION ACTIVE | TARP: LEVEL 1 GREEN", fill=(52, 211, 153))

        img.save(os.path.join(TMP_FRAMES, f"frame_{i:03d}.png"))

    subprocess.run([
        "ffmpeg", "-y", "-framerate", "15",
        "-i", os.path.join(TMP_FRAMES, "frame_%03d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", out_path
    ], capture_output=True, check=True)
    shutil.rmtree(TMP_FRAMES, ignore_errors=True)
    print(f"[✓] Created: {out_path}")


def render_mild_rockfall_video(out_path: str, n_frames: int = 45):
    """Renders mild slope ravelling: small pebbles rolling down, hairline crest tension crack, yellow bounding boxes."""
    os.makedirs(TMP_FRAMES, exist_ok=True)
    w, h = 640, 360

    for i in range(n_frames):
        img = Image.new("RGB", (w, h), color=(44, 40, 34))
        draw = ImageDraw.Draw(img)
        draw_highwall_base(draw, w, h, seed=42)

        # Hairline crest tension crack
        crack_x = int(w * 0.45)
        draw.line([(crack_x, 38), (crack_x + 10, 85), (crack_x + 4, 130)], fill=(15, 12, 10), width=3)
        draw.rectangle([crack_x - 14, 34, crack_x + 22, 134], outline="#f59e0b", width=2)
        draw.text((crack_x - 14, 18), "CRACK DILATION [2.4cm]", fill="#f59e0b")

        # Small loose pebbles and gravel tumbling down face
        t_frac = (i / n_frames)
        for p_idx, p_offset in enumerate([-18, 0, 20]):
            peb_y = int(90 + ((t_frac * 1.6 + p_idx * 0.28) % 1.0) * 125)
            peb_x = crack_x + p_offset + int(np.sin(peb_y * 0.08) * 10)
            draw.ellipse([(peb_x - 6, peb_y - 5), (peb_x + 6, peb_y + 5)], fill=(125, 115, 100))
            draw.line([(peb_x, peb_y - 10), (peb_x, peb_y)], fill="#f59e0b", width=2)
            # Bounding box around active pebble group
            if 105 < peb_y < 205:
                draw.rectangle([peb_x - 14, peb_y - 12, peb_x + 14, peb_y + 12], outline="#f59e0b", width=2)
                draw.text((peb_x - 18, peb_y - 24), "RAVELLING [1.8m/s | 92%]", fill="#f59e0b")

        # CCTV HUD
        draw.rectangle([(0, 0), (w, 24)], fill=(10, 15, 26))
        draw.rectangle([(0, h - 22), (w, h)], fill=(10, 15, 26))
        draw.text((12, 5), f"● REC | CAM-01 [NORTH HIGHWALL BENCH 4] | 2026-11-15 12:12:{i:02d} | 25 FPS", fill=(56, 189, 248))
        draw.text((12, h - 17), "⚠️ STATUS: MILD SLOPE RAVELLING | LOOSE GRAVEL DETECTED | TARP: LEVEL 2 YELLOW", fill="#fbbf24")

        img.save(os.path.join(TMP_FRAMES, f"frame_{i:03d}.png"))

    subprocess.run([
        "ffmpeg", "-y", "-framerate", "15",
        "-i", os.path.join(TMP_FRAMES, "frame_%03d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", out_path
    ], capture_output=True, check=True)
    shutil.rmtree(TMP_FRAMES, ignore_errors=True)
    print(f"[✓] Created: {out_path}")


def render_moderate_rockfall_video(out_path: str, n_frames: int = 45):
    """Renders moderate wedge dilation: widening fissure and sliding rock slabs with orange AI bounding boxes."""
    os.makedirs(TMP_FRAMES, exist_ok=True)
    w, h = 640, 360

    for i in range(n_frames):
        img = Image.new("RGB", (w, h), color=(44, 40, 34))
        draw = ImageDraw.Draw(img)
        draw_highwall_base(draw, w, h, seed=42)

        # Widened scarp fissure (8cm)
        crack_x = int(w * 0.44)
        draw.line([(crack_x, 30), (crack_x + 18, 90), (crack_x + 8, 165)], fill=(12, 8, 6), width=7)
        draw.rectangle([crack_x - 18, 24, crack_x + 32, 170], outline="#f97316", width=2)
        draw.text((crack_x - 18, 10), "SCARP EXTENSION [8.2cm]", fill="#f97316")

        # Sliding rock slabs and trailing dust plume
        t_frac = (i / n_frames)
        slab_y = int(105 + (t_frac ** 1.3) * 145)
        slab_x = crack_x + int(np.sin(t_frac * 2 * np.pi) * 16) + 14

        # Dust plume
        for d in range(30):
            dx = slab_x + np.random.randint(-35, 35)
            dy = max(0, slab_y - np.random.randint(5, 55))
            draw.ellipse([(dx, dy), (dx + 7, dy + 7)], fill=(148, 138, 122))

        # Tumbling rock slabs
        draw.polygon([(slab_x - 20, slab_y - 14), (slab_x + 22, slab_y - 10), (slab_x + 16, slab_y + 18), (slab_x - 18, slab_y + 14)], fill=(92, 84, 74))
        draw.line([(slab_x, slab_y - 24), (slab_x, slab_y)], fill="#f97316", width=3)

        # AI Detection Bounding Box
        draw.rectangle([slab_x - 28, slab_y - 22, slab_x + 30, slab_y + 24], outline="#f97316", width=2)
        draw.text((slab_x - 28, slab_y - 36), "SLAB DETACHMENT [4.2 m/s | CONF: 94%]", fill="#f97316")

        # CCTV HUD
        draw.rectangle([(0, 0), (w, 24)], fill=(10, 15, 26))
        draw.rectangle([(0, h - 22), (w, h)], fill=(10, 15, 26))
        draw.text((12, 5), f"● REC | CAM-01 [NORTH HIGHWALL BENCH 4] | 2026-11-15 12:18:{i:02d} | 25 FPS", fill=(56, 189, 248))
        draw.text((12, h - 17), "🟠 STATUS: MODERATE WEDGE DILATION | ROCK DETACHMENT | TARP: LEVEL 3 ORANGE", fill="#fb923c")

        img.save(os.path.join(TMP_FRAMES, f"frame_{i:03d}.png"))

    subprocess.run([
        "ffmpeg", "-y", "-framerate", "15",
        "-i", os.path.join(TMP_FRAMES, "frame_%03d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", out_path
    ], capture_output=True, check=True)
    shutil.rmtree(TMP_FRAMES, ignore_errors=True)
    print(f"[✓] Created: {out_path}")


def render_severe_rockfall_video(out_path: str, n_frames: int = 45):
    """Renders major bench collapse: tumbling boulder cascades with red AI bounding boxes and emergency banner."""
    os.makedirs(TMP_FRAMES, exist_ok=True)
    w, h = 640, 360

    for i in range(n_frames):
        img = Image.new("RGB", (w, h), color=(44, 40, 34))
        draw = ImageDraw.Draw(img)
        draw_highwall_base(draw, w, h, seed=42)

        # Massive structural tension chasm
        crack_x = int(w * 0.42)
        draw.line([(crack_x, 18), (crack_x + 26, 100), (crack_x + 14, 195)], fill=(10, 6, 4), width=12)
        draw.rectangle([crack_x - 24, 14, crack_x + 44, 200], outline="#ef4444", width=3)
        draw.text((crack_x - 24, 2), "CRITICAL SHEAR SCARP [18.5cm]", fill="#ef4444")

        # Heavy cascading boulders with motion blur
        t_frac = (i / n_frames)
        boulder_y = int(90 + (t_frac ** 1.6) * 195)
        boulder_x = crack_x + int(np.sin(t_frac * 3 * np.pi) * 30) + 18

        # Massive dust cloud
        for d in range(55):
            dx = boulder_x + np.random.randint(-60, 60)
            dy = max(0, boulder_y - np.random.randint(10, 85))
            draw.ellipse([(dx, dy), (dx + 12, dy + 12)], fill=(168, 155, 138))

        # Main falling boulder mass
        draw.ellipse([(boulder_x - 26, boulder_y - 20), (boulder_x + 26, boulder_y + 20)], fill=(75, 68, 58))
        draw.ellipse([(boulder_x - 18, boulder_y - 14), (boulder_x + 18, boulder_y + 14)], fill=(105, 96, 82))
        draw.line([(boulder_x, boulder_y - 38), (boulder_x, boulder_y)], fill="#ef4444", width=4)

        # Trailing secondary rock blocks
        for b_sub in range(3):
            sub_y = max(0, boulder_y - 38 - b_sub * 26)
            sub_x = boulder_x + (b_sub - 1) * 24
            draw.rectangle([(sub_x - 12, sub_y - 10), (sub_x + 12, sub_y + 10)], fill=(85, 78, 68))

        # AI Detection Bounding Box (Crimson Critical)
        draw.rectangle([boulder_x - 38, boulder_y - 32, boulder_x + 40, boulder_y + 34], outline="#ef4444", width=3)
        draw.text((boulder_x - 38, boulder_y - 48), "ROCKFALL CASCADE [8.4 m/s | CONF: 99%]", fill="#ef4444")

        # CCTV HUD Alert Banner
        draw.rectangle([(0, 0), (w, 24)], fill=(10, 15, 26))
        draw.rectangle([(0, h - 22), (w, h)], fill=(10, 15, 26))
        draw.text((12, 5), f"● REC | CAM-01 [NORTH HIGHWALL BENCH 4] | 2026-11-15 12:24:{i:02d} | 25 FPS", fill=(56, 189, 248))
        draw.text((12, h - 17), "🚨 CRITICAL ROCKFALL DETACHMENT IN PROGRESS | EVACUATION MANDATED | TARP: LEVEL 4 RED", fill="#f87171")

        img.save(os.path.join(TMP_FRAMES, f"frame_{i:03d}.png"))

    subprocess.run([
        "ffmpeg", "-y", "-framerate", "15",
        "-i", os.path.join(TMP_FRAMES, "frame_%03d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", out_path
    ], capture_output=True, check=True)
    shutil.rmtree(TMP_FRAMES, ignore_errors=True)
    print(f"[✓] Created: {out_path}")


if __name__ == "__main__":
    print("[*] Generating high-definition surveillance MP4 video suite...")
    # Multi-front normal operational CCTV streams
    render_cam01_north_wall(os.path.join(VIDEOS_DIR, "cam01_north_wall_normal.mp4"))
    render_cam02_south_crest(os.path.join(VIDEOS_DIR, "cam02_south_crest_normal.mp4"))
    render_cam03_ramp3(os.path.join(VIDEOS_DIR, "cam03_ramp3_normal.mp4"))
    render_cam04_haul_road(os.path.join(VIDEOS_DIR, "cam04_haul_road_normal.mp4"))
    # Legacy alias
    shutil.copyfile(os.path.join(VIDEOS_DIR, "cam01_north_wall_normal.mp4"), os.path.join(VIDEOS_DIR, "normal_ops.mp4"))
    
    # Progressive rockfall hazard scenarios
    render_mild_rockfall_video(os.path.join(VIDEOS_DIR, "mild_rockfall.mp4"))
    render_moderate_rockfall_video(os.path.join(VIDEOS_DIR, "moderate_rockfall.mp4"))
    render_severe_rockfall_video(os.path.join(VIDEOS_DIR, "severe_rockfall.mp4"))
    print("[✓] All multi-front and scenario surveillance video feeds generated successfully.")
