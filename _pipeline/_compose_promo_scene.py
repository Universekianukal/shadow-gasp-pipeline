"""Composite the REAL NORJAK cover (legible title, since it's the actual
rendered PDF page -- not AI text) into the PixArt-generated noir desk scene,
with a drop shadow and slight perspective tilt so it reads as a physical
object sitting in the scene, not a pasted sticker.

The scene (_gen_promo_pixart.py) was deliberately prompted with an EMPTY
rectangular gap in the center-left third -- this script places the cover
into that same region rather than guessing a layout blind.

Usage: python _compose_promo_scene.py
Reads ./norjak_promo_scene.jpeg + db-cooper-short/comic/panels/cover.jpg
Writes ./norjak_fb_post.jpeg
"""
import os

from PIL import Image, ImageDraw, ImageFilter

HERE = os.path.dirname(os.path.abspath(__file__))
SCENE_PATH = os.path.join(HERE, "norjak_promo_scene.jpeg")
COVER_PATH = os.path.join(HERE, "..", "db-cooper-short", "comic", "panels", "cover.jpg")
OUT_PATH = os.path.join(HERE, "norjak_fb_post.jpeg")


def build():
    scene = Image.open(SCENE_PATH).convert("RGB")
    cover = Image.open(COVER_PATH).convert("RGB")
    W, H = scene.size

    # Cover sized to ~46% of scene height, placed in the empty gap the scene
    # was prompted to leave in the center-left third.
    target_h = int(H * 0.80)
    scale = target_h / cover.height
    cover_r = cover.resize((int(cover.width * scale), target_h), Image.LANCZOS)
    cw, ch = cover_r.size

    x = int(W * 0.30) - cw // 2
    y = (H - ch) // 2

    # Slight counter-clockwise tilt so it reads as resting on the desk, not
    # floating flat-on in front of the camera.
    angle = -6
    cover_rot = cover_r.rotate(angle, expand=True, resample=Image.BICUBIC)

    # Soft drop shadow: a blurred black copy of the rotated cover's alpha
    # shape, offset down-right, composited before the cover itself.
    shadow = Image.new("RGBA", scene.size, (0, 0, 0, 0))
    shadow_shape = Image.new("L", cover_rot.size, 0)
    ImageDraw.Draw(shadow_shape).rectangle([0, 0, cover_rot.size[0], cover_rot.size[1]], fill=140)
    shadow_layer = Image.new("RGBA", cover_rot.size, (0, 0, 0, 255))
    shadow_layer.putalpha(shadow_shape)
    shadow.paste(shadow_layer, (x + 22, y + 26), shadow_layer)
    shadow = shadow.filter(ImageFilter.GaussianBlur(18))

    canvas = scene.convert("RGBA")
    canvas.alpha_composite(shadow)
    canvas.paste(cover_rot, (x, y))
    canvas.convert("RGB").save(OUT_PATH, quality=95)
    print(f"Saved -> {OUT_PATH}")


if __name__ == "__main__":
    build()
