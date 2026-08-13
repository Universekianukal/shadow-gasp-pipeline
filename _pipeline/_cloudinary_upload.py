"""Upload final.mp4 to Cloudinary as a permanent asset so the Telegram bot
can send back a direct, no-login-required download link alongside the Drive
link. Unlike _fb_ig_upload.py's Cloudinary usage (a temporary staging asset
deleted right after Instagram finishes processing it), this upload is never
deleted -- it's meant to persist as a standing download link.

Credentials: CLOUDINARY_CLOUD_NAME/API_KEY/API_SECRET (same account/vars as
_fb_ig_upload.py and everydayhypehq/scripts/upload_to_cloudinary.py).
"""
import os
import sys

VIDEO_PATH = "final.mp4"


def main():
    if not os.path.isfile(VIDEO_PATH):
        print(f"{VIDEO_PATH} not found", file=sys.stderr)
        sys.exit(1)

    import cloudinary
    import cloudinary.uploader

    cloudinary.config(
        cloud_name=os.environ["CLOUDINARY_CLOUD_NAME"],
        api_key=os.environ["CLOUDINARY_API_KEY"],
        api_secret=os.environ["CLOUDINARY_API_SECRET"],
    )

    public_id = sys.argv[1] if len(sys.argv) > 1 else "shadow_gasp_video"
    resp = cloudinary.uploader.upload_large(
        VIDEO_PATH,
        resource_type="video",
        public_id=public_id,
        overwrite=True,
    )
    print(f"cloudinary_link={resp['secure_url']}")


if __name__ == "__main__":
    main()
