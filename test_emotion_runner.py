import os
import requests
import time

# ==============================
# CONFIG
# ==============================
API_URL = "http://127.0.0.1:8000/emotion"
TEST_IMAGE_DIR = "test_image"

# Fixed UUID so aggregation works
USER_ID = "11111111-1111-1111-1111-111111111111"

# Image naming format
IMAGE_PREFIX = "Image_"
IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png"]

# ==============================
# Send one image
# ==============================
def test_image(image_path):
    with open(image_path, "rb") as img:
        files = {
            "file": (os.path.basename(image_path), img, "image/jpeg")
        }
        data = {
            "user_id": USER_ID
        }

        response = requests.post(API_URL, files=files, data=data)

    print("\n------------------------------------")
    print(f"🖼️  Image Sent : {os.path.basename(image_path)}")

    if response.status_code == 200:
        result = response.json()
        print(f"🎭 Emotion    : {result.get('emotion')}")
        print(f"📊 Confidence : {result.get('confidence')}")
        print(f"💬 Server Msg : {result.get('server_message')}")
    else:
        print("❌ Request Failed")
        print(response.text)

# ==============================
# Main runner
# ==============================
def run():
    print("🔍 Starting FER test (10 images)...")

    for i in range(1, 11):
        img_name = f"{IMAGE_PREFIX}{i:02d}"
        image_path = None

        for ext in IMAGE_EXTENSIONS:
            candidate = os.path.join(TEST_IMAGE_DIR, img_name + ext)
            if os.path.exists(candidate):
                image_path = candidate
                break

        if not image_path:
            print(f"\n⚠️  Missing file: {img_name}.jpg/.png")
            continue

        test_image(image_path)

        # Optional delay to simulate real capture (comment out if not needed)
        # time.sleep(1)

    print("\n✅ Test completed.")

if __name__ == "__main__":
    run()
