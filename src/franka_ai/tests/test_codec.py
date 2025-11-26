import os
import sys


"""
Run the code: 

python src/franka_ai/tests/test_codec.py
"""


def main():

    VIDEO = "/workspace/data/today_data/today_outliers/videos/chunk-000/observation.images.front_cam1/episode_000000.mp4"
    # VIDEO = "/home/leonardo/Documents/Coding/franka_AI/data/today_data/today_outliers/videos/chunk-000/observation.images.front_cam1/episode_000000.mp4"

    print("\n=== TEST 0: Check file exists ===")
    print("Exists:", os.path.exists(VIDEO))
    if not os.path.exists(VIDEO):
        print("❌ FILE NOT FOUND — wrong path")
        sys.exit(1)

    print("\n=== TEST 1: ffmpeg (shell) ===")
    import subprocess
    try:
        out = subprocess.check_output(["ffmpeg", "-i", VIDEO], stderr=subprocess.STDOUT)
        print("ffmpeg OK")
    except subprocess.CalledProcessError as e:
        print("ffmpeg output:")
        print(e.output.decode("utf-8"))
        print("❌ ffmpeg CANNOT read file")

    print("\n=== TEST 2: PyAV ===")
    try:
        import av
        container = av.open(VIDEO)
        print("PyAV OK — streams:", container.streams.video)
    except Exception as e:
        print("❌ PyAV ERROR:", e)

    print("\n=== TEST 3: TorchCodec ===")
    try:
        from torchcodec.decoders import VideoDecoder
        dec = VideoDecoder(VIDEO)
        frame = dec.read()
        print("TorchCodec OK — frame shape:", frame.shape)
    except Exception as e:
        print("❌ TorchCodec ERROR:", e)


if __name__ == "__main__":
    main()