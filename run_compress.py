# This script should be cross platform
from subprocess import run
from pathlib import Path

ffmpeg = "C://local/ffmpeg-4.3.1-2021-01-01-full_build/bin/ffmpeg.exe"

data_root = Path("data/v1")
videos = (data_root / "output").glob("**/*.mp4")
for video in videos:
    print(f"running for {video}")
    run(
        (
            f"{ffmpeg} -i {video} -vcodec libx265 -crf 28 " f"{video.parent}/output.mp4"
        ).split()
    )
