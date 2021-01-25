# This script should be cross platform
from subprocess import run
from pathlib import Path
import os

ffmpeg = "C://local/ffmpeg-4.3.1-2021-01-01-full_build/bin/ffmpeg.exe"

data_root = Path("data/v1")
videos = (data_root / "output").glob("**/labeled.mp4")
for video in videos:
    print(f"running for {video}")
    if (video.parent / "output.mp4").exists():
        os.remove(video.parent / "output.mp4")
    run(
        (
            f"{ffmpeg} -i {video} -vcodec libx264 -crf 24 " f"{video.parent}/output.mp4"
        ).split()
    )
