# This script should be cross platform
from subprocess import run
from pathlib import Path

python = "venv/Scripts/python.exe"

data_root = Path("data/v1")
videos = (data_root / "input").glob("*.mkv")
for video in videos:
    print(f"running for {video}")
    cmd = (
        f"{python} main.py find-template"
        f" {video} {data_root}/templates/name.png --relative 932"
    )
    print(cmd)
    res = run(cmd.split(), capture_output=True, check=True)
    offset = int(res.stdout.decode())
    cmd = (
        f"{python} main.py evaluate "
        f"{video} {data_root}/output/{video.name.split('.')[0]} "
        f"{data_root}/model/logistic_sliding {data_root}/templates "
        f"--offset-x {-175 + offset} --window 4 --batch-size 300"
    )
    print(cmd)
    run(cmd.split(), check=True)
