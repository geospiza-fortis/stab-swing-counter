from subprocess import run
import shutil
from pathlib import Path
import os
import json

shutil.rmtree("report/public/trial", ignore_errors=True)
shutil.copytree("data/v1/output", "report/public/trial")
for path in Path("report/public/trial").glob("**/*.mp4"):
    os.remove(path)
for path in Path("report/public/trial").glob("**/probs_*.png"):
    os.remove(path)

manifest = []
for path in Path("report/public/trial").glob("*"):
    manifest.append(path.name)
Path("report/public/trial/manifest.json").write_text(json.dumps(manifest, indent=2))


run(
    "gsutil -m rsync -r data/v1/ gs://geospiza/stab-swing-counter/v1/".split(),
    shell=True,
)
