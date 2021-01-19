from subprocess import run
import shutil
from pathlib import Path
import os
import json

shutil.rmtree("report/public/data", ignore_errors=True)
shutil.copytree("data/v1/output", "report/public/data")
for path in Path("report/public/data").glob("**/*.mp4"):
    os.remove(path)

manifest = []
for path in Path("report/public/data").glob("*"):
    manifest.append(path.name)
Path("report/public/data/manifest.json").write_text(json.dumps(manifest, indent=2))


run(
    "gsutil -m rsync -r data/v1/ gs://geospiza/stab-swing-counter/v1/".split(),
    shell=True,
)
