# counting frames

```bash
# watch the video, press q to quit
python main.py watch data/input/spear.mkv
python main.py watch data/input/spear.mkv --crop

# take 30 seconds of frames into images for preprocessing
python main.py prepare-samples data/input/spear.mkv data/output/spear_cropped/
python main.py prepare-samples data/input/polearm.mkv data/output/polearm_cropped/
```
