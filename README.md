# counting frames

```bash
# watch the video, press q to quit
python main.py watch data/input/spear.mkv
python main.py watch data/input/spear.mkv --crop

# take 30 seconds of frames into images for preprocessing
python main.py prepare-samples data/input/spear.mkv data/output/spear_cropped/
python main.py prepare-samples data/input/polearm.mkv data/output/polearm_cropped/

python cluster.py .\data\output\spear_cropped\ data/output/spear_clustered

python .\main.py calculate-diff .\data\output\spear_cropped\ .\data\output\spear_diff 547 589

.\classifier.py .\data\labeled\spear\ data/model/spear
```

Made a few templates based on a small region in the cropped screenshot. I will
run a template matching algorithm, and then pass the features into a multi-class
classifier.

```bash
python .\main.py train-classifier .\data\labeled\spear\ data/model/spear_logistic .\data\templates\stab.png .\data\templates\swing.png
```
