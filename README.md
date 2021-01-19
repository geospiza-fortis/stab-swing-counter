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

python .\main.py evaluate-model .\data\output\spear_cropped\ .\data\output\spear_eval\ data/model/spear_logistic .\data\templates\stab.png .\data\templates\swing.png

python .\main.py evaluate-model .\data\output\polearm_cropped\ .\data\output\polearm_eval\ data/model/spear_logistic .\data\templates\stab.png .\data\templates\swing.png

python .\main.py evaluate-video .\data\input\spear.mkv .\data\output\spear_eval_video\ data/model/spear_logistic .\data\templates\stab.png .\data\templates\swing.png
```

Here's 10 trials worth of data. The API changed since the previous one, where
I'm using multiple smaller templates and omitting the position.

```powershell
python main.py watch '.\data\v1\input\00.mkv' --crop --offset-x -175
python main.py prepare-samples '.\data\v1\input\00.mkv' data/v1/prep --offset-x -175 --frames 3600
python main.py train-classifier data/v1/labeled data/v1/model/logistic data/v1/templates
python main.py evaluate-model data/v1/prep data/v1/model_eval data/v1/model/logistic data/v1/templates

python main.py find-template '.\data\v1\input\00.mkv' '.\data\v1\templates\name.png'
# 932
python main.py find-template '.\data\v1\input\01.mkv' '.\data\v1\templates\name.png' --relative 932
# -69
```
