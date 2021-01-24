# counting frames

I run a template matching algorithm, and then pass the features into a
multi-class classifier. Made a few templates based on a small region in the
cropped screenshot for pattern matching in openCV. Here's 10 trials worth of
data. The API changed since the previous one, where I'm using multiple smaller
templates and omitting the position.

```powershell
python main.py watch '.\data\v1\input\00.mkv' --crop --offset-x -175
python main.py prepare-samples '.\data\v1\input\00.mkv' data/v1/prep --offset-x -175 --frames 3600
python main.py train-classifier data/v1/labeled data/v1/model/logistic data/v1/templates
python main.py evaluate-model data/v1/prep data/v1/model_eval data/v1/model/logistic data/v1/templates

python main.py find-template '.\data\v1\input\00.mkv' '.\data\v1\templates\name.png'
# 932
python main.py find-template '.\data\v1\input\01.mkv' '.\data\v1\templates\name.png' --relative 932
# -69

python run_trials.py
python run_compress.py
python run_copy.py
```

```
python main.py train-sliding-classifier data/v1/labeled data/v1/prep data/v1/model/logistic data/v1/templates

```
