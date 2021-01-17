import json
from pathlib import Path

import click
import cv2 as cv
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np


# 1080 x 1920
H = 1080
W = 1920

# renamed these to spear.mkv and polearm.mkv respectively
# SPEAR = "data/input/2021-01-16 21-38-22.mkv"
# POLEARM = "data/input/2021-01-16 21-41-58.mkv"


@click.group()
def cli():
    pass


@cli.command()
@click.argument("input", type=click.Path(dir_okay=False, exists=True))
@click.option("--crop/--no-crop", default=False)
def watch(input, crop):
    cap = cv.VideoCapture(input)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        img = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        upper_left = ((W // 2) - 250, (H // 2 + 50))
        lower_right = ((W // 2) + 50, (H // 2 + 350))
        if crop:
            img = img[upper_left[1] : lower_right[1], upper_left[0] : lower_right[0]]
        else:
            # show the full video with a rectangle
            cv.rectangle(
                img,
                upper_left,
                lower_right,
                (0, 255, 0),
                3,
            )
        cv.imshow("frame", img)
        if cv.waitKey(1) == ord("q"):
            break
    cap.release()
    cv.destroyAllWindows()


@cli.command()
@click.argument("input", type=click.Path(dir_okay=False, exists=True))
@click.argument("output", type=click.Path(file_okay=False))
@click.option("--frames", default=60 * 30)
def prepare_samples(input, output, frames):
    cap = cv.VideoCapture(input)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)

    for idx in tqdm(range(frames)):
        if not cap.isOpened():
            break
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        upper_left = ((W // 2) - 250, (H // 2 + 50))
        lower_right = ((W // 2) + 50, (H // 2 + 350))
        crop_img = gray[upper_left[1] : lower_right[1], upper_left[0] : lower_right[0]]
        cv.imwrite(f"{output}/img_{idx:05}.png", crop_img)
    print("done collecting frames")
    cap.release()
    cv.destroyAllWindows()


def mse(a, b):
    """Mean Squared Error."""
    return ((a - b) ** 2).mean()


@cli.command()
@click.argument("input", type=click.Path(file_okay=False, exists=True))
@click.argument("output", type=click.Path(file_okay=False))
@click.argument("stab", type=int)
@click.argument("swing", type=int)
def calculate_diff(input, output, stab, swing):
    input = Path(input)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)

    # we know the naming convention
    def copy_load(idx):
        name = f"img_{idx:05}.png"
        img = cv.imread(str(input / name), cv.IMREAD_GRAYSCALE)
        cv.imwrite(str(output / name), img)
        return img

    stab_img = copy_load(stab)
    swing_img = copy_load(swing)

    stab_diff = []
    swing_diff = []
    for path in tqdm(sorted(Path(input).glob("*.png"))):
        img = cv.imread(str(path), cv.IMREAD_GRAYSCALE)
        # method may not work well if the background or character position changes
        stab_diff.append(mse(stab_img, img))
        swing_diff.append(mse(swing_img, img))

    with (output / "data.json").open("w") as fp:
        json.dump(dict(stab=stab_diff, swing=swing_diff), fp, indent=2)

    x = list(range(len(stab_diff)))
    plt.plot(x, stab_diff, label="stab")
    plt.plot(x, swing_diff, label="swing")
    plt.legend()
    plt.savefig(f"{output}/mse.png")


# https://docs.opencv.org/master/d4/dc6/tutorial_py_template_matching.html
def get_template_features(template, img):
    # squared differences
    w, h = template.shape[::-1]
    res = cv.matchTemplate(img, template, cv.TM_SQDIFF)
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
    # drawing a rectangle is easy, but we just want the features
    # top_left = min_loc
    # bottom_right = (top_left[0] + w, top_left[1] + h)
    # v.rectangle(img,top_left, bottom_right, 255, 2)
    return np.array([min_val, *min_loc])


from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle


@cli.command()
@click.argument("input", type=click.Path(file_okay=False, exists=True))
@click.argument("output", type=click.Path(file_okay=False))
@click.argument("stab", type=click.Path(dir_okay=True, exists=True))
@click.argument("swing", type=click.Path(dir_okay=True, exists=True))
def train_classifier(input, output, stab, swing):
    input = Path(input)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)

    stab_img = cv.imread(str(stab), cv.IMREAD_GRAYSCALE)
    swing_img = cv.imread(str(swing), cv.IMREAD_GRAYSCALE)

    # use a label encoder instead?
    data = []
    labels = []
    for path in tqdm(sorted(Path(input).glob("**/*.png"))):
        label_class = path.parent.name
        img = cv.imread(str(path), cv.IMREAD_GRAYSCALE)
        row = np.append(
            get_template_features(stab_img, img), get_template_features(swing_img, img)
        )
        data.append(row)
        labels.append(label_class)

    le = LabelEncoder()
    le.fit(labels)
    print(list(le.classes_))
    y = le.transform(labels)

    X = np.array(data)
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    clf = LogisticRegression().fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"accuracy: {acc} with {len(y_test)} samples")

    # retrain on all the data
    clf = LogisticRegression().fit(X, y)
    acc = accuracy_score(y, clf.predict(X))
    print(f"accuracy: {acc} on all data of length {len(y)}")

    with (output / "model.pkl").open("wb") as fp:
        pickle.dump(clf, fp)
    with (output / "labels.json").open("w") as fp:
        # for mapping label back to the original string later
        json.dump({i: v for i, v in enumerate(le.classes_)}, fp)


if __name__ == "__main__":
    cli()
