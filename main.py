import json
import pickle
from pathlib import Path

import click
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression, RidgeClassifierCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

# 1080 x 1920 is the default resolution of the hd client
H = 1080
W = 1920


# https://docs.opencv.org/master/d4/dc6/tutorial_py_template_matching.html
def get_template_features(template, img, include_pos):
    # squared differences
    w, h = template.shape[::-1]
    res = cv.matchTemplate(img, template, cv.TM_SQDIFF)
    min_val, _, min_loc, _ = cv.minMaxLoc(res)
    # drawing a rectangle is easy, but we just want the features
    if include_pos:
        return np.array([min_val, min_loc[0], min_loc[1]])
    else:
        # let's not even use the location, and just use the min value
        return np.array([min_val])


def read_templates(templates):
    template_img = []
    for path in sorted(Path(templates).glob("*.png")):
        img = cv.imread(str(path), cv.IMREAD_GRAYSCALE)
        # hacky, but lets put the name first so we can normalize
        if "name" in path.name:
            template_img = [img] + template_img
        else:
            template_img.append(img)
    return template_img


def create_template_feature(img, template_img, include_pos=False):
    features = [get_template_features(tmpl, img, include_pos) for tmpl in template_img]
    if include_pos:
        # normalize the features against the name feature to get relative
        # positions in the frame which is fixed underneath the player
        new_features = []
        # TODO: this could (or maybe should) be vectorized...
        relative = np.array(features[0])
        for feature in features[1:]:
            f = np.array(feature) - relative
            new_features.append(f)
        features = new_features
    return np.array(features).reshape(-1)


def generate_images(input_path):
    for path in sorted(Path(input_path).glob("**/*.png")):
        yield cv.imread(str(path), cv.IMREAD_GRAYSCALE)


def generate_capture_frames(input, offset_x, crop=True):
    cap = cv.VideoCapture(input)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        img = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        upper_left = ((W // 2) + offset_x, (H // 2 + 75))
        lower_right = ((W // 2) + offset_x + 300, (H // 2 + 375))
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
        yield img
    cap.release()


def evaluate_batch(clf, labels, data, frames, video, output, batch_num, plot=False):
    X = np.array(data)
    # check if it has a decision function first
    if hasattr(clf, "decision_function"):
        y = clf.decision_function(X)
    else:
        y = clf.predict_proba(X)
    y_label = [labels[str(p)] for p in clf.predict(X)]

    for img, label in zip(frames, y_label):
        font = cv.FONT_HERSHEY_SIMPLEX
        img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
        cv.putText(img, label, (10, 50), font, 2, (0, 0, 0), 2, cv.LINE_AA)
        video.write(img)

    if plot:
        plt.clf()
        xs = list(range(len(data)))
        for k, v in labels.items():
            if v == "other":
                continue
            plt.plot(xs, y[:, int(k)], label=v)
        plt.legend()
        plt.savefig(f"{output}/probs_{batch_num:03}.png")

    return y, y_label


def transform_labels(data, threshold=20):
    """Transform the labels for each frame in the video into a table of events.
    We omit events that do not seem plausible using the threshold."""
    res = []
    start = 0
    value = "other"
    for i, e in enumerate(data):
        if e == value:
            continue
        if value != "other":
            # must wait at least this many frames before the last change
            diff = (start - res[-1]["start"]) if res else i
            row = dict(
                pos=len(res),
                label=value,
                start=start,
                end=i,
                diff=diff,
                stab=(
                    len([x for x in res if x["label"] == "stab"]) + int(value == "stab")
                ),
                swing=(
                    len([x for x in res if x["label"] == "swing"])
                    + int(value == "swing")
                ),
                total=len(res) + 1,
            )
            res.append(row)
        start = i
        value = e
    return [
        {
            **x,
            **dict(
                stab_pct=round(x["stab"] / x["total"], 3),
                swing_pct=round(x["swing"] / x["total"], 3),
            ),
        }
        for x in res
        if x["diff"] > threshold
    ]


@click.group()
def cli():
    pass


@cli.command()
@click.argument("input", type=click.Path(dir_okay=False, exists=True))
@click.argument("template", type=click.Path(dir_okay=False, exists=True))
@click.option("--relative", type=int, default=0)
def find_template(input, template, relative):
    cap = cv.VideoCapture(input)
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
    img = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    cap.release()
    cv.destroyAllWindows()
    template = cv.imread(template, cv.IMREAD_GRAYSCALE)
    w, h = template.shape[::-1]
    res = cv.matchTemplate(img, template, cv.TM_SQDIFF)
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
    print(min_loc[0] - relative)


@cli.command()
@click.argument("input", type=click.Path(dir_okay=False, exists=True))
@click.option("--crop/--no-crop", default=False)
@click.option("--offset-x", default=-200)
def watch(input, crop, offset_x):
    for frame in tqdm(generate_capture_frames(input, offset_x, crop)):
        cv.imshow("frame", frame)
        if cv.waitKey(1) == ord("q"):
            break
    cv.destroyAllWindows()


@cli.command()
@click.argument("input", type=click.Path(dir_okay=False, exists=True))
@click.argument("output", type=click.Path(file_okay=False))
@click.option("--offset-x", default=-200)
@click.option("--frames", default=60 * 30)
def prepare_samples(input, output, offset_x, frames):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)

    for idx, frame in tqdm(enumerate(range(input, offset_x))):
        cv.imwrite(f"{output}/img_{idx:05}.png", frame)
    print("done collecting frames")
    cv.destroyAllWindows()


@cli.command()
@click.argument("input", type=click.Path(file_okay=False, exists=True))
@click.argument("input-full", type=click.Path(file_okay=False, exists=True))
@click.argument("output", type=click.Path(file_okay=False))
@click.argument("templates", type=click.Path(file_okay=False, exists=True))
@click.option("--window", type=int, default=8)
@click.option("--include-pos/--no-include-pos", default=False)
@click.option(
    "--model",
    type=click.Choice(["logistic", "decisiontree", "ridge"]),
    default="logistic",
)
def train(input, input_full, output, templates, window, include_pos, model):
    input = Path(input)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)

    template_img = read_templates(templates)
    # find the frames we need to read into memory and cache them
    def parse_number(name):
        return int(name.split("_")[-1].split(".")[0].lstrip("0") or "0")

    # cache frames
    frames = {}
    input_full_path = sorted(Path(input_full).glob("**/*.png"))
    input_paths = sorted(Path(input).glob("**/*.png"))
    for path in tqdm(input_paths):
        idx = parse_number(path.name) + 1
        if idx - window < 0:
            raise ValueError(f"lag on frame is less than 0 - {idx} on {path.name}")
        for frame in input_full_path[idx - window : idx]:
            if frame.name in frames:
                continue
            img = cv.imread(str(frame), cv.IMREAD_GRAYSCALE)
            feature = create_template_feature(img, template_img, include_pos)
            frames[frame.name] = feature

    data = []
    labels = []
    for path in input_paths:
        label_class = path.parent.name
        idx = parse_number(path.name) + 1
        row = np.array(
            [frames[p.name] for p in input_full_path[idx - window : idx]]
        ).reshape(-1)
        data.append(row)
        labels.append(label_class)

    le = LabelEncoder()
    le.fit(labels)
    print(list(le.classes_))
    y = le.transform(labels)

    X = np.array(data)
    print(f"data in shape {X.shape}")
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    models = {
        "logistic": LogisticRegression,
        "ridge": RidgeClassifierCV,
        "decisiontree": DecisionTreeClassifier,
    }

    clf = models[model]().fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"accuracy: {acc} with {len(y_test)} samples")

    # retrain on all the data
    clf = models[model]().fit(X, y)
    acc = accuracy_score(y, clf.predict(X))
    print(f"accuracy: {acc} on all data of length {len(y)}")

    with (output / "model.pkl").open("wb") as fp:
        pickle.dump(clf, fp)
    with (output / "labels.json").open("w") as fp:
        # for mapping label back to the original string later
        json.dump({i: v for i, v in enumerate(le.classes_)}, fp)


@cli.command()
@click.argument("input", type=click.Path(exists=True))
@click.argument("output", type=click.Path(file_okay=False))
@click.argument("model_input", type=click.Path(file_okay=False, exists=True))
@click.argument("templates", type=click.Path(file_okay=False, exists=True))
@click.option("--window", type=int, default=8)
@click.option("--include-pos/--no-include-pos", default=False)
@click.option("--batch-size", type=int, default=128)
@click.option("--input-type", type=click.Choice(["video", "image"]), default="video")
@click.option("--offset-x", default=-200)
def evaluate(
    input,
    output,
    model_input,
    templates,
    window,
    include_pos,
    batch_size,
    input_type,
    offset_x,
):
    model_input = Path(model_input)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)

    template_img = read_templates(templates)

    with (model_input / "model.pkl").open("rb") as fp:
        clf = pickle.load(fp)
    labels = json.loads((model_input / "labels.json").read_text())

    fourcc = cv.VideoWriter_fourcc(*"mp4v")
    video = cv.VideoWriter(f"{output}/labeled.mp4", fourcc, 60, (300, 300))

    history = []
    data = []
    frames = []
    pred_prob = []
    pred_label = []
    batch_num = 0

    generate = (
        generate_capture_frames(input, offset_x, True)
        if input_type == "video"
        else generate_images(input)
    )

    for frame in tqdm(generate):
        feature = create_template_feature(frame, template_img, include_pos)
        frames.append(frame)
        history.append(feature)
        if len(history) < window:
            continue
        assert len(history) == window
        data.append(np.array(history).reshape(-1))
        history.pop(0)
        if len(data) >= batch_size:
            y, y_label = evaluate_batch(
                clf, labels, data, frames, video, output, batch_num, True
            )
            pred_prob += y.tolist()
            pred_label += y_label
            data = []
            frames = []
            batch_num += 1

    y, y_label = evaluate_batch(
        clf, labels, data, frames, video, output, batch_num, True
    )
    pred_prob += y.tolist()
    pred_label += y_label

    np.savetxt(f"{output}/pred.csv", np.array(pred_prob), delimiter=",")
    (output / "pred.json").write_text(json.dumps(pred_label))

    cv.destroyAllWindows()
    video.release()


@cli.command()
@click.argument("input", type=click.Path(exists=True))
def summarize(input):
    paths = list(Path(input).glob("**/pred.json"))
    summaries = []
    for path in paths:
        print(f"writing transformed data for {path}")
        data = json.loads(path.read_text())
        summary = transform_labels(data)
        # write one for the local trial
        (path.parent / "summary.json").write_text(json.dumps(summary, indent=2))
        summaries.append(
            dict(
                name=path.parent.name,
                **{
                    k: v
                    for k, v in summary[-1].items()
                    if k in ["stab", "swing", "total", "stab_pct", "swing_pct"]
                },
            )
        )
    # and dump this to the current directoy
    (Path(input) / "summary.json").write_text(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    cli()
