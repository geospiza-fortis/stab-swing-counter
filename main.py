import numpy as np
import cv2 as cv
import click
from pathlib import Path
from tqdm import tqdm

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


if __name__ == "__main__":
    cli()