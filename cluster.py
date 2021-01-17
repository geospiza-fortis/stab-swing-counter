import json
from pathlib import Path

import click
import cv2 as cv
import hdbscan
import matplotlib.pyplot as plt
import numpy as np
import umap
from tqdm import tqdm


@click.command()
@click.argument("input", type=click.Path(file_okay=False, exists=True))
@click.argument("output", type=click.Path(file_okay=False))
def cluster(input, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)

    data = []
    for path in tqdm(sorted(Path(input).glob("*.png"))):
        # assume square images, lets resize for efficiency
        img = cv.imread(str(path), cv.IMREAD_GRAYSCALE)
        img = cv.resize(img, (64, 64))
        data.append(img.ravel())
    # create matrix
    x = np.array(data)
    print(f"created array of shape {x.shape}")

    # https://umap-learn.readthedocs.io/en/latest/clustering.html#
    # this first one is for visualization
    standard_embedding = umap.UMAP().fit_transform(x)

    clusterable_embedding = umap.UMAP(
        n_neighbors=30, min_dist=0.0, n_components=2
    ).fit_transform(x)

    labels = hdbscan.HDBSCAN(
        min_samples=10,
        min_cluster_size=50,
    ).fit_predict(clusterable_embedding)

    clustered = labels >= 0
    plt.scatter(
        standard_embedding[~clustered, 0],
        standard_embedding[~clustered, 1],
        c=(0.5, 0.5, 0.5),
        alpha=0.5,
    )
    plt.scatter(
        standard_embedding[clustered, 0],
        standard_embedding[clustered, 1],
        c=labels[clustered],
        cmap="Spectral",
    )
    plt.savefig(f"{output}/plot.png")
    with (output / "labels.json").open("w") as fp:
        json.dump(labels.tolist(), fp, indent=2)


if __name__ == "__main__":
    cluster()
