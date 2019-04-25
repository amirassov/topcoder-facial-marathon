import argparse
import os
from functools import partial

import cv2
import numpy as np
import pandas as pd
from multiprocessing import Pool
from tqdm import tqdm

from reid.insightface.model import ArcFaceModel


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mtcnn_path', type=str)
    parser.add_argument('--embedder_path', type=str)
    parser.add_argument('--root', type=str)
    parser.add_argument('--annotation_path', type=str)
    parser.add_argument('--output_path', type=str)
    parser.add_argument('--n_jobs', type=int, default=40)
    return parser.parse_args()


# dirty hack for parallel
args = parse_args()
model = ArcFaceModel(mtcnn_path=args.mtcnn_path, embedder_path=args.embedder_path, image_size=(112, 112))


def prepare_bboxes(group):
    x_min = group['FACE_X'].values
    y_min = group['FACE_Y'].values

    x_max = x_min + group['FACE_WIDTH'].values
    y_max = y_min + group['FACE_HEIGHT'].values

    bboxes = np.stack([x_min, y_min, x_max, y_max], axis=-1)
    return np.array(bboxes).astype(int)


def predict(group, root, output_path):
    image_name, group = group
    image = cv2.imread(os.path.join(root, image_name))
    embeddings = []
    labels = []
    ids = []
    for label, id_, bbox in zip(group['SUBJECT_ID'].values, group['FACE_ID'].values, prepare_bboxes(group)):
        x_min, y_min, x_max, y_max = bbox
        embedding = model.predict(image[y_min:y_max, x_min:x_max].copy())
        if embedding is not None:
            embeddings.append(embedding)
            labels.append(label)
            ids.append(id_)
    np.savez(
        file=os.path.join(output_path, image_name),
        embeddings=np.array(embeddings),
        labels=labels,
        ids=ids
    )


def main():
    annotation = pd.read_csv(args.annotation_path)

    print(len(annotation), len(set(annotation['FILE'])))
    groups = list(annotation.groupby('FILE'))

    partial_predict = partial(predict, root=args.root, output_path=args.output_path)
    with Pool(args.n_jobs) as p:
        list(tqdm(iterable=p.imap_unordered(partial_predict, groups), total=len(groups)))


if __name__ == '__main__':
    main()
