import argparse
import os
import pickle
from functools import partial
from multiprocessing import Pool

import pandas as pd
from tqdm import tqdm

from detection.utils import prepare_bboxes_labels
import jpeg4py as jpeg
import cv2

HEIGHT = 2048
WIDTH = 3072


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotation', type=str)
    parser.add_argument('--root', type=str)
    parser.add_argument('--output', type=str)
    parser.add_argument('--output_root', type=str, default=None)
    parser.add_argument('--n_jobs', type=int, default=40)
    parser.add_argument('--n_samples', type=int, default=-1)
    return parser.parse_args()


def convert(group: dict, root: str, output_root=None) -> dict:
    image_name, group = group
    image = jpeg.JPEG(os.path.join(root, image_name)).decode()
    h, w = image.shape[:2]
    if 'FACE_X' in group:
        bboxes, labels = prepare_bboxes_labels(group)
    else:
        bboxes, labels = None, None
    if output_root is not None:
        cv2.imwrite(os.path.join(output_root, image_name), cv2.resize(image, (WIDTH, HEIGHT))[:, :, ::-1])
        bboxes[:, [0, 2]] *= WIDTH / w
        bboxes[:, [1, 3]] *= HEIGHT / h

    return {'filename': image_name, 'width': WIDTH, 'height': HEIGHT, 'ann': {'bboxes': bboxes, 'labels': labels}}


def main():
    args = parse_args()
    annotation = pd.read_csv(args.annotation)
    files = sorted(os.listdir(args.root))
    if args.n_samples != -1:
        files = files[:args.n_samples]
    annotation = annotation.loc[annotation['FILE'].isin(set(files))]
    print(len(annotation), len(set(annotation['FILE'])))
    partial_convert = partial(convert, root=args.root, output_root=args.output_root)
    groups = list(annotation.groupby('FILE'))

    with Pool(args.n_jobs) as p:
        samples = list(tqdm(iterable=p.imap_unordered(partial_convert, groups), total=len(groups)))

    with open(args.output, 'wb') as f:
        pickle.dump(samples, f)


if __name__ == '__main__':
    main()
