import argparse

import pandas as pd
from tqdm import tqdm
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--detector_solution', type=str)
    parser.add_argument('--nmslib_predictions', type=str)
    parser.add_argument('--output', type=str)
    parser.add_argument('--threshold', type=float, default=0.47)
    return parser.parse_args()


def prepare(detector_solution, nmslib_predictions, threshold):
    detector_solution = detector_solution.set_index('FACE_ID')
    detector_solution['SUBJECT_ID'] = -1
    for label, distance, id_ in tqdm(
        zip(nmslib_predictions['neighbours'], nmslib_predictions['distances'], nmslib_predictions['ids']),
        total=len(nmslib_predictions['ids'])
    ):
        if distance[0] < threshold:
            detector_solution.loc[id_, 'SUBJECT_ID'] = label[0]
    detector_solution['FACE_ID'] = detector_solution.index
    return detector_solution


def main():
    args = parse_args()
    detector_solution = pd.read_csv(args.detector_solution)
    nmslib_predictions = np.load(args.nmslib_predictions)
    submission = prepare(detector_solution, nmslib_predictions, args.threshold)
    submission.to_csv(args.output, index=False)


if __name__ == '__main__':
    main()
