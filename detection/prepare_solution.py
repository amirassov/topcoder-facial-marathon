import argparse
from multiprocessing import Pool

import mmcv
import pandas as pd
from tqdm import tqdm

from detection.utils import test_submission

COLUMNS = ['ImageId', 'FACE_X', 'FACE_Y', 'W', 'H', 'Confidence']


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--predictions', type=str)
    parser.add_argument('--annotation', type=str)
    parser.add_argument('--output', type=str)
    parser.add_argument('--n_jobs', type=int, default=80)
    parser.add_argument('--identification', action='store_true')
    return parser.parse_args()


def convert(prediction_annotation: tuple) -> pd.DataFrame:
    prediction, annotation = prediction_annotation
    prediction = prediction[0]
    solution = pd.DataFrame()
    solution['FACE_X'] = prediction[:, 0]
    solution['FACE_Y'] = prediction[:, 1]
    solution['W'] = prediction[:, 2] - solution['FACE_X']
    solution['H'] = prediction[:, 3] - solution['FACE_Y']
    solution['ImageId'] = annotation['filename']
    solution['Confidence'] = prediction[:, 4]
    return solution[COLUMNS]


def main():
    args = parse_args()
    predictions = mmcv.load(args.predictions)
    annotation = mmcv.load(args.annotation)
    print(len(predictions))
    with Pool(args.n_jobs) as p:
        samples = list(tqdm(iterable=p.imap(convert, zip(predictions, annotation)), total=len(predictions)))

    submission = pd.concat(samples)
    test_submission(submission)
    if args.identification:
        submission['SubjectId'] = range(len(submission))
    submission.to_csv(args.output, index=False)


if __name__ == '__main__':
    main()
