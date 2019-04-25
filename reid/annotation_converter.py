import argparse
import pandas as pd
from detection.utils import test_submission

COLUMNS = ['ImageId', 'SubjectId', 'FACE_X', 'FACE_Y', 'W', 'H', 'Confidence']
TEST_COLUMNS = ['ImageId', 'SubjectId', 'W', 'H']
TRAIN_COLUMNS = ['FILE', 'SUBJECT_ID', 'FACE_WIDTH', 'FACE_HEIGHT']


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotation', type=str)
    parser.add_argument('--output', type=str)
    parser.add_argument('--to_test', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    annotation = pd.read_csv(args.annotation)
    if args.to_test:
        convert_dict = dict(zip(TRAIN_COLUMNS, TEST_COLUMNS))
    else:
        convert_dict = dict(zip(TEST_COLUMNS, TRAIN_COLUMNS))
        annotation['FACE_ID'] = range(len(annotation))
        annotation['SUBJECT_ID'] = range(len(annotation))
    annotation = annotation.rename(columns=convert_dict)
    if args.to_test:
        annotation = annotation[COLUMNS]
        test_submission(annotation)
    annotation.to_csv(args.output, index=False)


if __name__ == '__main__':
    main()
