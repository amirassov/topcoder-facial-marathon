import argparse
import os

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str)
    parser.add_argument('--output', type=str)
    return parser.parse_args()


def main():
    args = parse_args()
    output = pd.DataFrame()
    output['FILE'] = sorted(os.listdir(args.root))
    output.to_csv(args.output, index=False)


if __name__ == '__main__':
    main()
