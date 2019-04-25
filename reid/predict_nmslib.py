import argparse
from glob import glob

import numpy as np

from reid.knn import NMSLibNeighbours


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--embedding_path', type=str)
    parser.add_argument('--knn_path', type=str)
    parser.add_argument('--ids_path', type=str)
    parser.add_argument('--n_jobs', type=int, default=16)
    parser.add_argument('--output_path', type=str)
    return parser.parse_args()


def main():
    args = parse_args()

    paths = sorted(glob(args.embedding_path))
    knn = NMSLibNeighbours(n_neighbours=10, space='cosinesimil', n_jobs=args.n_jobs)
    knn.load(args.knn_path)
    predictions = knn.predict(paths)
    np.savez(args.output_path, **predictions)


if __name__ == '__main__':
    main()
