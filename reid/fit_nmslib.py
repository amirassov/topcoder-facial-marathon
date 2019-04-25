import argparse
from glob import glob

from reid.knn import NMSLibNeighbours


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--embedding_path', type=str)
    parser.add_argument('--knn_path', type=str)
    parser.add_argument('--n_jobs', type=int, default=16)
    return parser.parse_args()


def main():
    args = parse_args()

    paths = sorted(glob(args.embedding_path))
    knn = NMSLibNeighbours(n_neighbours=1, space='cosinesimil', n_jobs=args.n_jobs)
    knn.fit(paths=paths)
    knn.dump(args.knn_path)


if __name__ == '__main__':
    main()
