from argparse import ArgumentParser

import mmcv
import numpy as np

from mmdet import datasets
from mmdet.core import eval_map


def evaluate(result_file, dataset, iou_thresholds):
    det_results = mmcv.load(result_file)
    gt_bboxes = []
    gt_labels = []
    gt_ignore = []
    for i in range(len(dataset)):
        ann = dataset.get_ann_info(i)
        bboxes = ann['bboxes']
        labels = ann['labels']
        if 'bboxes_ignore' in ann:
            ignore = np.concatenate(
                [np.zeros(bboxes.shape[0], dtype=np.bool),
                 np.ones(ann['bboxes_ignore'].shape[0], dtype=np.bool)]
            )
            gt_ignore.append(ignore)
            bboxes = np.vstack([bboxes, ann['bboxes_ignore']])
            labels = np.concatenate([labels, ann['labels_ignore']])
        gt_bboxes.append(bboxes)
        gt_labels.append(labels)
    if not gt_ignore:
        gt_ignore = None
    if hasattr(dataset, 'year') and dataset.year == 2007:
        dataset_name = 'voc07'
    else:
        dataset_name = dataset.CLASSES

    mean_aps = []
    for iou_thr in iou_thresholds:
        mean_ap, _ = eval_map(
            det_results,
            gt_bboxes,
            gt_labels,
            gt_ignore=gt_ignore,
            scale_ranges=None,
            iou_thr=iou_thr,
            dataset=dataset_name,
            print_summary=False
        )
        mean_aps.append(mean_ap)
    print(f'MAP: {np.mean(mean_aps)}')


def main():
    parser = ArgumentParser(description='Evaluation')
    parser.add_argument('result', help='result file path')
    parser.add_argument('config', help='config file path')
    parser.add_argument('--ann_file', default=None, type=str)
    parser.add_argument('--img_prefix', default=None, type=str)
    args = parser.parse_args()
    cfg = mmcv.Config.fromfile(args.config)
    if args.ann_file is not None:
        cfg.data.test.ann_file = args.ann_file
    if args.img_prefix is not None:
        cfg.data.test.img_prefix = args.img_prefix
    test_dataset = mmcv.runner.obj_from_dict(cfg.data.test, datasets)
    evaluate(args.result, test_dataset, np.arange(0.5, 1.0, 0.05))


if __name__ == '__main__':
    main()
