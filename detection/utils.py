import numpy as np


def prepare_bboxes_labels(group, is_test=False):
    x_min = group['FACE_X'].values
    y_min = group['FACE_Y'].values

    x_max = x_min + group[('FACE_WIDTH', 'W')[is_test]].values
    y_max = y_min + group[('FACE_HEIGHT', 'H')[is_test]].values

    bboxes = np.stack([x_min, y_min, x_max, y_max], axis=-1)
    labels = np.ones(len(bboxes))
    return np.array(bboxes, dtype=np.float32), np.array(labels, dtype=np.int)


def test_submission(submission):
    assert max(submission['Confidence']) <= 1.0 and min(submission['Confidence']) >= 0.0
    assert np.all(submission[['FACE_X', 'FACE_Y', 'W', 'H']] >= 0)
