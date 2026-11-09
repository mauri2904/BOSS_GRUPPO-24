
import numpy as np


def box_area(box):
    x1, y1, x2, y2 = box

    width = max(0, x2 - x1)
    height = max(0, y2 - y1)

    return width * height


def compute_iou(box_a, box_b):

    x1_inter = max(box_a[0], box_b[0])
    y1_inter = max(box_a[1], box_b[1])
    x2_inter = min(box_a[2], box_b[2])
    y2_inter = min(box_a[3], box_b[3])

    intersection_area = (
        max(0, x2_inter - x1_inter)
        * max(0, y2_inter - y1_inter)
    )

    area_a = box_area(box_a)
    area_b = box_area(box_b)

    union_area = area_a + area_b - intersection_area

    if union_area == 0:
        return 0.0

    return intersection_area / union_area


def compute_iou_matrix(boxes_a, boxes_b):

    iou_matrix = np.zeros((len(boxes_a), len(boxes_b)))

    for i, box_a in enumerate(boxes_a):
        for j, box_b in enumerate(boxes_b):

            iou_matrix[i, j] = compute_iou(
                box_a, box_b
            )

    return iou_matrix
