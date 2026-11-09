
from scipy.optimize import linear_sum_assignment
from .iou_utils import compute_iou_matrix


def match_detections(current_boxes, track_boxes, iou_threshold=0.3):
    if len(current_boxes) == 0 or len(track_boxes) == 0:
        return []

    iou_matrix = compute_iou_matrix(current_boxes, track_boxes)
    det_indices, track_indices = linear_sum_assignment(-iou_matrix)

    matches = []

    for det_idx, track_idx in zip(det_indices, track_indices):

        if iou_matrix[det_idx, track_idx] >= iou_threshold:
            matches.append((det_idx, track_idx))

    return matches


class IoUTracker:

    def __init__(self, iou_threshold=0.3, max_missing=1):

        self.iou_threshold = iou_threshold
        self.max_missing = max_missing

        self.tracks = {}
        self.next_track_id = 1

    def update(self, detections):

        current_boxes = [
            detection["bbox_xyxy"]
            for detection in detections
        ]

        track_ids = list(self.tracks.keys())

        track_boxes = [
            self.tracks[track_id]["bbox_xyxy"]
            for track_id in track_ids
        ]

        matches = match_detections(
            current_boxes,
            track_boxes,
            self.iou_threshold
        )

        tracked_detections = [
            detection.copy()
            for detection in detections
        ]

        matched_detections = set()
        matched_tracks = set()

        for det_idx, track_idx in matches:

            track_id = track_ids[track_idx]

            self.tracks[track_id]["bbox_xyxy"] = current_boxes[det_idx]

            self.tracks[track_id]["missing"] = 0

            tracked_detections[det_idx]["track_id"] = track_id

            matched_detections.add(det_idx)
            matched_tracks.add(track_id)

        for det_idx, detection in enumerate(detections):

            if det_idx not in matched_detections:

                track_id = self.next_track_id
                self.next_track_id += 1

                self.tracks[track_id] = {
                    "bbox_xyxy": current_boxes[det_idx],
                    "missing": 0
                }

                tracked_detections[det_idx]["track_id"] = track_id

        for track_id in track_ids:

            if track_id not in matched_tracks:

                self.tracks[track_id]["missing"] += 1

                if self.tracks[track_id]["missing"] > self.max_missing:
                    del self.tracks[track_id]

        return tracked_detections
