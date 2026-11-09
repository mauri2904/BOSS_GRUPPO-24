

from pathlib import Path
import json
import sys

from trackers.iou_tracker import IoUTracker


input_path = Path(sys.argv[1])

output_path = input_path.with_name("tracked_iou.jsonl")

tracker = IoUTracker(iou_threshold=0.3, max_missing=1)


with open(input_path, "r", encoding="utf-8") as input_file, \
     open(output_path, "w", encoding="utf-8") as output_file:

    for line in input_file:

        if not line.strip():
            continue

        frame_data = json.loads(line)

        detections = frame_data["detections"]

        tracked_detections = tracker.update(detections)

        frame_data["detections"] = tracked_detections

        output_file.write(json.dumps(frame_data) + "\n")


print("IoU tracking completed!")
print("File created:", output_path)

