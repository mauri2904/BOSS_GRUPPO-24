

from pathlib import Path
import json
import sys
import csv


input_path = Path(sys.argv[1])

output_dir = Path("results") / "iou"
output_dir.mkdir(parents=True, exist_ok=True)

source_id = input_path.parent.name

output_path = output_dir / f"{source_id}.txt"


with open(input_path, "r", encoding="utf-8") as input_file, \
     open(output_path, "w", encoding="utf-8", newline="") as output_file:

    writer = csv.writer(output_file)

    for line in input_file:

        if not line.strip():
            continue

        frame_data = json.loads(line)

        frame = frame_data["frame_idx"] + 1

        for detection in frame_data["detections"]:

            x1, y1, x2, y2 = detection["bbox_xyxy"]

            width = x2 - x1
            height = y2 - y1

            track_id = detection["track_id"]
            score = detection["score"]

            writer.writerow([
                frame,
                track_id,
                x1,
                y1,
                width,
                height,
                score,
                -1,
                -1,
                -1
            ])


print("Conversion completed!")
print("File created:", output_path)

