


from pathlib import Path
import configparser
import json
import sys


sequence_dir = Path(sys.argv[1])
source_id = sequence_dir.name
detector = source_id.split("-")[-1]

config = configparser.ConfigParser()
config.read(sequence_dir / "seqinfo.ini")

fps = int(config["Sequence"]["frameRate"])
num_frames = int(config["Sequence"]["seqLength"])

detections_path = sequence_dir / "det" / "det.txt"

detections = {}

for frame_idx in range(num_frames):
    detections[frame_idx] = []

with open(detections_path, "r", encoding="utf-8") as input_file:
    for line in input_file:
        if not line.strip():
            continue

        fields = line.strip().split(",")

        frame_idx = int(fields[0]) - 1

        x = float(fields[2])
        y = float(fields[3])
        width = float(fields[4])
        height = float(fields[5])
        score = float(fields[6])

        if not 0 <= frame_idx < num_frames:
            raise ValueError("Invalid frame number")

        if not 0 <= score <= 1:
            raise ValueError(
                f"Score outside the BOSS range: {score}"
            )

        bbox_xyxy = [
            x,
            y,
            x + width,
            y + height
        ]

        detection = {
            "bbox_xyxy": bbox_xyxy,
            "cls": "person",
            "score": score
        }

        detections[frame_idx].append(detection)


output_path = sequence_dir / "detections.jsonl"

with open(output_path, "w", encoding="utf-8") as output_file:
    for frame_idx in range(num_frames):

        frame_data = {
            "schema": "boss.perception/0.1",
            "source_id": source_id,
            "frame_idx": frame_idx,
            "ts_ms": round(frame_idx / fps * 1000, 2),
            "models": {
                "det": detector
            },
            "detections": detections[frame_idx]
        }

        output_file.write(json.dumps(frame_data) + "\n")

print("Conversion completed!")
print("File created:", output_path)


