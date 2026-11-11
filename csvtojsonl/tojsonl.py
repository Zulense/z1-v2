import csv
import json
from pathlib import Path


def convert_csv_to_jsonl(
    csv_path: str,
    json_parts_path: str,
    jsonl_path: str,
    parts_to_keep: list[str] | None = None,
    video_root: str = "",
    clean_keys: bool = True,
):
    if parts_to_keep is None:
        parts_to_keep = [f"part{i}" for i in range(1, 11)]
    target_parts = set(parts_to_keep)

    # Load allowed video filenames from part1 to part10 into a fast lookup set
    allowed_videos = set()
    with open(json_parts_path, mode="r", encoding="utf-8") as jf:
        parts_data = json.load(jf)
        for item in parts_data:
            for part_name, video_list in item.items():
                if part_name in target_parts:
                    allowed_videos.update(video_list)

    print(f"Loaded {len(allowed_videos):,} allowed videos from parts 1 to 10.")

    kept_count = 0
    skipped_count = 0

    with (
        open(csv_path, mode="r", encoding="utf-8") as csv_file,
        open(jsonl_path, mode="w", encoding="utf-8") as jsonl_file,
    ):
        reader = csv.DictReader(csv_file)

        for row in reader:
            video_filename = row["video"].strip()
            
            # Filter: only keep videos present in part1 to part10
            if video_filename not in allowed_videos:
                skipped_count += 1
                continue

            video_path = (
                str(Path(video_root) / video_filename) if video_root else video_filename
            )

            if clean_keys:
                record = {
                    "video": f"/Data/OpenVidHD/video_folder/{video_path}",
                    "caption": row["caption"],
                    "aesthetic_score": float(row["aesthetic score"]),
                    "motion_score": float(row["motion score"]),
                    "temporal_consistency_score": float(row["temporal consistency score"]),
                    "camera_motion": row["camera motion"],
                    "frame": int(row["frame"]),
                    "fps": float(row["fps"]),
                    "seconds": float(row["seconds"]),
                }
            else:
                record = {
                    "video": f"/Data/OpenVidHD/video_folder/{video_path}",
                    "caption": row["caption"],
                    "aesthetic score": float(row["aesthetic score"]),
                    "motion score": float(row["motion score"]),
                    "temporal consistency score": float(row["temporal consistency score"]),
                    "camera motion": row["camera motion"],
                    "frame": int(row["frame"]),
                    "fps": float(row["fps"]),
                    "seconds": float(row["seconds"]),
                }

            jsonl_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            kept_count += 1

    print(f"Conversion complete! Kept: {kept_count:,} | Skipped: {skipped_count:,}")


import csv
import json
from pathlib import Path

def convert_part11_csv_to_jsonl(
    csv_path: str,
    json_parts_path: str,
    jsonl_path: str,
    target_part: str = "part11",
    video_root: str = "",
    clean_keys: bool = True,
):
    # Load allowed video filenames from part11
    allowed_videos = set()
    with open(json_parts_path, mode="r", encoding="utf-8") as jf:
        parts_data = json.load(jf)
        for item in parts_data:
            if target_part in item:
                allowed_videos.update(item[target_part])

    print(f"Loaded {len(allowed_videos):,} allowed videos from {target_part}.")

    kept_count = 0
    skipped_count = 0

    with (
        open(csv_path, mode="r", encoding="utf-8") as csv_file,
        open(jsonl_path, mode="w", encoding="utf-8") as jsonl_file,
    ):
        reader = csv.DictReader(csv_file)

        for row in reader:
            video_filename = row["video"].strip()
            
            # Filter: only keep videos present in part11
            if video_filename not in allowed_videos:
                skipped_count += 1
                continue

            video_path = (
                str(Path(video_root) / video_filename) if video_root else video_filename
            )

            if clean_keys:
                record = {
                    "video": f"/Data/OpenVidHD/video_folder/{video_path}",
                    "caption": row["caption"],
                    "aesthetic_score": float(row["aesthetic score"]),
                    "motion_score": float(row["motion score"]),
                    "temporal_consistency_score": float(row["temporal consistency score"]),
                    "camera_motion": row["camera motion"],
                    "frame": int(row["frame"]),
                    "fps": float(row["fps"]),
                    "seconds": float(row["seconds"]),
                }
            else:
                record = {
                    "video": f"/Data/OpenVidHD/video_folder/{video_path}",
                    "caption": row["caption"],
                    "aesthetic score": float(row["aesthetic score"]),
                    "motion score": float(row["motion score"]),
                    "temporal consistency score": float(row["temporal consistency score"]),
                    "camera motion": row["camera motion"],
                    "frame": int(row["frame"]),
                    "fps": float(row["fps"]),
                    "seconds": float(row["seconds"]),
                }

            jsonl_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            kept_count += 1

    print(f"Validation conversion complete! Kept: {kept_count:,} | Skipped: {skipped_count:,}")

if __name__ == "__main__":
    convert_part11_csv_to_jsonl(
        csv_path="/home/manish/Desktop/projects/z1-v2/LeanVAE/annotation/OpenVidHD.csv",
        json_parts_path="/home/manish/Desktop/projects/z1-v2/LeanVAE/annotation/OpenVidHD.json",
        jsonl_path="OpenVidHD_val_part11.jsonl",
        target_part="part59",
        video_root="",
        clean_keys=True,
    )

