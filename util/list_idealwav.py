import re
import os
OUTPUT_DIR="output_chunksV2"
short_clips_path = os.path.join(OUTPUT_DIR, "short_clips.txt")
with open(short_clips_path, "w", encoding="utf-8") as f_out:
    print("\n📁 Listing clips with ≤15s duration in subfolders of output_chunksV2:\n")
    for root, _, files in os.walk(OUTPUT_DIR):
        for file in files:
            match = re.match(r"(clip_|ideal_)\d+_(\d+)s\.wav", file)
            if match:
                duration = int(match.group(2))
                if duration <= 20:
                    full_path = os.path.join(root, file)
                    print(f"- {full_path}")
                    f_out.write(full_path + "\n")

print(f"\n📝 Saved list of short clips to: {short_clips_path}")