import os
import json
import numpy as np
import re
import sys

OUTPUT_DIR = "./output_similarity_scores"
JSON_RESULTS_DIR = "./jsonresults"  # ✅ Ensures JSON output is stored properly

def extract_track_number(track_label):
    match = re.search(r'\d+', track_label)
    return int(match.group()) if match else track_label

def get_frame_numbers(video_name):
    """Finds all available frames for a given video."""
    video_dir = os.path.join(OUTPUT_DIR, video_name)
    
    if not os.path.exists(video_dir):
        print(f"❌ No directory found for video: {video_name}")
        return []

    frame_numbers = sorted(set(
        f.split("_")[-2] for f in os.listdir(video_dir) if f.endswith("_similarity.npy")
    ), key=int)

    if not frame_numbers:
        print(f"❌ No valid frames found in {video_dir}!")
    return frame_numbers

def process_video_frames(video_name, anomaly_list):
    """
    Processes all frames for a given video and returns a JSON object summarizing all frame mappings.
    Ensures hazards are saved properly.
    """
    anomaly_set = set(anomaly_list)  # ✅ Convert to a set for fast lookup

    json_output_dir = os.path.join(JSON_RESULTS_DIR, video_name)
    os.makedirs(json_output_dir, exist_ok=True)  # ✅ Ensure directory exists

    json_output_file = os.path.join(json_output_dir, f"{video_name}_mappings.json")

    video_data = {"video": video_name, "frames": {}}  # ✅ Default structure

    frame_numbers = get_frame_numbers(video_name)
    if not frame_numbers:
        print(f"❌ No frames found for {video_name}. Exiting.")
        return

    for frame_number in frame_numbers:
        frame_key = f"frame_{frame_number}"

        similarity_file = os.path.join(OUTPUT_DIR, video_name, f"{video_name}_frame_{frame_number}_similarity.npy")
        track_ids_file = os.path.join(OUTPUT_DIR, video_name, f"{video_name}_frame_{frame_number}_track_ids.npy")
        hazard_labels_file = os.path.join(OUTPUT_DIR, video_name, f"{video_name}_frame_{frame_number}_hazard_labels.npy")

        if not all(os.path.exists(f) for f in [similarity_file, track_ids_file, hazard_labels_file]):
            print(f"⚠️ Missing files for Frame {frame_number}. Skipping...")
            continue

        # ✅ Load data
        similarity_matrix = np.load(similarity_file)
        track_ids = np.load(track_ids_file, allow_pickle=True)
        hazard_labels = np.load(hazard_labels_file, allow_pickle=True)

        track_numbers = [extract_track_number(t) for t in track_ids]

        # ✅ Compute threshold for "important" detections (top 10% similarity)
        top_10_mask = similarity_matrix >= np.percentile(similarity_matrix, 90, axis=1, keepdims=True)

        mapping = {}
        for i, track_number in enumerate(track_numbers):
            for j in range(len(hazard_labels)):
                if top_10_mask[i, j]:
                    hazard = hazard_labels[j]
                    if hazard not in mapping:
                        mapping[hazard] = []
                    mapping[hazard].append(track_number)

        # ✅ Save all frames (even if no hazards found)
        video_data["frames"][frame_key] = mapping

    # ✅ Save full mappings
    with open(json_output_file, "w") as f:
        json.dump(video_data, f, indent=4)

    print(f"✅ Full mapping saved: {json_output_file}")

    # ✅ Filter out hazards NOT in `anomaly_list`
    filtered_frames = {
        frame: {hazard: tracks for hazard, tracks in hazards.items() if hazard in anomaly_set}
        for frame, hazards in video_data["frames"].items()
    }

    filtered_json_output_file = os.path.join(json_output_dir, f"{video_name}_filtered_mappings.json")
    with open(filtered_json_output_file, "w") as f:
        json.dump({"video": video_name, "frames": filtered_frames}, f, indent=4)

    print(f"✅ Filtered mapping saved: {filtered_json_output_file}")

    return video_data

if __name__ == "__main__":
    video_name = sys.argv[1]
    anomaly_list = json.loads(sys.argv[2])  # ✅ Read anomaly list from arguments

    process_video_frames(video_name, anomaly_list)
