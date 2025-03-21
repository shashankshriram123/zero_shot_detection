import json
import os
import cv2
import numpy as np
from sklearn.linear_model import LinearRegression

# Load JSON data from file
json_path = "/Users/shashankshriram/Desktop/zero_shot_detection/test.json"
with open(json_path, "r") as f:
    output_data = json.load(f)

# Validate the output format
assert "video" in output_data and "frames" in output_data, "Invalid JSON structure"

video = output_data["video"]

# Determine the total number of frames based on extracted snippets folder
extracted_snippets_path = f"./extractedSnippets/{video}"
num_frames = len([name for name in os.listdir(extracted_snippets_path) if os.path.isdir(os.path.join(extracted_snippets_path, name))])

# Open results file
results_file = open("test_results.csv", 'w')
results_file.write("ID,Driver_State_Changed,Hazard_Track,Hazard_Name\n")

# Convert frame-based output to a {frame: {track_id: description}} dictionary
frame_descriptions = {}
for frame, hazards in output_data["frames"].items():
    for description, track_ids in hazards.items():
        for track_id in track_ids:
            frame_descriptions.setdefault(frame, {})[track_id] = description

# Simulated driver state change detection variables
median_dists = []
driver_state_flag = False

# Process all frames and write to CSV
for frame_num in range(num_frames):
    frame_key = f"frame_{frame_num:03d}"
    hazards = frame_descriptions.get(frame_key, {})
    
    # Simulated distance calculations for driver state change
    if len(median_dists) > 1:
        x = np.array(range(len(median_dists))).reshape(-1, 1)
        y = np.array(median_dists)
        speed_model = LinearRegression().fit(x, y)
        if speed_model.coef_[0] < 0:
            driver_state_flag = True
    
    # If there are no hazards, write only ID and Driver_State_Changed
    if not hazards:
        results_file.write(f"{video}_{frame_num},{driver_state_flag},,\n")
    else:
        for track_id, description in hazards.items():
            results_file.write(f"{video}_{frame_num},{driver_state_flag},{track_id},{description}\n")
    
    # Simulating median distance recording for next iteration
    median_dists.append(np.random.rand())  # Replace with actual calculations

results_file.close()
print("Test results saved to test_results.csv")
