import json
import pandas as pd
from cosine_similarity_score import cosine_similarity

# File Paths
CSV_FILE = "/home/sshriram2/mi3Testing/zero_shot_detection/cooolerGroundTruth_cleaned.csv"
JSON_FILE = "/home/sshriram2/mi3Testing/zero_shot_detection/jsonresults/refined_output.json"
OUTPUT_FILE = "/home/sshriram2/mi3Testing/zero_shot_detection/jsonresults/similarity_scores_output.json"

# Load CSV Data (Ground Truth)
ground_truth_df = pd.read_csv(CSV_FILE)

# Debug: Print available column names
print("CSV Columns:", ground_truth_df.columns.tolist())

# Define correct column names
scene_col = "Scene"
desc_col = "Description Summary"
hazard_cols = ["Ideal_Hazard_1", "Ideal_Hazard_2", "Ideal_Hazard_3", "Ideal_Hazard_4"]

# Load JSON Data (Detected Output)
with open(JSON_FILE, "r") as f:
    detected_data = json.load(f)

# Create dictionary to store similarity results
similarity_results = {}

for _, row in ground_truth_df.iterrows():
    video_id = str(row[scene_col]).strip()
    ground_truth_desc = str(row[desc_col]).strip()

    # Combine all hazard columns into one string
    ground_truth_hazards = " ".join(
        str(row[col]).strip() for col in hazard_cols if pd.notna(row[col])
    )

    # Debug: Print values before computing similarity
    print(f"\nProcessing Video: {video_id}")
    print(f"Ground Truth Description: {ground_truth_desc}")
    print(f"Ground Truth Hazards: {ground_truth_hazards}")

    if video_id not in detected_data:
        print(f"Skipping {video_id}, not found in JSON.")
        continue  # Skip if no matching video in JSON

    detected_desc_list = detected_data[video_id]["description"]
    detected_hazards_list = detected_data[video_id]["nouns"]

    # Compute similarity for descriptions
    desc_similarities = []
    if ground_truth_desc and detected_desc_list:
        for detected_desc in detected_desc_list:
            similarity_score = cosine_similarity(ground_truth_desc, detected_desc)
            desc_similarities.append(similarity_score)

    # Compute similarity for hazards (nouns)
    hazard_similarities = []
    if ground_truth_hazards and detected_hazards_list:
        for detected_hazard in detected_hazards_list:
            similarity_score = cosine_similarity(ground_truth_hazards, detected_hazard)
            hazard_similarities.append(similarity_score)

    # Store results
    similarity_results[video_id] = {
        "description_similarity": desc_similarities,
        "hazard_similarity": hazard_similarities
    }

# Save similarity results to JSON
with open(OUTPUT_FILE, "w") as f:
    json.dump(similarity_results, f, indent=4)

print(f"Similarity results saved to {OUTPUT_FILE}")