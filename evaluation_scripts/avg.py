import json

# Load the similarity results
SIMILARITY_FILE = "/home/sshriram2/mi3Testing/zero_shot_detection/jsonresults/similarity_scores_output.json"

with open(SIMILARITY_FILE, "r") as f:
    similarity_data = json.load(f)

def compute_holistic_metric(similarity_data):
    """
     -------------------------SAM_F---------------------------
    1. Averaging all description similarity scores per video.
    2. Ignoring videos with no results or all zero values.
    3. Averaging across valid videos to compute M.
    """
    video_scores = []
    valid_video_count = 0

    for video_id, scores in similarity_data.items():
        description_sim = scores.get("description_similarity", [])

        # Ignore videos with no valid similarity scores (empty or all zeros)
        if not description_sim or all(score == 0 for score in description_sim):
            print(f"Skipping Video {video_id} - No valid similarity scores.")
            continue

        print(f"\nProcessing Video: {video_id}")

        # Compute average description similarity
        m = len(description_sim)
        sum_desc = sum(description_sim)
        avg_similarity = sum_desc / m if m > 0 else 0

        print(f"Average Description Similarity Score (S̄): {avg_similarity:.4f}")

        video_scores.append(avg_similarity)
        valid_video_count += 1

    # Compute  SAM_F metric
    holistic_metric = sum(video_scores) / valid_video_count if valid_video_count > 0 else 0

    return holistic_metric

# Compute SAM_F metric
M = compute_holistic_metric(similarity_data)

print(f"\n🎯 Final Holistic Metric (M): {M:.4f}")

