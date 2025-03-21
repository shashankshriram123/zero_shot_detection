import json

# Load the similarity results
SIMILARITY_FILE = "/home/sshriram2/mi3Testing/zero_shot_detection/jsonresults/similarity_scores_output.json"

with open(SIMILARITY_FILE, "r") as f:
    similarity_data = json.load(f)

def compute_besm(similarity_data):
    """
    ---------------------------BESM_F----------------------------
    1. Computing adjusted similarity scores per video:
       - If multiple similarity values exist, use (max + min) / 2.
       - If only one unique similarity value exists, use that value.
    2. Ignoring videos with no results or all zero values.
    3. Averaging across all valid videos to compute BESM.
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

        # Compute adjusted similarity score
        unique_scores = set(description_sim)
        if len(unique_scores) > 1:
            adjusted_similarity = (max(unique_scores) + min(unique_scores)) / 2
        else:
            adjusted_similarity = description_sim[0]  # Only one unique value

        print(f"Adjusted Similarity Score (S̄): {adjusted_similarity:.4f}")

        video_scores.append(adjusted_similarity)
        valid_video_count += 1

    # Compute  BESM_F metric
    besm = sum(video_scores) / valid_video_count if valid_video_count > 0 else 0

    return besm

# Compute BESM_F metric
BESM = compute_besm(similarity_data)

print(f"\n🎯 Final Better Estimated Similarity Metric (BESM): {BESM:.4f}")