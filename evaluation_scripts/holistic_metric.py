import json

# Load the similarity results
SIMILARITY_FILE = "/home/sshriram2/mi3Testing/zero_shot_detection/jsonresults/similarity_scores_output.json"

with open(SIMILARITY_FILE, "r") as f:
    similarity_data = json.load(f)



def compute_holistic_metric(similarity_data):
    """
    Computes the holistic metric M by:
    1. Taking the average of the highest & lowest value in description/hazard similarity if multiple values exist.
    2. Keeping the single value as is if only one unique similarity exists.
    3. Averaging the final description and hazard similarity per video.
    4. Printing debug info with warnings for low scores (below 0.5).
    """
    video_scores = []

    for video_id, scores in similarity_data.items():
        description_sim = scores.get("description_similarity", [])
        hazard_sim = scores.get("hazard_similarity", [])

        # Convert to unique sets and remove duplicates
        unique_desc_sim = list(set(description_sim))
        unique_hazard_sim = list(set(hazard_sim))

        print(f"\n🎬 Processing Video: {video_id}")

        # Compute adjusted description similarity
        if len(unique_desc_sim) > 1:
            max_desc = max(unique_desc_sim)
            min_desc = min(unique_desc_sim)
            desc_sim_final = (max_desc + min_desc) / 2
            print(f"📊 Description Similarity: Averaging max ({max_desc:.4f}) and min ({min_desc:.4f}) → {desc_sim_final:.4f}")
        elif unique_desc_sim:  # If only one value exists
            desc_sim_final = unique_desc_sim[0]
            print(f"✅ Single Description Similarity Value: {desc_sim_final:.4f}")
        else:
            desc_sim_final = 0  # Default if empty
            print("⚠️ No Description Similarity Data Found! Setting to 0.")

        # Compute adjusted hazard similarity
        if len(unique_hazard_sim) > 1:
            max_haz = max(unique_hazard_sim)
            min_haz = min(unique_hazard_sim)
            haz_sim_final = (max_haz + min_haz) / 2
            print(f"📊 Hazard Similarity: Averaging max ({max_haz:.4f}) and min ({min_haz:.4f}) → {haz_sim_final:.4f}")
        elif unique_hazard_sim:  # If only one value exists
            haz_sim_final = unique_hazard_sim[0]
            print(f"✅ Single Hazard Similarity Value: {haz_sim_final:.4f}")
        else:
            haz_sim_final = 0  # Default if empty
            print("⚠️ No Hazard Similarity Data Found! Setting to 0.")

        # Compute final similarity score for the video
        final_video_score = (desc_sim_final + haz_sim_final) / 2

        # Warning for low scores 🚨
        if final_video_score < 0.5:
            print(f"⚠️🚨 LOW AVERAGE SCORE ({final_video_score:.4f}) FOR VIDEO {video_id} ⚠️🚨")

        video_scores.append(final_video_score)

    # Compute final holistic metric
    holistic_metric = sum(video_scores) / len(video_scores) if video_scores else 0

    return holistic_metric

# Compute holistic metric
M = compute_holistic_metric(similarity_data)

print(f"\n🎯 Final Holistic Metric (M): {M:.4f}")