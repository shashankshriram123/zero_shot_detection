import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import re
import sys  # To accept command-line arguments
import sys
print(f"Using Python: {sys.executable}")

def extract_track_number(track_label):
    """Extracts the numerical ID from track labels like 'track8' -> 8"""
    match = re.search(r'\d+', track_label)
    return int(match.group()) if match else track_label

def process_similarity_matrix(similarity_file, track_ids_file, hazard_labels_file):
    """
    Loads similarity scores, visualizes heatmap, and prints hazard mappings.

    Args:
        similarity_file (str): Path to the .npy file storing similarity scores.
        track_ids_file (str): Path to the .npy file storing track IDs.
        hazard_labels_file (str): Path to the .npy file storing hazard labels.
    """
    # Load data
    similarity_matrix = np.load(similarity_file)  
    track_ids = np.load(track_ids_file, allow_pickle=True)  
    hazard_labels = np.load(hazard_labels_file, allow_pickle=True)  

    # Ensure matrix shape matches expected dimensions
    num_tracks, num_hazards = similarity_matrix.shape
    assert num_tracks == len(track_ids) and num_hazards == len(hazard_labels), \
        f"Matrix shape mismatch! Expected ({len(track_ids)}, {len(hazard_labels)}), got {similarity_matrix.shape}"

    # Compute row-wise top 10% mask
    top_10_mask = np.zeros_like(similarity_matrix, dtype=bool)

    for i in range(num_tracks):  
        num_high_values = max(1, int(0.1 * num_hazards))  
        threshold = np.percentile(similarity_matrix[i], 90)  
        
        row_mask = similarity_matrix[i] >= threshold
        top_10_mask[i] = row_mask  
        
        if row_mask.sum() < num_high_values:
            sorted_indices = np.argsort(similarity_matrix[i])[-num_high_values:]
            top_10_mask[i, sorted_indices] = True  

    # Construct the mapping using actual track numbers
    mapping = {}
    for i in range(num_tracks):  
        track_number = extract_track_number(track_ids[i])  
        for j in range(num_hazards):  
            if top_10_mask[i, j]:  
                hazard = hazard_labels[j]  
                if hazard not in mapping:
                    mapping[hazard] = []
                mapping[hazard].append(track_number)  

    print("\n--- Mapping of Detected Objects to Descriptions ---\n")
    print(mapping)

    # Create a colormap
    custom_cmap = sns.color_palette("coolwarm", as_cmap=True)

    # Plot heatmap
    plt.figure(figsize=(12, 8))
    ax = sns.heatmap(similarity_matrix, annot=True, cmap=custom_cmap, fmt=".2f",
                     xticklabels=hazard_labels, yticklabels=track_ids, cbar=True)

    # Highlight top 10% values with a bold black box
    for i in range(num_tracks):
        for j in range(num_hazards):
            if top_10_mask[i, j]:  
                ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False, edgecolor='black', lw=2))

    frame_name = similarity_file.split("_")[-1].split(".")[0]  
    plt.xlabel("Hazard Labels")
    plt.ylabel("Track Objects")
    plt.title(f"CLIP Similarity Scores (Frame {frame_name})\nTop 10% Values Highlighted")

    plt.show()

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python visualize_data.py <similarity_file.npy> <track_ids.npy> <hazard_labels.npy>")
        sys.exit(1)

    process_similarity_matrix(sys.argv[1], sys.argv[2], sys.argv[3])
