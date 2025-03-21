import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Load saved data
similarity_matrix = np.load("./similarity_video_0180.npy")  # Full similarity matrix (num_tracks, num_hazards)
track_ids = np.load("./track_ids.npy", allow_pickle=True)  # Track object names
hazard_labels = np.load("./hazard_labels.npy", allow_pickle=True)  # Hazard nouns

# Ensure matrix shape matches expected dimensions
assert similarity_matrix.shape == (len(track_ids), len(hazard_labels)), \
    f"Matrix shape mismatch! Expected ({len(track_ids)}, {len(hazard_labels)}), got {similarity_matrix.shape}"

# Convert similarity matrix to a DataFrame for readability
df_similarity = pd.DataFrame(similarity_matrix, index=track_ids, columns=hazard_labels)

# Print the similarity matrix
print("\n--- Full Similarity Matrix ---\n")
print(df_similarity)

# Compute row-wise top 10% threshold
top_10_mask = np.zeros_like(similarity_matrix, dtype=bool)

for i in range(similarity_matrix.shape[0]):  # Iterate over tracks (rows)
    num_high_values = max(1, int(0.1 * similarity_matrix.shape[1]))  # At least 1 value per row
    threshold = np.percentile(similarity_matrix[i], 90)  # Compute dynamic 90th percentile threshold
    
    # Mark values that are above or equal to the threshold
    top_10_mask[i] = similarity_matrix[i] >= threshold

    # Ensure at least `num_high_values` are selected (fallback if percentile filtering is too strict)
    sorted_indices = np.argsort(similarity_matrix[i])[-num_high_values:]  # Get top 10% indices
    top_10_mask[i, sorted_indices] = True

# Convert top 10% mask to a DataFrame for readability
df_top_10_mask = pd.DataFrame(top_10_mask, index=track_ids, columns=hazard_labels)

# Print the top 10% mask
print("\n--- Top 10% Highlighted Values (True = Selected) ---\n")
print(df_top_10_mask)

# Create the mapping dictionary
mapping = {}

for i in range(similarity_matrix.shape[0]):  # Iterate over tracks
    for j in range(similarity_matrix.shape[1]):  # Iterate over hazards
        if top_10_mask[i, j]:  # If it's in the top 10%
            hazard = hazard_labels[j]  # Hazard name
            if hazard not in mapping:
                mapping[hazard] = []
            mapping[hazard].append(i)  # Add track ID index

# Print the mapping dictionary
print("\n--- Mapping of Detected Objects to Descriptions ---\n")
print(mapping)

# Create a colormap
custom_cmap = sns.color_palette("coolwarm", as_cmap=True)

# Plot heatmap
plt.figure(figsize=(12, 8))
ax = sns.heatmap(similarity_matrix, annot=True, cmap=custom_cmap, fmt=".2f",
                 xticklabels=hazard_labels, yticklabels=track_ids, cbar=True)

# Highlight top 10% values with a bold black box
for i in range(similarity_matrix.shape[0]):
    for j in range(similarity_matrix.shape[1]):
        if top_10_mask[i, j]:  # If value is in the top 10%
            ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False, edgecolor='black', lw=2))

# Labels and title
plt.xlabel("Hazard Labels")
plt.ylabel("Track Objects")
plt.title("CLIP Similarity Scores (Track Objects vs Hazards)\nTop 10% Values Highlighted")

# Show plot
plt.show()
