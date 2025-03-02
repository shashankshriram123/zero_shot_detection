import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load saved similarity matrix
file_path = "/Users/shashankshriram/Desktop/zero_shot_detection/similarity_video_0050_average.npy"
similarity_matrix = np.load(file_path, allow_pickle=True)  # Allow mixed data types

# Load track IDs
track_ids_file = "/Users/shashankshriram/Desktop/zero_shot_detection/track_ids.npy"
track_ids = np.load(track_ids_file, allow_pickle=True)  # Ensures correct row labels

# Define noun labels (columns)
nouns_list = ["dog", "truck", "person", "traffic cone", "pothole", "debris"]

# Ensure the matrix shape matches the labels
if similarity_matrix.shape[1] != len(nouns_list):
    print("Warning: Mismatch between matrix columns and noun labels!")

# Convert similarity matrix to float while keeping "detected" labels
float_matrix = np.zeros_like(similarity_matrix, dtype=np.float32)
mask = np.zeros_like(similarity_matrix, dtype=bool)

for i in range(len(track_ids)):  # Use track_ids to ensure correct row labels
    for j in range(len(nouns_list)):
        if similarity_matrix[i, j] == "detected":
            float_matrix[i, j] = 1.0  # Assign highest value for detected
            mask[i, j] = True  # Mark it for special coloring
        else:
            float_matrix[i, j] = similarity_matrix[i, j]  # Normal numerical score

# Create a heatmap with improved formatting
plt.figure(figsize=(10, 6))  # Larger figure
ax = sns.heatmap(float_matrix, annot=True, cmap="coolwarm",
                 xticklabels=nouns_list, yticklabels=track_ids,  # ✅ Now correctly labels track objects
                 linewidths=0.1, linecolor="black", mask=mask)

# Overlay "DETECTED" labels manually
for i in range(len(track_ids)):
    for j in range(len(nouns_list)):
        if similarity_matrix[i, j] == "detected":
            ax.text(j + 0.5, i + 0.5, "DETECTED", ha='center', va='center', color='green', fontsize=12, fontweight='bold')

# Label axes
plt.xlabel("Nouns", fontsize=14)
plt.ylabel("Track Objects", fontsize=14)
plt.title("CLIP Similarity Heatmap (Per Track Object)", fontsize=16)

# Show the heatmap
plt.show()
