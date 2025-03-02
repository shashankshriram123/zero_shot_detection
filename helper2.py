import numpy as np

# Load the saved highest similarity matrix
highest_file_path = "/Users/shashankshriram/Desktop/zero_shot_detection/similarity_video_0050_highest.npy"
highest_similarity_matrix = np.load(highest_file_path, allow_pickle=True)

# Print results
print("Highest Similarity Matrix Shape:", highest_similarity_matrix.shape)
print("\nTrack Object | Max Similarity | Frame Number | Noun Detected")

# Iterate through the matrix, correctly unpacking all 4 columns
for track_id, max_similarity, frame_number, noun_detected in highest_similarity_matrix:
    print(f"{track_id}: ({float(max_similarity):.4f}, Frame {int(frame_number)}, '{noun_detected}')")
