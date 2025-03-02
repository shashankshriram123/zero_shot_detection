import os
import torch
import clip
from PIL import Image
import numpy as np

# Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using Device: {device}")

model, preprocess = clip.load("ViT-B/32", device=device)

# Define text prompts
nouns_list = ["dog", "truck", "person", "traffic cone", "pothole", "debris"]
text_tokens = clip.tokenize(nouns_list).to(device)

# Encode text embeddings once
with torch.no_grad():
    text_features = model.encode_text(text_tokens).to(torch.float32)  # Force full precision
    text_features /= text_features.norm(dim=-1, keepdim=True)  # Normalize

# Path to extracted snippets
base_snippets_folder = "/Users/shashankshriram/Desktop/zero_shot_detection/extractedSnippets"
video_id = "video_0050"
video_path = os.path.join(base_snippets_folder, video_id)

if not os.path.isdir(video_path):
    print(f"Error: Video folder {video_id} not found!")
    exit()

print(f"\nProcessing Video: {video_id}")

track_scores = {}  # Stores similarity scores per track
highest_scores = {}  # Stores highest similarity score and frame number

# Iterate through all frames
for frame_folder in sorted(os.listdir(video_path)):
    frame_path = os.path.join(video_path, frame_folder)
    
    if not os.path.isdir(frame_path):
        continue

    frame_num = int(frame_folder.split("_")[-1])  # Extract frame number
    print(f"\nProcessing Frame: {frame_folder}")

    # Load all images in the current frame
    batch_images = []
    track_ids = []  # Store track IDs for alignment

    for snippet_file in sorted(os.listdir(frame_path)):
        snippet_path = os.path.join(frame_path, snippet_file)
        if not snippet_file.endswith(".jpg"):
            continue
        
        track_id = snippet_file.split(".")[0]  # Example: "track1"
        track_ids.append(track_id)  # Save track ID

        try:
            image = Image.open(snippet_path)
            image = preprocess(image)  # Preprocess without adding batch dimension
            batch_images.append(image)
        except Exception as e:
            print(f"ERROR: Failed to process {snippet_file}: {e}")

    # Skip empty batches
    if len(batch_images) == 0:
        continue

    # Convert batch to tensor
    batch_tensor = torch.stack(batch_images).to(device)  # Shape: (batch_size, 3, 224, 224)

    print(f"  Batch Processing {len(batch_tensor)} Images at Once")  # Debugging Output

    # Compute embeddings in a batch
    with torch.no_grad():
        image_features = model.encode_image(batch_tensor).to(torch.float32)  # Force precision
        image_features /= image_features.norm(dim=-1, keepdim=True)  # Normalize

        # Compute similarity scores for all images in batch
        similarity_scores = (image_features @ text_features.T).cpu().numpy()  # Shape: (batch_size, num_nouns)

    # Store similarity scores for each track in the batch
    for i, track_id in enumerate(track_ids):
        similarity = similarity_scores[i]

        # Track highest similarity
        max_sim = np.max(similarity)
        max_index = np.argmax(similarity)

        if track_id not in highest_scores or highest_scores[track_id][0] < max_sim:
            highest_scores[track_id] = (max_sim, frame_num, nouns_list[max_index])  # (max_sim, frame_number, noun)

        # Store per-frame scores
        if track_id not in track_scores:
            track_scores[track_id] = []
        track_scores[track_id].append(similarity)

# Compute final similarity matrix
if track_scores:
    all_tracks = []
    track_ids_sorted = sorted(track_scores.keys())  # Ensure order

    highest_similarity_list = []  # Store highest similarity details

    for track_id in track_ids_sorted:
        if track_scores[track_id] == "detected":
            all_tracks.append(["detected"] * len(nouns_list))  # Mark entire row as detected
        else:
            avg_score = np.mean(track_scores[track_id], axis=0)  # Average only over existing frames
            all_tracks.append(avg_score)  # Shape: (num_nouns,)

        # Store highest similarity info (similarity, frame number, detected noun)
        if track_id in highest_scores:
            highest_similarity_list.append((track_id, highest_scores[track_id][0], highest_scores[track_id][1], highest_scores[track_id][2]))
        else:
            highest_similarity_list.append((track_id, 0.0, -1, "None"))  # Default if no match

    # Convert to structured NumPy array with mixed data types
    avg_similarity_matrix = np.array(all_tracks, dtype=object)
    highest_similarity_matrix = np.array(highest_similarity_list, dtype=object)

    # Save track IDs separately
    track_ids_file = f"/Users/shashankshriram/Desktop/zero_shot_detection/track_ids.npy"
    np.save(track_ids_file, np.array(track_ids_sorted))

    # Save final similarity matrices
    avg_output_file = f"/Users/shashankshriram/Desktop/zero_shot_detection/similarity_{video_id}_average.npy"
    highest_output_file = f"/Users/shashankshriram/Desktop/zero_shot_detection/similarity_{video_id}_highest.npy"

    np.save(avg_output_file, avg_similarity_matrix)
    np.save(highest_output_file, highest_similarity_matrix)

    print(f"\nFinal Averaged Similarity Matrix Saved: {avg_output_file}")
    print(f"Highest Similarity Scores Saved: {highest_output_file}")
    print(f"Track Object Names Saved: {track_ids_file}")

print(f"\nCLIP Similarity Analysis Complete for Video: {video_id}!")
print(f"Total Unique Track Objects Processed: {len(track_scores)}")
