import os
import cv2
import pickle

# Define paths
pickle_file = "/Users/shashankshriram/Desktop/zero_shot_detection/bounding_boxes_output.pkl"
video_folder_path = "/Users/shashankshriram/Desktop/zero_shot_detection/dataset"  # Folder containing all video files
base_output_folder = "/Users/shashankshriram/Desktop/zero_shot_detection/extractedSnippets"

# Load bounding box data from the pickle file
with open(pickle_file, "rb") as f:
    bounding_boxes = pickle.load(f)

# Get the unique video IDs from the dataset
video_ids = set(bb['video_id'] for bb in bounding_boxes)

for video_id in video_ids:
    video_path = os.path.join(video_folder_path, f"{video_id}.mp4")  # Video file name must match video_id

    # Check if video file exists
    if not os.path.exists(video_path):
        print(f"Warning: Video file {video_id}.mp4 not found. Skipping...")
        continue

    print(f"Processing video: {video_id}.mp4")

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video {video_id}.mp4")
        continue

    frame_count = 0
    while True:
        ret, frame = cap.read()

        if not ret:
            break  # Exit when video ends

        # Filter bounding boxes for the current video and frame
        relevant_bboxes = [bb for bb in bounding_boxes if bb['video_id'] == video_id and bb['frame_id'] == frame_count and bb['type'] == "challenge_object"]

        for bbox_data in relevant_bboxes:
            frame_id = bbox_data["frame_id"]
            track_id = bbox_data["track_id"]
            bbox = bbox_data["bbox"]
            x1, y1, x2, y2 = map(int, bbox)

            # Define output directory structure
            video_output_folder = os.path.join(base_output_folder, f"{video_id}")
            frame_output_folder = os.path.join(video_output_folder, f"frame_{frame_id:03d}")  # Zero-padded frame numbers

            # Create directories if they don't exist
            os.makedirs(frame_output_folder, exist_ok=True)

            # Crop the snippet
            snippet = frame[y1:y2, x1:x2]

            # Ensure snippet is not empty
            if snippet.size == 0:
                continue

            # Define filename
            snippet_filename = os.path.join(frame_output_folder, f"track{track_id}.jpg")

            # Save the snippet
            cv2.imwrite(snippet_filename, snippet)
            print(f"Saved: {snippet_filename}")

        frame_count += 1

    # Release video capture
    cap.release()
    print(f"Finished processing {video_id}.mp4\n")

cv2.destroyAllWindows()
