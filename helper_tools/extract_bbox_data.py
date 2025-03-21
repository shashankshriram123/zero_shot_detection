import pickle

# Load the pickle file
pickle_file_path = "/Users/shashankshriram/Desktop/zero_shot_detection/annotations_public.pkl" 
with open(pickle_file_path, "rb") as f:
    data = pickle.load(f)

# Function to extract bounding box data from `traffic_scene` and `challenge_object`
def extract_bounding_box_data(data):
    all_bounding_boxes = []

    for video_id, video_data in data.items():
        for frame_id, frame_content in video_data.items():
            # Extract `traffic_scene` bounding boxes
            traffic_scene = frame_content.get("traffic_scene", [])
            for obj in traffic_scene:
                bounding_box_info = {
                    "video_id": video_id,
                    "frame_id": int(frame_id),
                    "track_id": obj["track_id"],
                    "bbox": obj["bbox"],
                    "type": "traffic_scene",  # Distinguish the source
                    "attributes": obj.get("attributes", {})
                }
                all_bounding_boxes.append(bounding_box_info)

            # Extract `challenge_object` bounding boxes
            challenge_object = frame_content.get("challenge_object", [])
            for obj in challenge_object:
                bounding_box_info = {
                    "video_id": video_id,
                    "frame_id": int(frame_id),
                    "track_id": obj["track_id"],
                    "bbox": obj["bbox"],
                    "type": "challenge_object",  # Distinguish the source
                    "attributes": obj.get("attributes", {})
                }
                all_bounding_boxes.append(bounding_box_info)

    return all_bounding_boxes

# Extract bounding box data
bounding_boxes = extract_bounding_box_data(data)

# Print summary information
print(f"Total videos: {len(data)}")
total_frames = sum(len(video_data) for video_data in data.values())
print(f"Total frames: {total_frames}")
print(f"Total bounding boxes: {len(bounding_boxes)} \n\n\n")


# Save the bounding box data to a file
output_file_path = "/Users/shashankshriram/Desktop/zero_shot_detection/bounding_boxes_output.pkl"  # You can change the output file name and path
with open(output_file_path, "wb") as f:
    pickle.dump(bounding_boxes, f)
print(f"Bounding box data saved to {output_file_path}")


# Example: Print the first few bounding boxes
for box in bounding_boxes[:5]:  # Adjust the range for more boxes
    print(box)