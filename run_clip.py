import os
import torch
import open_clip
import gc
import numpy as np
from PIL import Image
from torchvision import transforms
import subprocess
import json

def process_video(video_folder: str, nouns_list: list, anomaly_list: list, output_dir: str):
    """
    Processes all frames for a video, computes similarity scores, and saves outputs.
    Calls visualization and mapping scripts automatically.
    """
    if not os.path.isdir(video_folder):
        print(f"❌ Error: Video folder '{video_folder}' not found!")
        return

    if not nouns_list:
        print("❌ Error: No hazard labels provided!")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using Device: {device}")

    gc.collect()
    torch.cuda.empty_cache()

    model = open_clip.create_model("ViT-H-14", pretrained="laion2b_s32b_b79k", device=device)
    image_size = model.visual.image_size
    preprocess = open_clip.image_transform(image_size, is_train=False)

    tokenizer = open_clip.get_tokenizer("ViT-H-14")
    text_tokens = tokenizer(nouns_list).to(device)

    with torch.no_grad():
        text_features = model.encode_text(text_tokens).to(torch.float32)
        text_features /= text_features.norm(dim=-1, keepdim=True)

    video_name = os.path.basename(video_folder)
    video_output_dir = os.path.join(output_dir, video_name)
    os.makedirs(video_output_dir, exist_ok=True)

    print(f"\n📽️ Processing Video: {video_name}")

    for frame_folder in sorted(os.listdir(video_folder)):
        frame_path = os.path.join(video_folder, frame_folder)
        if not os.path.isdir(frame_path):
            continue

        print(f"\n📸 Processing Frame: {frame_folder}")

        batch_images = []
        track_ids = []

        for snippet_file in sorted(os.listdir(frame_path)):
            snippet_path = os.path.join(frame_path, snippet_file)
            if not snippet_file.endswith(".jpg"):
                continue

            track_id = snippet_file.split(".")[0]

            try:
                image = Image.open(snippet_path).convert("RGB")
                width, height = image.size
                area = width * height

                if (width >= 175 and height >= 175) or area > 35000:
                    track_ids.append(track_id)
                    image = preprocess(image)
                    batch_images.append(image)
                else:
                    print(f"⏩ Skipping {snippet_file} due to low resolution ({width}x{height})")

            except Exception as e:
                print(f"⚠️ ERROR: Failed to process {snippet_file}: {e}")

        if len(batch_images) == 0:
            continue

        batch_tensor = torch.stack(batch_images).to(device).to(torch.float32)

        with torch.no_grad():
            image_features = model.encode_image(batch_tensor).to(torch.float32)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            similarity_scores = (image_features @ text_features.T).cpu().numpy()

        frame_num = frame_folder.split("_")[-1]

        similarity_file = os.path.join(video_output_dir, f"{video_name}_frame_{frame_num}_similarity.npy")
        track_ids_file = os.path.join(video_output_dir, f"{video_name}_frame_{frame_num}_track_ids.npy")
        hazard_labels_file = os.path.join(video_output_dir, f"{video_name}_frame_{frame_num}_hazard_labels.npy")

        np.save(similarity_file, np.array(similarity_scores))
        np.save(track_ids_file, np.array(track_ids, dtype=object))
        np.save(hazard_labels_file, np.array(nouns_list, dtype=object))

        print(f"✅ Saved similarity data for {video_name} - Frame {frame_num}.")

        subprocess.run(["python3", "visualize_data.py", similarity_file, track_ids_file, hazard_labels_file])

    print(f"\n🔄 Generating mappings for {video_name}...")
    subprocess.run(["python3", "print_frame_data.py", video_name, json.dumps(anomaly_list)])