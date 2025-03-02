import torch
import clip
from PIL import Image

# Check if MPS (Apple GPU) is available
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

# Load CLIP model
model, preprocess = clip.load("ViT-B/32", device=device)

# Load an image
image = preprocess(Image.open("/Users/shashankshriram/Desktop/zero_shot_detection/extractedSnippets/video_0050/frame_169/track1.jpg")).unsqueeze(0).to(device)

# Tokenize and process text
text = clip.tokenize(["a photo of a cat", "a photo of a truck", "truck"]).to(device)

# Compute similarity
with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    similarity = (image_features @ text_features.T).softmax(dim=-1)

print("Similarity Scores:", similarity)
