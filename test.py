from run_clip import process_video
import sys
import ast

def run_pipeline(video_id, nouns_list, anomaly_list):
    """
    Runs the CLIP processing pipeline for a given video.
    """
    video_folder = f"/home/sshriram2/mi3Testing/zero_shot_detection/extractedSnippets/{video_id}"
    output_dir = "./output_similarity_scores"

    # Convert string input back to list format
    nouns_list = ast.literal_eval(nouns_list)
    anomaly_list = ast.literal_eval(anomaly_list)

    # Call `process_video` with extracted values
    process_video(video_folder, nouns_list, anomaly_list, output_dir)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python test.py <video_id> '<nouns_list>' '<anomaly_list>'")
        sys.exit(1)

    video_id = sys.argv[1]
    nouns_list = sys.argv[2]
    anomaly_list = sys.argv[3]

    run_pipeline(video_id, nouns_list, anomaly_list)
