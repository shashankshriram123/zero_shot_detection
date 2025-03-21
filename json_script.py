import json
import subprocess

# Load the JSON file
json_file = "/home/sshriram2/mi3Testing/zero_shot_detection/formatted_omnivlm.json"
with open(json_file, "r") as f:
    data = json.load(f)

def run_all_videos():
    for video_id, details in data.items():
        nouns_list = details["nouns_list"]
        anomaly_list = details["anomaly_list"]

        # Convert lists to string representation for subprocess call
        nouns_str = str(nouns_list)
        anomaly_str = str(anomaly_list)

        # Construct the command to run test.py with extracted values
        command = f'python3 test.py "{video_id}" "{nouns_str}" "{anomaly_str}"'

        print(f"Running command: {command}")
        
        # Execute the script
        subprocess.run(command, shell=True)

# Run the pipeline for all videos
run_all_videos()
