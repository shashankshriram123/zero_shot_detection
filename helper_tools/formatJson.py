import json
import re
# Load the input file
input_file = "/Users/shashankshriram/Desktop/coool/zero_shot_detection/final_omnivlm.json"
output_file = "./formatted_omnivlm.json"

# Read the JSON data
with open(input_file, "r") as f:
    data = json.load(f)

# Function to clean and convert Python list strings into actual lists
def extract_list_from_string(list_string):
    # Remove ```python and ``` formatting if present
    list_string = re.sub(r"```python|\n```", "", list_string).strip()
    
    # Convert the string representation of a list to an actual Python list
    try:
        return eval(list_string) if list_string.startswith("[") and list_string.endswith("]") else []
    except:
        return []

# Process and reformat the data
formatted_data = {}
for video, values in data.items():
    formatted_data[video] = {
        "nouns_list": extract_list_from_string(values[0]),  # Convert to actual list
        "anomaly_list": extract_list_from_string(values[1])  # Convert to actual list
    }

# Write the cleaned JSON to an output file
with open(output_file, "w") as f:
    json.dump(formatted_data, f, indent=4)

print(f"Formatted JSON saved to {output_file}")