import pandas as pd



def rename_and_clean_scenes(csv_path, output_path):
    # Load the CSV file
    df = pd.read_csv(csv_path)
    
    # Ensure the 'Scene' column exists
    if 'Scene' in df.columns:
        # Convert Scene column to integers and rename to 'video_000n' format
        df['Scene'] = df['Scene'].astype(int).apply(lambda x: f"video_{x:04d}")
    else:
        print("Error: 'Scene' column not found in the CSV file.")
        return
    
    # Keep only the required columns
    columns_to_keep = ["Scene", "Description Summary", "Ideal_Hazard_1", "Ideal_Hazard_2", "Ideal_Hazard_3", "Ideal_Hazard_4"]
    df = df[columns_to_keep]
    
    # Save the cleaned data
    df.to_csv(output_path, index=False)
    print(f"Final cleaned data saved to: {output_path}")

# Example usage
input_file = "/Users/shashankshriram/Desktop/coool/zero_shot_detection/cooolerGroundTruth.csv"  # Update this to the correct path
output_file = "/Users/shashankshriram/Desktop/coool/zero_shot_detection/cooolerGroundTruth_cleaned.csv"
rename_and_clean_scenes(input_file, output_file)
