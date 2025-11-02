import pandas as pd
from datasets import load_dataset
import os

def filter_csv_files():
    """
    Filters CSV files in the batch_outputs directory to only include questions
    present in the EleutherAI/wmdp_bio_robust_mcqa dataset.
    """
    # Load the Hugging Face dataset
    print("Loading Hugging Face dataset...")
    dataset = load_dataset("EleutherAI/wmdp_bio_robust_mcqa", split="test")
    robust_questions = set(dataset["question"])
    print(f"Loaded {len(robust_questions)} robust questions.")

    # Directory containing the CSV files
    input_dir = "batch_outputs"
    output_dir = "batch_outputs_filtered"
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # List of CSV files to process
    csv_files = [f for f in os.listdir(input_dir) if f.endswith(".csv") or f == "real_words_sciencycsv"]

    for filename in csv_files:
        input_path = os.path.join(input_dir, filename)
        output_filename = os.path.splitext(filename)[0] + "_filtered.csv"
        output_path = os.path.join(output_dir, output_filename)
        
        print(f"Processing {input_path}...")
        
        try:
            # Read the CSV file
            df = pd.read_csv(input_path)
            
            # Check if 'original' column exists
            if "original" not in df.columns:
                print(f"  'original' column not found in {filename}. Skipping.")
                continue

            # Filter the DataFrame
            initial_rows = len(df)
            filtered_df = df[df["original"].isin(robust_questions)]
            final_rows = len(filtered_df)
            
            # Save the filtered DataFrame
            filtered_df.to_csv(output_path, index=False)
            
            print(f"  Filtered {initial_rows} rows down to {final_rows}.")
            print(f"  Saved filtered file to {output_path}")

        except Exception as e:
            print(f"  Error processing {filename}: {e}")

if __name__ == "__main__":
    filter_csv_files()
