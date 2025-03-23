from datasets import load_dataset
import os

def convert_huggingface_dataset_to_csv(dataset_name="librarian-bots/dataset-columns", csv_output_folder="./converted_csv"):
    os.makedirs(csv_output_folder, exist_ok=True)

    # Load the dataset (this fetches the entire dataset structure)
    dataset = load_dataset(dataset_name)

    converted_paths = []
    
    # Iterate over all splits (e.g., "train", "test", "validation")
    for split in dataset.keys():
        df = dataset[split].to_pandas()  # Convert to Pandas
        csv_path = os.path.join(csv_output_folder, f"{split}.csv")
        df.to_csv(csv_path, index=False)  # Save as CSV
        converted_paths.append(csv_path)
        print(f"Converted {split} split to CSV: {csv_path}")

    return converted_paths




