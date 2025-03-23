import os
from datasets import load_dataset

def display_huggingface_dataset(dataset_name="librarian-bots/dataset-columns", split="train"):
    """
    Read and display a dataset from Huggingface
    """
    # Load the dataset
    dataset = load_dataset(dataset_name)
    
    # Get the specified split
    data = dataset[split]
    
    # Convert to pandas for better display
    df = data.to_pandas()
    
    print(f"\nDataset: {dataset_name}")
    print(f"Split: {split}")
    print("\nFirst few rows:")
    print(df.head())
    
    return df

def list_dataset_splits(dataset_name="librarian-bots/dataset-columns"):
    """
    List all available splits in a Huggingface dataset
    """
    dataset = load_dataset(dataset_name)
    print(f"\nAvailable splits in {dataset_name}:")
    for split in dataset.keys():
        print(f"- {split}")
