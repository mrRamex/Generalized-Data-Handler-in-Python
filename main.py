import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_from_disk
from torch.utils.data import DataLoader
from dataUpload import convert_huggingface_dataset_to_csv
import os
from datasets import load_dataset, load_from_disk


# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ColumnClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(ColumnClassifier, self).__init__()
        self.embedding = nn.Embedding(128, input_size)
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        x = self.embedding(x)
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])

class SmartDataHandler:
    def __init__(self):
        self.column_types = ['numeric', 'categorical', 'datetime', 'text']
        self.model = ColumnClassifier(
            input_size=32,
            hidden_size=64,
            num_classes=len(self.column_types)
        ).to(device)
        
    def prepare_training_data(self):
        # Example column names and their types
        training_examples = {
            'numeric': [
                'amount', 'price', 'quantity', 'age', 'height', 'weight', 'score', 'id', 'count', 'total', 
                'likes', 'downloads', 'version', 'user_age', 'price_usd', 'license', 'tags', 'rating',
                'views', 'followers', 'points', 'rank', 'percentage', 'duration', 'value', 'balance',
                'threshold', 'limit', 'sequence', 'priority_number'
            ],
            'datetime': [
                'date', 'timestamp', 'created_at', 'updated_at', 'birth_date', 'start_time', 'end_time', 
                'login_time', 'order_date', 'transaction_date', 'last_modified', 'expiry_date', 'due_date',
                'schedule_time', 'registration_date', 'hub_id', 'config_name', 'status_code', 'completion_date',
                'delivery_time', 'arrival_date'
            ],
            'categorical': [
                'status', 'category', 'type', 'gender', 'color', 'size', 'country', 'state', 'grade', 
                'level', 'language', 'category_type', 'priority', 'role', 'department', 'permission_level',
                'created_at', 'last_modified', 'status_type', 'access_level', 'user_type'
            ],
            'text': [
                'name', 'description', 'title', 'address', 'comment', 'message', 'email', 'notes', 'summary', 
                'product_description', 'email_address', 'features', 'biography', 'feedback', 'review',
                'content', 'details', 'specifications', 'remarks', 'instructions'
            ]
        }
        
        all_names = []
        all_labels = []
        
        for label_idx, (col_type, examples) in enumerate(training_examples.items()):
            for name in examples:
                processed_name = self.preprocess_column_name(name)
                all_names.append(processed_name)
                all_labels.append(label_idx)
        
        names_tensor = torch.stack(all_names)
        labels_tensor = torch.tensor(all_labels, dtype=torch.long)
        
        dataset = torch.utils.data.TensorDataset(names_tensor, labels_tensor)
        train_loader = DataLoader(dataset, batch_size=16, shuffle=True)
        
        return train_loader
    
    def prepare_huggingface_data(dataset_path):
        # Load the dataset from disk
        dataset = load_from_disk(dataset_path)
        
        # Convert to format compatible with SmartDataHandler
        all_names = []
        all_labels = []
        
        for example in dataset:
            processed_name = handler.preprocess_column_name(example['column_name'])
            all_names.append(processed_name)
            all_labels.append(example['label'])
        
        names_tensor = torch.stack(all_names)
        labels_tensor = torch.tensor(all_labels, dtype=torch.long)

        # Create DataLoader
        dataset = torch.utils.data.TensorDataset(names_tensor, labels_tensor)
        return DataLoader(dataset, batch_size=16, shuffle=True)
    
    def prepare_arrow_training_data(self, dataset_name="librarian-bots/dataset-columns"):
        """
        Prepare training data from Huggingface dataset
        """
        dataset = load_dataset(dataset_name)
        train_data = dataset['train']
        
        all_names = []
        all_labels = []
        
        # Use the actual columns from the dataset
        columns_to_process = ['hub_id', 'config_name', 'likes', 'downloads', 'tags', 
                            'language', 'license', 'created_at', 'last_modified', 
                            'features', 'version']
        
        for example in train_data:
            for column in columns_to_process:
                processed_name = self.preprocess_column_name(column)
                all_names.append(processed_name)
                # Map column types to our classification system
                if column in ['likes', 'downloads', 'tags', 'version']:
                    label = 0  # numeric
                elif column in ['created_at', 'last_modified']:
                    label = 1  # datetime
                elif column in ['hub_id', 'config_name', 'language', 'license']:
                    label = 2  # categorical
                else:
                    label = 3  # text
                all_labels.append(label)
        
        names_tensor = torch.stack(all_names)
        labels_tensor = torch.tensor(all_labels, dtype=torch.long)
        
        dataset = torch.utils.data.TensorDataset(names_tensor, labels_tensor)
        return DataLoader(dataset, batch_size=16, shuffle=True)



    def train_model(self, train_data, epochs):
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters())
        
        # Track metrics
        losses = []
        accuracies = []
        epoch_points = list(range(1, epochs + 1))  # Create x-axis points
        
        self.model.train()
        for epoch in range(epochs):
            epoch_loss = 0
            correct = 0
            total = 0
            
            for names, labels in train_data:
                names = names.to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()
                outputs = self.model(names)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            
            avg_loss = epoch_loss / len(train_data)
            accuracy = 100 * correct / total
            
            losses.append(avg_loss)
            accuracies.append(accuracy)
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')
        
        # Plot metrics with proper epoch labeling
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(epoch_points, losses)
        plt.title(f'Training Loss Over {epochs} Epochs')
        plt.xlabel('Epoch Number')
        plt.ylabel('Loss')
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(epoch_points, accuracies)
        plt.title(f'Training Accuracy Over {epochs} Epochs')
        plt.xlabel('Epoch Number')
        plt.ylabel('Accuracy (%)')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(f'training_metrics_{epochs}epochs.png')
        plt.show()
    
    def preprocess_column_name(self, name, max_length=20):
        ascii_indices = [ord(c) % 128 for c in name.lower()]
        if len(ascii_indices) < max_length:
            ascii_indices += [0] * (max_length - len(ascii_indices))
        return torch.tensor(ascii_indices[:max_length], device=device)
    
    def predict_column_type(self, column_name):
        self.model.eval()
        with torch.no_grad():
            x = self.preprocess_column_name(column_name).unsqueeze(0)
            output = self.model(x)
            predicted = torch.argmax(output, dim=1)
            return self.column_types[predicted.item()]
    
    def analyze_csv(self, filepath):
        df = pd.read_csv(filepath, low_memory=False)
        column_predictions = {}
        
        for column in df.columns:
            predicted_type = self.predict_column_type(column)
            column_predictions[column] = predicted_type
                
        return column_predictions


    def save_model(self, path='column_classifier.pth'):
        torch.save(self.model.state_dict(), path)
        
    def load_model(self, path='column_classifier.pth'):
        self.model.load_state_dict(torch.load(path, weights_only=True))
        self.model.to(device)




def test_saved_model(test_columns):
    """
    Test the saved model with sample column names
    """
    handler = SmartDataHandler()
    handler.load_model('trained_column_classifier.pth')
    print("\nTesting saved model predictions:")
    print("-" * 40)
    
    for column in test_columns:
        prediction = handler.predict_column_type(column)
        print(f"Column '{column}' -> Predicted type: {prediction}")

def main():
    handler = SmartDataHandler()
    
    # Convert all arrow files in folder
    csv_paths = convert_huggingface_dataset_to_csv()
    
    # Prepare training data from arrow files
    #train_loader = handler.prepare_arrow_training_data()
    
    # Train the model
    #handler.train_model(train_loader, epochs=90)

    # Prepare training data
    train_loader = handler.prepare_training_data()
    
    ## Train the model
    handler.train_model(train_loader, epochs=90)
    
    # Save the trained model
    handler.save_model('trained_column_classifier.pth')
    print("Model saved successfully to 'trained_column_classifier.pth'")
    
    # Test the saved model with some sample columns
    test_columns = [
        'user_age',
        'transaction_date',
        'product_description',
        'category_type',
        'email_address',
        'price_usd',
        'status_code'
    ]
    test_saved_model(test_columns)
    
    # Analyze each converted CSV file
    for csv_path in csv_paths:
        print(f"\nAnalyzing {os.path.basename(csv_path)}:")
        predictions = handler.analyze_csv(csv_path)
        
        for column, predicted_type in predictions.items():
            print(f"Column '{column}' is predicted to be: {predicted_type}")

if __name__ == "__main__":
    main()
