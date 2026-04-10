import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import mlflow
import mlflow.pytorch

torch.manual_seed(42)

class ODIRDataset(Dataset):
    def __init__(self, excel_path, data_dir, transform=None):
        df = pd.read_csv(excel_path)
        self.data_dir = data_dir
        self.transform = transform
        
        self.classes = ['Normal', 'Diabetes', 'Glaucoma', 'Cataract', 'AMD', 'Hypertension', 'Myopia', 'Others']
        
        # 1. Look inside the Azure folder and get a list of what's ACTUALLY there
        try:
            existing_files = set(os.listdir(data_dir))
            print(f"DEBUG: Successfully found {len(existing_files)} files in the Azure container.")
        except Exception as e:
            print(f"CRITICAL: Could not read Azure directory. Error: {e}")
            existing_files = set()

        self.samples = []
        # 2. Add "_clahe" to the CSV names to match the physical files in Azure
        for _, row in df.iterrows():
            left_img_name = str(row['Left-Fundus']).replace('.jpg', '_clahe.jpg')
            right_img_name = str(row['Right-Fundus']).replace('.jpg', '_clahe.jpg')

            if left_img_name in existing_files:
                self.samples.append({
                    'img': left_img_name,
                    'label': self._get_label(row)
                })
            if right_img_name in existing_files:
                self.samples.append({
                    'img': right_img_name,
                    'label': self._get_label(row)
                })
                
        print(f"DEBUG: Cleaned dataset! Training on {len(self.samples)} valid images.")

    def _get_label(self, row):
        labels = [row['N'], row['D'], row['G'], row['C'], row['A'], row['H'], row['M'], row['O']]
        return labels.index(1) if 1 in labels else 7

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_path = os.path.join(self.data_dir, sample['img'])
        
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)
        return image, sample['label']

def main(args):
    print(f"DEBUG: Azure mounted the data at: {args.data_dir}")
    print("DEBUG: Let's look inside that folder...")
    print(os.listdir(args.data_dir)[:20])
    
    mlflow.autolog()
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load data from the CSV file
    full_dataset = ODIRDataset(excel_path="data.csv", data_dir=args.data_dir, transform=transform)
    
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

    # num_workers=0 prevents the multiprocessing crash
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)

    model = models.resnet50(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 8) # 8 ODIR classes

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    for epoch in range(args.epochs):
        model.train()
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        # Validation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        acc = 100 * correct / total
        print(f"Epoch {epoch+1}: Acc {acc}%")
        mlflow.log_metric("val_accuracy", acc, step=epoch)

    mlflow.pytorch.log_model(model, "model", registered_model_name="ResNet50_Ocular")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()
    main(args)