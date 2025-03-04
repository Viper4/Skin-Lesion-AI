import pandas as pd
import torch
import torch.nn as nn
import zipfile
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import io
import math
import os
import multiprocessing as mp
from multiprocessing import Pool


class SkinLesionNN(nn.Module):
    def __init__(self, num_metadata_features):
        super(SkinLesionNN, self).__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Convolutional layers for image processing
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            self.maxpool,  # 112x112
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            self.maxpool,  # 56x56
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            self.maxpool,  # 28x28
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            self.maxpool,  # 14x14
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            self.maxpool,  # 7x7
        )

        # Metadata processing
        self.metadata_fc = nn.Sequential(
            nn.Linear(num_metadata_features, 16),
            nn.ReLU(),
            nn.Linear(16, 32)
        )

        self.flatten = nn.Flatten()

        # Classifier to combine image and metadata
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7 + 32, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, image, metadata):
        x_img = self.conv_layers(image)
        x_img = self.flatten(x_img)

        x_meta = self.metadata_fc(metadata)

        x = torch.cat((x_img, x_meta), 1)

        x = self.classifier(x)
        return x


# Has to be separate from class to prevent caching tons of data
def process_data(metadata, zf, transform, min_age, max_age):
    images = []
    metadata_features = []
    labels = []
    for index, row in metadata.iterrows():
        if index % 100 == 0:
            print(f"Processing image {index}/{len(metadata)}")
        image_data = zf.read("ISIC_2019_Training_Input/" + row["image"] + ".jpg")
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        if transform:
            image = transform(image)

        if math.isnan(row["age_approx"]):
            age = 0
        else:
            age = (row["age_approx"] - min_age) / (max_age - min_age)

        if row["sex"] == "male":
            sex = 0
        else:
            sex = 1

        site = [0, 0, 0, 0, 0, 0]
        site_mapping = {
            "head/neck": 0,
            "upper extremity": 1,
            "posterior torso": 2,
            "anterior torso": 3,
            "lower extremity": 4,
            "palms/soles": 5
        }
        if row["anatom_site_general"] in site_mapping:
            site[site_mapping[row["anatom_site_general"]]] = 1

        if str(row["lesion_id"]) == "nan":
            diagnosis = 0
        else:
            diagnosis = 1

        images.append(image)
        metadata_features.append([age, sex, *site])
        labels.append(diagnosis)

    return images, metadata_features, labels


class SkinLesionDataset(Dataset):
    def __init__(self, image_path, metadata_path, transform=None, num_processes=16):
        self.image_path = image_path

        self.transform = transform

        if os.path.exists("Data/dataset.pt"):
            print(f"Loading dataset from Data/dataset.pt...")
            dataset = torch.load("Data/dataset.pt", weights_only=False)

            self.images = dataset["image"]
            self.metadata_features = dataset["metadata"]
            self.labels = dataset["label"]
            print(f"Loaded {len(self.images)} data")
        else:
            self.images = []
            self.metadata_features = []
            self.labels = []

            print(f"Reading metadata from {metadata_path}...")
            metadata = pd.read_csv(metadata_path)
            self.min_age = metadata["age_approx"].min()
            self.max_age = metadata["age_approx"].max()
            print(f" Age range: {self.min_age} - {self.max_age}")

            '''pool = Pool(processes=num_processes)
            print(f"Loading images from {self.image_path}...")
            with zipfile.ZipFile(self.image_path) as zf:
                self.images, self.metadata_features, self.labels = pool.apply_async(process_data,
                                                                                    args=(metadata,
                                                                                          zf,
                                                                                          self.transform,
                                                                                          self.min_age,
                                                                                          self.max_age)).get()
            pool.close()
            pool.join()'''

            with zipfile.ZipFile(self.image_path) as zf:
                for index, row in metadata.iterrows():
                    if index % 100 == 0:
                        print(f"Processing image {index}/{len(metadata)}")
                    image_data = zf.read("ISIC_2019_Training_Input/" + row["image"] + ".jpg")
                    image = Image.open(io.BytesIO(image_data)).convert("RGB")
                    if self.transform:
                        image = self.transform(image)

                    if math.isnan(row["age_approx"]):
                        age = 0
                    else:
                        age = (row["age_approx"] - self.min_age) / (self.max_age - self.min_age)

                    if row["sex"] == "male":
                        sex = 0
                    else:
                        sex = 1

                    site = [0, 0, 0, 0, 0, 0]
                    site_mapping = {
                        "head/neck": 0,
                        "upper extremity": 1,
                        "posterior torso": 2,
                        "anterior torso": 3,
                        "lower extremity": 4,
                        "palms/soles": 5
                    }
                    if row["anatom_site_general"] in site_mapping:
                        site[site_mapping[row["anatom_site_general"]]] = 1

                    if str(row["lesion_id"]) == "nan":
                        diagnosis = 0
                    else:
                        diagnosis = 1

                    self.images.append(image)
                    self.metadata_features.append([age, sex, *site])
                    self.labels.append(diagnosis)
            print("Saving data...")
            torch.save({"image": self.images, "metadata": self.metadata_features, "label": self.labels},
                       "Data/dataset.pt",
                       _use_new_zipfile_serialization=True,
                       pickle_protocol=4
                       )
            print(f"Saved to Data/dataset.pt")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return {
            "image": self.images[index],
            "metadata": torch.tensor(self.metadata_features[index], dtype=torch.float),
            "label": torch.tensor([self.labels[index]], dtype=torch.float)
        }


class Trainer:
    def __init__(self, image_path, metadata_path):
        self.dataset = SkinLesionDataset(image_path, metadata_path, self.get_transform())
        train_size = int(0.8 * len(self.dataset))
        validate_size = len(self.dataset) - train_size
        self.train_dataset, self.validate_dataset = torch.utils.data.random_split(self.dataset, [train_size, validate_size])

    @staticmethod
    def get_transform():
        return transforms.Compose([
            transforms.Resize((224, 224)),  # Resize to input dimensions
            transforms.RandomHorizontalFlip(),  # Data augmentation
            transforms.RandomVerticalFlip(),  # Data augmentation
            transforms.RandomRotation(20),  # Data augmentation
            transforms.ColorJitter(brightness=0.1, contrast=0.1),  # For skin lesions
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet stats
        ])

    def train(self, model, criterion, optimizer, epochs):
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=32,
            shuffle=True,
            num_workers=4,  # Adjust based on CPU cores
            pin_memory=True,  # Speed up host to GPU transfers
            prefetch_factor=2  # Prefetch batches
        )
        validate_loader = DataLoader(
            self.validate_dataset,
            batch_size=32,
            shuffle=True,
            num_workers=4,  # Adjust based on CPU cores
            pin_memory=True,  # Speed up host to GPU transfers
        )
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Training on: {device}")
        model = model.to(device)

        scaler = torch.amp.GradScaler("cuda") if torch.cuda.is_available() else None

        best_acc = 0.0
        best_model = None

        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")
            model.train()
            running_loss = 0.0
            for batch in train_loader:
                images = batch["image"].to(device)
                metadata = batch["metadata"].to(device)
                labels = batch["label"].to(device)

                optimizer.zero_grad()

                if scaler is not None:
                    # Mixed numerical precision
                    with torch.amp.autocast("cuda"):
                        outputs = model(images, metadata)
                        loss = criterion(outputs, labels)

                    # Back propagation
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    # Forward propagation
                    outputs = model(images, metadata)
                    loss = criterion(outputs, labels)

                    # Back propagation
                    loss.backward()
                    optimizer.step()

                running_loss += loss.item()

            print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(train_loader):.4f}")

            # Validation phase
            acc = self.validate(model, criterion, validate_loader, device)
            if acc > best_acc:
                best_acc = acc
                best_model = model
                torch.save(model.state_dict(), "skin_lesion_cnn.pt")

        print("Training complete")
        return best_model

    def validate(self, model, criterion, loader, device):
        model.eval()

        correct = 0
        total = 0
        running_loss = 0.0
        with torch.no_grad():
            for batch in loader:
                images = batch["image"].to(device)
                metadata = batch["metadata"].to(device)
                labels = batch["label"].to(device)

                outputs = model(images, metadata)

                loss = criterion(outputs, labels)
                running_loss += loss.item() * images.size(0)

                predicted = (outputs > 0.5).float()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f"Validation Loss: {running_loss / total:.4f}, Accuracy: {correct / total:.4f} ({correct}/{total})")
        return correct / total


if __name__ == "__main__":
    trainer = Trainer("Data/ISIC_2019_Training_Input.zip", "Data/ISIC_2019_Training_Metadata.csv")
    model = SkinLesionNN(8)
    if os.path.exists("skin_lesion_cnn.pt"):
        model.load_state_dict(torch.load("skin_lesion_cnn.pt"))

    model = trainer.train(model, nn.BCEWithLogitsLoss(), torch.optim.Adam(model.parameters(), lr=0.001), 100)
