import os
import cv2
import numpy as np
from torch.utils.data import Dataset


def load_images_and_labels(image_folder, label):
    images = []
    labels = []
    for image_file in os.listdir(image_folder):
        image_path = os.path.join(image_folder, image_file)
        image = cv2.imread(image_path)
        images.append(image)
        labels.append(label)
    return np.array(images), np.array(labels)

def crop_faces(image, detections):
    faces = []
    image_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    for box in detections:
        x1, y1, x2, y2 = map(int,box[0])
        face = image_np[y1:y2, x1:x2]  # Crop the face
        faces.append(face)
    return faces

class CustomDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)  # Apply transformations if any

        return image, label