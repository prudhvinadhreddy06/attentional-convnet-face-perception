import numpy as np
import pandas as pd
from glob import glob
from os.path import join
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torchvision
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import torch.optim as optim

class AgeDataset(torch.utils.data.Dataset):

    def __init__(self, data_path, annot_path, train=True):
        super(AgeDataset, self).__init__()

        self.annot_path = annot_path
        self.data_path = data_path
        self.train = train

        self.ann = pd.read_csv(annot_path)
        self.files = self.ann['file_id']
        if train:
            self.ages = self.ann['age']
        self.transform = self._transform(224)

    @staticmethod
    def _convert_image_to_rgb(image):
        return image.convert("RGB")

    def _transform(self, n_px):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        return Compose([
            Resize(n_px),
            self._convert_image_to_rgb,
            ToTensor(),
            Normalize(mean, std),
        ])

    def read_img(self, file_name):
        im_path = join(self.data_path,file_name)
        img = Image.open(im_path)
        img = self.transform(img)
        return img

    def __getitem__(self, index):
        file_name = self.files[index]
        img = self.read_img(file_name)
        if self.train:
            age = self.ages[index]
            return img, age
        else:
            return img

    def __len__(self):
        return len(self.files)


train_path = 'content/faces_dataset/train'
train_ann = 'content/faces_dataset/train.csv'
train_dataset = AgeDataset(train_path, train_ann, train=True)


test_path = 'content/faces_dataset/test'
test_ann = 'content/faces_dataset/submission.csv'
test_dataset = AgeDataset(test_path, test_ann, train=False)



train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

print(train_dataset[0][0].shape)

## attention cnn
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class AttentionalConvNet(nn.Module):
#     def __init__(self, num_age_groups):
#         super(AttentionalConvNet, self).__init__()
#         # Define your convolutional layers
#         self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=4, padding=1,stride=2)
#         self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, padding=1,stride=2)
#         # Add more convolutional layers as needed

#         # Attention mechanism
#         self.attention = nn.Sequential(
#             nn.Conv2d(in_channels=128, out_channels=1, kernel_size=3,padding=1),
#             nn.Sigmoid()
#         )

#         # Fully connected layers
#         self.fc1 = nn.Linear(128*56*56, 512)  # Adjust based on your output from convolutional layers
#         self.fc2 = nn.Linear(512, num_age_groups)

#     def forward(self, x):
#         # Apply convolutional layers
#         #print(x.shape)
#         x = F.relu(self.conv1(x))
#         #print(x.shape)
#         x = F.relu(self.conv2(x))

#         # Compute attention weights
#         attention_weights = self.attention(x)
#         #print(x.shape,attention_weights.shape)
#         x = x * attention_weights
#         #print(x.shape)

#         # Flatten and pass through fully connected layers
#         #x = x.view(x.size(0), -1)  # Flatten the feature maps
#         x = x.view(x.size(0), 128*56*56)
#         x = F.relu(self.fc1(x))
#         age_logits = self.fc2(x)

#         return age_logits

# #resnet_classifcation
# model = torchvision.models.resnet18(pretrained=True)
# #print(model)
# num_features = model.fc.in_features  # Get number of input features from the previous layer
# model.fc = nn.Linear(num_features, 150)  # Replace with number of age groups

import torch
import torch.nn as nn
from torchvision import models
#
class AgePredictionModel(nn.Module):
    def __init__(self):
        super(AgePredictionModel, self).__init__()
        # Load a pre-trained ResNet-50 model
        self.base_model = models.resnet18(pretrained=True)
        # Replace the final fully connected layer for age regression
        num_ftrs = self.base_model.fc.in_features
        self.base_model.fc = nn.Linear(num_ftrs, 1) # 1 output for age

    def forward(self, x):
        x = self.base_model(x)
        return x

# Sample usage
model = AgePredictionModel()
# Assuming 'image' is a preprocessed face image tensor
#predicted_age = model(image)

# Instantiate the model
num_age_groups = 150  # Adjust based on your dataset
#model = AttentionalConvNet(num_age_groups)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model.to(device)

# Define loss function and optimizer
#criterion = nn.CrossEntropyLoss()
criterion = nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
num_epochs = 15

# Training loop (you'll need to load your dataset and preprocess images)
for epoch in range(num_epochs):
    avg_loss =0
    for batch_images, batch_labels in train_loader:
        batch_images, batch_labels = batch_images.to(device), batch_labels.to(device)
        optimizer.zero_grad()
        age_logits = model(batch_images)
        age_logits = age_logits.squeeze(1)
        #print(age_logits.shape,batch_labels.shape,batch_labels,age_logits)
        loss = criterion(age_logits, batch_labels)
        loss.backward()
        avg_loss += loss.item()
        optimizer.step()
    print(avg_loss/len(train_loader))

model.eval()
with torch.no_grad():
    for batch_images in test_loader:
        batch_images = batch_images.to(device)
        age_logits = model(batch_images)
        predicted_ages = torch.argmax(age_logits, dim=1)

print(predicted_ages)  # These are the predicted age group indices

###### SUBMISSION CSV FILE #####

@torch.no_grad
def predict(loader, model):
    model.eval()
    predictions = []

    for img in tqdm(loader):
        img = img.to(device)

        pred = model(img)
        #print(pred.shape)
        #pred = torch.argmax(pred, dim=1)
        #print(pred.shape)
        predictions.extend(pred.flatten().detach().tolist())

    return predictions
preds = predict(test_loader, model)
print(len(preds))
submit = pd.read_csv('content/faces_dataset/submission.csv')
submit['age'] = preds
submit.head()

submit.to_csv('baseline.csv',index=False)