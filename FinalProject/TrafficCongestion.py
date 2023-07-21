!unzip train

!pip install barbar

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load
from IPython.display import Image, Video, HTML

import time
import copy
import pickle
from barbar import Bar

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import scipy
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from PIL import Image
import cv2
%matplotlib inline

import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision import transforms

from tqdm import tqdm
from pathlib import Path
import gc
RANDOMSTATE = 0

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All"
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# Find if any accelerator is presented, if yes switch device to use CUDA or else use CPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

datasetPath = Path('train')
congested = datasetPath/'dense_traffic'
sparse = datasetPath/'sparse_traffic'
df = pd.DataFrame()

for f in os.listdir(congested):
    if os.path.isfile(os.path.join(congested, f)):
        df = df.append({
                'image': str(os.path.join(congested,f)),
                'congestion': 1
            }, ignore_index=True)

for f in os.listdir(sparse):
    if os.path.isfile(os.path.join(sparse, f)):
        df = df.append({
                'image': str(os.path.join(sparse,f)),
                'congestion': 0
            }, ignore_index=True)

df = df.sample(frac=1).reset_index(drop=True)
df.head()

# Torch helper class to streamline data to DataLoaders
# Includes:
# - Transformation routines
# - Other functions used by dataloaders
class Dataset(Dataset):

    def __init__(self, dataFrame):
        self.dataFrame = dataFrame

        self.transformations = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __getitem__(self, key):
        if isinstance(key, slice):
            raise NotImplementedError('slicing is not supported')

        row = self.dataFrame.iloc[key]
        image = self.transformations(Image.open(row['image']))
        label = torch.tensor([row['congestion']], dtype=float)
        return image,label

    def __len__(self):
        return len(self.dataFrame.index)

!pip install efficientnet_pytorch

from efficientnet_pytorch import EfficientNet

class TransferLearningModel(nn.Module):
    @staticmethod
    def model(version):
        basemodel = EfficientNet.from_pretrained(version)

        # Freezing model weights
        for param in basemodel.parameters():
            param.requires_grad = False
        num_ftrs = basemodel._fc.in_features

        basemodel._fc = nn.Linear(num_ftrs, 1)

        return basemodel

# Intermediate Function to process data from the data retrival class
def prepare_data(DF):
    trainDF, validateDF = train_test_split(DF, test_size=0.15, random_state=RANDOMSTATE)
    train_set = Dataset(trainDF)
    validate_set = Dataset(validateDF)

    return train_set, validate_set

def load_ckpt(checkpoint_fpath, model, optimizer):
    """
    checkpoint_path: path to save checkpoint
    model: model that we want to load checkpoint parameters into
    optimizer: optimizer we defined in previous training
    """
    # load check point
    checkpoint = torch.load(checkpoint_fpath)

    # initialize state_dict from checkpoint to model
    model.load_state_dict(checkpoint['model_state_dict'])

    # initialize optimizer from checkpoint to optimizer
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # return model, optimizer, epoch value, min validation loss
    return model, optimizer, checkpoint['epoch']

def save_checkpoint(state, filename):
    print ("=> Saving a new best")
    torch.save(state, filename)  # save checkpoint

def train_model(model,
                start_epoch,
                criterion,
                optimizer,
                num_epochs):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(start_epoch, start_epoch+num_epochs+1):
        print('Epoch {}/{}'.format(epoch, start_epoch+num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for idx,(inputs, labels) in enumerate(Bar(dataloaders[phase])):
                inputs = inputs.to(device)
                labels = torch.as_tensor(labels, dtype = torch.float)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    m = nn.Sigmoid()

                    preds = torch.sigmoid(outputs)
                    preds = preds>0.5

                    loss = criterion(m(outputs).to(device), labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                save_checkpoint(state={
                                    'epoch': epoch,
                                    'state_dict': model.state_dict(),
                                    'best_accuracy': best_acc,
                                    'optimizer_state_dict':optimizer.state_dict()
                                },filename='ckpt_epoch_{}.pt'.format(epoch))

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, optimizer, epoch_loss

#EPOCHS = 50
EPOCHS = 5
start_epoch = 0
RETRAIN=False

trainDF, validateDF = prepare_data(df)

# declare the dataloader from the dataframe
dataloaders = {'train': DataLoader(trainDF, batch_size=32, shuffle=True, num_workers=1) ,
                'val':DataLoader(validateDF, batch_size=32, num_workers=1)
                }

dataset_sizes = {'train': len(trainDF),'val':len(validateDF)}

#initialize the model and put it acceleration(if available)
model = TransferLearningModel.model(version='efficientnet-b2')
model = model.to(device)

criterion = nn.BCELoss()
# Observe that all parameters are being optimized
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

if RETRAIN==True:
    # load the saved checkpoint
    model, optimizer, start_epoch = load_ckpt(args['model'], model, optimizer)
    print('Checkpoint Loaded...')

# calling the training function
model, optimizer, loss = train_model(model=model,
                    start_epoch=start_epoch,
                    criterion=criterion,
                    optimizer=optimizer,
                    num_epochs=EPOCHS)

# saving the model
torch.save({
            'epoch': EPOCHS,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, 'efficientnet_v1.pt')

# Loading the model from the memory
model = TransferLearningModel.model(version='efficientnet-b2')
model.load_state_dict(torch.load('efficientnet_v1.pt', map_location=device)['model_state_dict'], strict=False)

model = model.to(device)
model.eval()

# declaring transformations
transformations = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

from IPython.display import HTML

HTML("""
    <video width="480" height="270" controls>
        <source src="https://im4.ezgif.com/tmp/ezgif-4-67d65b9c93.mp4" type="video/mp4">
    </video>
""")

font = cv2.FONT_HERSHEY_SIMPLEX
labels = ['Sparse', 'Congested']

# Green = Sparse
# Red = Congested
labelColor = [(0, 255, 0),(0, 0, 255)]

vs = cv2.VideoCapture('train/video001.mp4')

#videowriter arguments
codec = "MJPG"
fps = 20
capture_size = (int(vs.get(3)), int(vs.get(4)))
fourcc = cv2.VideoWriter_fourcc(*codec)
writer = cv2.VideoWriter('output.avi', fourcc, fps,capture_size)

num_frames = int(vs.get(cv2.CAP_PROP_FRAME_COUNT))

# To make it more efficient and fast, frame-skipping is applied, which means we'll make a prediction in alternate frames.
process_this_frame = True
for i in tqdm(range(num_frames)):

    _ , frame = vs.read()

    if process_this_frame:
        outputs = model(transformations(frame).unsqueeze(0).to(device))
        predicted = torch.sigmoid(outputs)
        predicted = predicted>0.5

    process_this_frame = not process_this_frame

    # center text according to the face frame
    textSize = cv2.getTextSize(labels[predicted], font, 1, 2)[0]

    # draw prediction label
    cv2.putText(frame,
                'Traffic Status:{}'.format(labels[predicted]),
                (0, 80),
                font, 1, labelColor[predicted], 2)


    if writer is not None:
        writer.write(frame)


vs.release()

%%HTML
<video width="320" height="240" controls>
  <source src="output.avi" type="video/avi">
</video>
