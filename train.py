import loader, model, logger
from torch.utils.data import DataLoader
from torch import optim
import pandas as pd
import torch

MODEL_DIR = './models/'
CUDA_AVAILABLE = torch.cuda.is_available()

labels = pd.read_csv('train_labels.csv').set_index('seq_id')
metadata = pd.read_csv('train_metadata.csv').set_index('seq_id')

# Indexing for the seasons I have available

season_index = metadata.file_name.str.startswith('S1/')
season_index = season_index | metadata.file_name.str.startswith('S2/')
season_index = season_index | metadata.file_name.str.startswith('S3/')
season_index = season_index | metadata.file_name.str.startswith('S4/')
season_index = season_index | metadata.file_name.str.startswith('S5/')
season_index = season_index | metadata.file_name.str.startswith('S6/')

metadata_1_6 = metadata[season_index]

# Getting recommended train/val sets

jsonfilename = 'recommended_splits.json'
with open(jsonfilename, 'r') as jsonfile:
    splits = json.load(jsonfile)

train_folders = splits['splits']['train']
val_folders = splits['splits']['val']

# splitting into train and val dataframes

def build_df_index(df, folders):
    df_index = df.file_name=='DOESNT EXIST ANYWHERE' # start with an all False index (this is janky, whatever)
    for folder in folders:
        df_index = df_index | df.file_name.str.contains(folder)
    return df_index

train_idx = build_df_index(metadata_1_6, train_folders)
train_df = metadata_1_6[train_idx]

val_idx = build_df_index(metadata_1_6, val_folders)
val_df = metadata_1_6[val_idx]

# Getting train and val data loaders

trainset = loader.SerengetiSequenceDataset(metadata_df=train_df, labels_df=labels, data_dir='..')
trainloader = DataLoader(trainset, batch_size=16, shuffle=True, num_workers=12)

clf = model.ImageSequenceClassifier(256, 50, 54)
optimizer = optim.SGD(clf.parameters(), lr=1e-4, momentum=0.9)

if CUDA_AVAILABLE:
    logger.logger.info("Using CUDA for model load.")
    clf.cuda()

for batch_samples, batch_labels in dl:
    
    if CUDA_AVAILABLE:
        batch_samples, batch_labels = batch_samples.cuda(), batch_labels.cuda()
    
    optimizer.zero_grad()
    
    loss = 0
    for ix in range(batch_samples.shape[0]):
        X, labels = batch_samples[ix], batch_labels[ix]
        predictions = clf(X)
        loss += model.TotalLogLoss(predictions, labels)
        
    logger.logger.info("Total loss: %6.2f" % loss)
    
    loss.backward()
    optimizer.step()