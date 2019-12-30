import loader, model, logger
from torch.utils.data import DataLoader
import json
from torch import optim
import pandas as pd
import torch

MODEL_DIR = "../models/"
CUDA_AVAILABLE = torch.cuda.is_available()
MAX_SAMPLES_PER_LABEL = 22554  # discovered empirically from S1_6 TRAINING data count of Hartebeest (not val)
CHECKPOINT_EVERY_N_BATCHES = 5000  # save model out every N batches
BATCH_SIZE = 8


labels = pd.read_csv("../train_labels.csv").set_index("seq_id")
metadata = pd.read_csv("../train_metadata.csv").set_index("seq_id")

logger.logger.info("Loaded data.")

# Indexing for the seasons I have available

season_index = metadata.file_name.str.startswith("S1/")
season_index = season_index | metadata.file_name.str.startswith("S2/")
season_index = season_index | metadata.file_name.str.startswith("S3/")
season_index = season_index | metadata.file_name.str.startswith("S4/")
season_index = season_index | metadata.file_name.str.startswith("S5/")
season_index = season_index | metadata.file_name.str.startswith("S6/")

metadata_1_6 = metadata[season_index]

# Getting recommended train/val sets

jsonfilename = "recommended_splits.json"
with open(jsonfilename, "r") as jsonfile:
    splits = json.load(jsonfile)

train_folders = splits["splits"]["train"]
val_folders = splits["splits"]["val"]

# splitting into train and val dataframes


def build_df_index(df, folders):
    df_index = df.file_name == "DOESNT EXIST ANYWHERE"  # start with an all False index (this is janky, whatever idc)
    for folder in folders:
        df_index = df_index | df.file_name.str.contains(folder)
    return df_index


train_idx = build_df_index(metadata_1_6, train_folders)
train_df = metadata_1_6[train_idx]

val_idx = build_df_index(metadata_1_6, val_folders)
val_df = metadata_1_6[val_idx]

logger.logger.info("Got training splits.")

# balancing across labels in training--oversampling and undersampling as necessary
# leaving test set unbalanced, as it likely will be in practice

decoded_labels = labels.idxmax(axis=1).rename("decoded")  # labels given as 'zebra' 'lion' or 'empty
balanced_train_df = (
    train_df.join(decoded_labels)
    .groupby("decoded", as_index=False)
    .apply(lambda df: df.sample(MAX_SAMPLES_PER_LABEL, replace=True))
    .droplevel(0)
)

logger.logger.info("Balanced data set.")

# Getting train and val data loaders

# I have a janky as hell solution to allow data loading from multiple disks
# this isn't neat because Docker doesn't allow symlinking and my data is in many volumes
possible_data_dirs = ["..", "../disks/s2/", "../disks/s3/", "../disks/s4/", "../disks/s5/", "../disks/s6/"]

trainset = loader.SerengetiSequenceDataset(
    metadata_df=balanced_train_df, labels_df=labels, data_dirs=possible_data_dirs
)
trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=BATCH_SIZE)

valset = loader.SerengetiSequenceDataset(metadata_df=val_df, labels_df=labels, data_dirs=possible_data_dirs)

# loading model

clf = model.ImageSequenceClassifier(256, 50, 54)
optimizer = optim.SGD(clf.parameters(), lr=1e-4, momentum=0.9)

def evaluate(model, valset, max_N):
    """ Evaluate on a subset of the test data """
    valloader = DataLoader(valset, batch_size=BATCH_SIZE, shuffle=True, num_workers=BATCH_SIZE)

    #TODO

if CUDA_AVAILABLE:
    logger.logger.info("Using CUDA for model load.")
    clf.cuda()

EPOCHS = 1000

for epoch in range(EPOCHS):

    logger.logger.info("EPOCH: %d" % epoch)

    for N, (batch_samples, batch_labels) in enumerate(trainloader):

        if CUDA_AVAILABLE:
            batch_samples, batch_labels = batch_samples.cuda(), batch_labels.cuda()

        optimizer.zero_grad()

        loss = 0
        for ix in range(batch_samples.shape[0]):
            X, labels = batch_samples[ix], batch_labels[ix]
            predictions = clf(X)
            loss += model.TotalLogLoss(predictions, labels)

        mean_loss = "%6.2f" % (loss / BATCH_SIZE)
        logger.logger.info("Mean loss: %s" % mean_loss)

        loss.backward()
        optimizer.step()

        if N % CHECKPOINT_EVERY_N_BATCHES == 0:
            torch.save(clf, f"{MODEL_DIR}/resnet_18_loss_{mean_loss}_iter_{str(N)}.pt")
