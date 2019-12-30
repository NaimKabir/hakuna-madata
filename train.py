from hakuna_madata import loader, model, logger
from torch.utils.data import DataLoader
from torch import optim
import pandas as pd
import torch

labels = pd.read_csv('train_labels.csv').set_index('seq_id')
metadata = pd.read_csv('train_metadata.csv').set_index('seq_id')
metadata_s1 = metadata[metadata.file_name.str.startswith('S1/')]
sample = metadata_s1.sample(1000)

dataset = loader.SerengetiSequenceDataset(metadata_df=sample, labels_df=labels, data_dir='.')


dl = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=4)




CUDA_AVAILABLE = torch.cuda.is_available()

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
        %time predictions = clf(X)
        %time loss += model.TotalLogLoss(predictions, labels)
        
    logger.logger.info("Total loss: %6.2f" % loss)
    
    %time loss.backward()
    %time optimizer.step()
    break