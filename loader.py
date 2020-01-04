import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import torch.nn.functional as F
import torch
from torchvision import transforms
from collections.abc import Iterable
from PIL import Image
from PIL import ImageFile
import os
from typing import Optional, List
import functools

ImageFile.LOAD_TRUNCATED_IMAGES = True  # I have rare truncated data that I'm just... allowing. Don't judge me
MAX_CACHE_SIZE = int(os.environ.get("MAX_CACHE_SIZE", 2048))
RESIZE_TARGET = (224, 300)  # anything 3:4 ratio should keep original aspect (1536, 2048) and not distort too much
MEANS_NORMALIZE = [0.485, 0.456, 0.406]
STDS_NORMALIZE = std = [0.229, 0.224, 0.225]

# deal with image rescaling
# deal with empty images
# augment data with random axis flips
# use separate post-embedding models to do class-by-class classification (to avoid softmaxing)
# no rescaling necessary--ResNets will adapt. Maybe training should involve random resizing however.
# call eval() on a model after loading it in order to make it run in execution mode.

# All pre-trained models expect input images normalized in the same way, i.e. mini-batches of
# 3-channel RGB images of shape (3 x H x W), where H and W are expected to be at least 224.
# The images have to be loaded in to a range of [0, 1] and then normalized using
# mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225].

# finetuning https://towardsdatascience.com/transfer-learning-picking-the-right-pre-trained-model-for-your-problem-bac69b488d16


class SerengetiSequenceDataset(Dataset):
    """
        This Dataset models the basic classifiable units of the LILA Serengeti Snapshots
        data set: image sequences.
        
        Each sequence has its images perturbed (in the same way for all images in the sequence) in some
        way upon every retrieval, to augment the available data.
        
        Images are cached: so be smart with the Dataframe fed to this dataset.
    """

    def __init__(
        self,
        metadata_df: pd.DataFrame,
        data_dirs: List[str] = ["."],
        labels_df: Optional[pd.DataFrame] = None,
        training_mode=False,
        input_resize=RESIZE_TARGET,
    ):
        """
        The data set interfaces with a set of files stored on disk in the same filenaming scheme that exists
        in the original LILA .zip archives.
        
        The the metadata_df can be molded according to user preferences: oversampling, undersampling, etc.
        of sequences in a training set is up to user discretion.
        
        Args:
            metadata_df: Path to the metadata file that maps image filenames to sequence ids.
            data_dir: Directory with all the data--used to prefix image retrieval.
            labels_df: A dataframe of labels is used if you're training a model.
                       This will serve up true/false values for each class.
            sequence_max: The maximum number of pictures we expect to see in a sequence. Used for padding.
                          If there are more images in a sequence than this number, the sequence will be truncated.
                          In seasons 1-6, the maximum number of images per sequence was 39.
        """

        # attrs from args

        assert metadata_df.index.name == "seq_id", "metadata_df must be indexed by seq_id"

        self.metadata = metadata_df
        self.data_dirs = data_dirs
        self.input_resize = input_resize
        self.training_mode = training_mode

        if labels_df is not None:
            assert labels_df.index.name == "seq_id", "labels_df must be indexed by seq_id"

        self.labels = labels_df

        # derived attrs

        self.seq_ids = np.array(self.metadata.index.unique())  # nparray

        # torchvision preprocessing

        transform_operations = []

        resize = transforms.Resize(self.input_resize)
        transform_operations.append(resize)

        if self.training_mode:
            # training data augmentation

            jitter = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.1)
            transform_operations.append(jitter)

        tensorize = transforms.ToTensor()
        transform_operations.append(tensorize)

        normalize = transforms.Normalize(MEANS_NORMALIZE, STDS_NORMALIZE, inplace=True)
        transform_operations.append(normalize)

        self.preprocess = transforms.Compose(transform_operations)

    def __len__(self):
        """The base unit is a sequence of images."""
        return len(self.seq_ids)

    def __getitem__(self, idx: int):
        """ 
            Get an item and tensorize it, but also make sure to resize to a constant size. 
            We also apply random set of transformations for data augmentation--useful if we oversample
            particular rare sequences.
            
            Finally, images are mean-subtracted for use with models pretrained on ImageNet.
        """

        # get sequence id

        seq_id = self.seq_ids[idx]

        # get all images associated with sequence
        # for my purposes, order doesn't actually matter so I don't enforce an order
        # (this would change if we ended up using sequence encoders of any sort)

        img_files = self.metadata.loc[seq_id]
        if isinstance(img_files, pd.Series):
            img_files = [img_files.file_name]
        elif isinstance(img_files, pd.DataFrame):
            img_files = img_files.file_name.values

        assert isinstance(img_files, Iterable), "Something went wrong with loading img_files, should be iterable."

        imgs = (self.load_img(file_name) for file_name in img_files)

        # processing images

        imgs = [self.preprocess(img) for img in imgs]

        # get sequence tensor

        sequence = self.prepare_sequence(imgs)

        # provision label tensor if available

        if self.labels is not None:
            labels = self.labels.loc[seq_id].values
            labels = torch.from_numpy(labels)

            return sequence, labels

        return sequence

    def prepare_sequence(self, imgs):
        """Prepare series of images: create one tensor to represent the whole sequence."""

        # we either truncate to the max expected sequence length or pad to it

        imgs = [img.unsqueeze(0) for img in imgs]  # # (1x3xWxH) tensors
        sequence = torch.cat(imgs, 0)
        return sequence

    @functools.lru_cache(MAX_CACHE_SIZE)
    def load_img(self, file_name):
        """
            Load an image from the filesystem. It's assumed that a file_name is a good absolute path,
            so we must make our filesystem conform to the naming scheme in the LILA zips.
            
            Image loads are cached in memory for a time.
        """
        possible_prefixes = self.data_dirs[:]
        while len(possible_prefixes) > 0:
            try:
                prefix = possible_prefixes.pop(0)
                img = Image.open(f"{prefix}/{file_name}")
                return img
            except FileNotFoundError as e:
                if len(possible_prefixes) == 0:
                    raise e
