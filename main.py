import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import logger
from loader import SerengetiSequenceDataset, RESIZE_TARGET

CUDA_AVAILABLE = torch.cuda.is_available()
ASSET_PATH = Path(__file__).parents[0] / "assets"
MODEL_PATH = ASSET_PATH / "test_submission.pt"

# The images will live in a folder called 'data' in the container
DATA_PATH = Path(__file__).parents[0] / "data"


def predict(clf, valset):
    """ Evaluate on a subset of the test data. 
        Binarize: anything below 5% is dropped to 0 and everything else is jumped to 1. """

    clf  # go into eval mode so we don't accrue grads
    valloader = DataLoader(valset, batch_size=32, shuffle=False)

    all_preds = []
    for batch_samples in valloader:

        batch_preds = []

        if CUDA_AVAILABLE:
            batch_samples = batch_samples.cuda()

        for ix in range(batch_samples.shape[0]):
            X = batch_samples[ix]
            predictions = clf(X).unsqueeze(0)
            batch_preds.append(predictions)

        batch_preds_tensor = torch.cat(batch_preds, 0)

        all_preds.append(batch_preds_tensor)

    return torch.cat(all_preds, 0)


def perform_inference():
    """This is the main function executed at runtime in the cloud environment. """

    logger.logger.info("Loading model.")
    clf = torch.load(MODEL_PATH).eval()
    if CUDA_AVAILABLE:
        clf.cuda()

    logger.logger.info("Loading and processing metadata.")

    test_metadata = pd.read_csv(DATA_PATH / "test_metadata.csv", index_col="seq_id")

    logger.logger.info("Starting inference.")

    # Preallocate prediction output
    submission_format = pd.read_csv(DATA_PATH / "submission_format.csv", index_col=0)

    # Instantiate test data generator
    dataset = SerengetiSequenceDataset(
        metadata_df=test_metadata,
        data_dirs=[DATA_PATH],
        training_mode=False,
        input_resize=RESIZE_TARGET,
        sequence_max=50,
    )

    # Perform (and time) inference
    inference_start = datetime.now()
    with torch.no_grad():
        preds = predict(clf, dataset)
    inference_stop = datetime.now()
    logger.logger.info(f"Inference complete. Took {inference_stop - inference_start}.")

    logger.logger.info("Creating submission.")

    my_submission = pd.DataFrame(
        preds.cpu().numpy(),
        # Remember that we are predicting at the sequence, not image level
        index=test_metadata.index.unique(),
        columns=submission_format.columns,
    )

    # We want to ensure all of our data are floats, not integers
    my_submission = my_submission.astype(np.float)

    # Save out submission to root of directory
    my_submission.to_csv("submission.csv", index=True)
    logger.logger.info(f"Submission saved.")


if __name__ == "__main__":
    perform_inference()
