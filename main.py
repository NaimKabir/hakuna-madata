import torch
from torch.utils.data import DataLoader
import logger
from loader import SerengetiSequenceDataset, RESIZE_TARGET

ASSET_PATH = Path(__file__).parents[0] / "assets"
MODEL_PATH = ASSET_PATH / "test_submission.pt"

# The images will live in a folder called 'data' in the container
DATA_PATH = Path(__file__).parents[0] / "data"

def predict(clf, valset):
    """ Evaluate on a subset of the test data """

    clf.eval() # go into eval mode so we don't accrue grads
    valloader = DataLoader(valset, batch_size=16, shuffle=False)

    all_preds = []
    for N, (batch_samples, batch_labels) in enumerate(valloader):

        batch_preds = []

        if CUDA_AVAILABLE:
            batch_samples, batch_labels = batch_samples.cuda(), batch_labels.cuda()

        for ix in range(batch_samples.shape[0]):
            X, labels = batch_samples[ix], batch_labels[ix]
            predictions = clf(X)
            batch_preds.append(predictions)
        
        batch_preds_tensor = torch.cat(batch_preds, 0)

        all_preds.append(batch_preds_tensor)

    return torch.cat(all_preds, 0)

def perform_inference():
    """This is the main function executed at runtime in the cloud environment. """

    CUDA_AVAILABLE = torch.cuda.is_available()

    logging.info("Loading model.")
    clf = torch.load(MODEL_PATH)
    if CUDA_AVAILABLE:
        clf.cuda()

    logging.info("Loading and processing metadata.")

    # Our preprocessing selects the first image for each sequence
    test_metadata = pd.read_csv(DATA_PATH / "test_metadata.csv", index_col="seq_id")

    logger.logger.info("Starting inference.")

    # Preallocate prediction output
    submission_format = pd.read_csv(DATA_PATH / "submission_format.csv", index_col=0)
    num_labels = submission_format.shape[1]
    output = np.zeros((test_metadata.shape[0], num_labels))

    # Instantiate test data generator
    dataset = SerengetiSequenceDataset(
        metadata_df=test_metadata,
        data_dirs = [DATA_PATH],
        training_mode=False,
        input_resize=RESIZE_TARGET,
        sequence_max= 50,
    )

    # Perform (and time) inference
    inference_start = datetime.now()
    preds = predict(clf, dataset)
    inference_stop = datetime.now()
    logging.info(f"Inference complete. Took {inference_stop - inference_start}.")

    logging.info("Creating submission.")

    # Check our predictions are in the same order as the submission format
    assert np.all(
        test_metadata.seq_id.unique().tolist() == submission_format.index.to_list()
    )

    output[: preds.shape[0], :] = preds[: output.shape[0], :]
    my_submission = pd.DataFrame(
        np.stack(output),
        # Remember that we are predicting at the sequence, not image level
        index=test_metadata.seq_id,
        columns=submission_format.columns,
    )

    # We want to ensure all of our data are floats, not integers
    my_submission = my_submission.astype(np.float)

    # Save out submission to root of directory
    my_submission.to_csv("submission.csv", index=True)
    logging.info(f"Submission saved.")


if __name__ == "__main__":
    perform_inference()
