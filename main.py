"""Pipeline using ConvAE and LSTM."""
from torch.utils.data import DataLoader
from TimeSeriesDL.model import BaseModel
from TimeSeriesDL.data import AutoEncoderCollate
from TimeSeriesDL.model import ConvAE, LSTM
from TimeSeriesDL.utils import config

from data import AudioDataset

def train(path: str, conv_ae: ConvAE = None) -> BaseModel:
    """Trains based on a given config.

    Args:
        path (str): The path to a train config.
        conv_ae (ConvAE): A trained ConvAE to use with a dataset wrapper.

    Returns:
        BaseModel: The trained model.
    """
    # load training arguments (equals example/simple_model.py)
    train_args = config.get_args(path)

    # create a dataset loader which loads a matplotlib matrix from ./train.mat
    data = AudioDataset(**train_args["dataset"])
    dataloader = DataLoader(data, **train_args["dataloader"])
    if conv_ae:
        aew = AutoEncoderCollate(conv_ae, device="cuda")
        dataloader = DataLoader(data, collate_fn=aew.collate_fn(), **train_args["dataloader"])

    # create a model based on what is defined in the config
    # to do so, a model needs to be registered using config.register_model()
    model_name = train_args["model_name"]
    model: BaseModel = config.get_model(model_name)(**train_args["model"])
    model.use_device(train_args["device"])

    # train the model on the dataset for 5 epochs and log the progress in a CLI
    # to review the model's training performance, open TensorBoard in a browser
    model.learn(train=dataloader, epochs=train_args["train_epochs"], verbose=True)

    # save the model to its default location 'runs/{time_stamp}/model_SimpleModel.torch'
    model.save_to_default()

    # also, store a modified copy of the training arguments containing the model path
    # this makes comparisons between multiple experiments easier<
    train_args["model_path"] = model.log_path + "/model.torch"
    config.store_args(f"{model.log_path}/config.yml", train_args)
    return model

if __name__ == "__main__":
    # train the auto encoder and encode the dataset
    print("Train the ConvAE")
    ae: ConvAE = train("./config/ae.yml")

    # train the lstm on the encoded dataset, then decode: ae.decode(lstm.predict(x))
    print("\nTrain LSTM")
    lstm: LSTM = train("./config/lstm.yml", ae)
