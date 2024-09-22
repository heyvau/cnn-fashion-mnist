from dataset import DataSet
from model import ModelCNN
import os
import json
from pathlib import Path
import logging, logging.config

os.chdir(Path(__file__).parent)

logging.config.fileConfig("logging.ini")
logger = logging.getLogger()

CONFIG_FILE = "config.json"

def app(config_file: str):
    fashion_mnist = DataSet.from_fashion_mnist()

    with open(config_file, mode="r", encoding="utf-8") as f:
        config = json.load(f)

    model = ModelCNN.create_from_config(config=config)

    history = model.train(
        train_images=fashion_mnist.X_train,
        train_labels=fashion_mnist.y_train,
        batch_size=32, n_epochs=30
    )
    model.plot_accuracy(history)


if __name__ == "__main__":
    app(config_file=CONFIG_FILE)

