from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import numpy as np
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger()

class ModelCNN:
    def __init__(self, model: Sequential) -> None:
        self.model = model

    @classmethod
    def create_from_config(cls, config: dict):
        """
        Classmethod to build and compile Sequential model
        based on config file.
        """
        model = cls._create(config)
        return cls(
            model=model
        )

    @classmethod
    def _create(cls, config: dict) -> Sequential:
        input_shape = (
            config["img_height"], config["img_weight"], config["n_channels"])

        logger.debug(f"{input_shape=}")
        logger.info("Model building")

        model = Sequential([
            Conv2D(
                filters=config["conv_l1"]["n_filters"],
                kernel_size=(config["kernel_size"], config["kernel_size"]),
                input_shape=input_shape,
                activation=config["conv_l1"]["af"]),
            MaxPooling2D(
                pool_size=(config["pool_size"], config["pool_size"])),
            Conv2D(
                filters=config["conv_l2"]["n_filters"],
                kernel_size=(config["kernel_size"], config["kernel_size"]),
                activation=config["conv_l2"]["af"]),
            MaxPooling2D(
                pool_size=(config["pool_size"], config["pool_size"])),
            Dropout(config["dropout_l1"]["fraction"]),
            Flatten(),
            Dense(
                config["dense_l1"]["dim"],
                activation=config["dense_l1"]["af"]),
            Dropout(config["dropout_l2"]["fraction"]),
            Dense(
                config["dense_l2"]["dim"],
                activation=config["dense_l2"]["af"])
        ])

        logger.info("Model compilation")

        model.compile(
            optimizer=config["optimizer"],
            loss=config["loss"],
            metrics=config["metrics"]
        )
        return model        

    def train(
        self, train_images: np.ndarray, train_labels: np.ndarray,
        batch_size: int, n_epochs: int) -> dict:
        """
        Method fits the model with train data and returns training history.
        """
        history = self.model.fit(
            train_images, train_labels,
            epochs=n_epochs, batch_size=batch_size, validation_split=0.1
        ).history

        logger.debug(f"{history.items()=}")

        return history

    def plot_accuracy(self, history: dict) -> None:
        """
        Method for plotting training and validation accuracy.
        """
        plt.figure(figsize=(10, 5))
        plt.plot(history["accuracy"], label="Training accuracy")
        plt.plot(history["val_accuracy"], label="Validation accuracy")
        plt.title("Training vs validation accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.grid()
        plt.show()

    def predict(self, test_images: np.ndarray) -> list:
        """
        Method returs list of predicted lables for test images.
        """
        predictions = self.model.predict(test_images)
        return np.argmax(predictions, axis=1)
