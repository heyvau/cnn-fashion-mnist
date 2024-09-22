from __future__ import annotations
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.utils import to_categorical
import numpy as np
import logging

logger = logging.getLogger()


class DataSet:
    def __init__(
        self, X_train: np.ndarray, y_train: np.ndarray,
              X_test: np.ndarray, y_test: np.ndarray,
              labels: tuple) -> None:

        self.X_train = X_train
        self.y_train = y_train

        self.X_test = X_test
        self.y_test = y_test

        self.labels = labels

    @classmethod
    def from_fashion_mnist(cls) -> DataSet:
        """
        Classmethod for loading and calling methods
        to preprocess FashionMNIST data set.

        Returns DataSet object included prepared
        train and test images(X) and labels(y).
        """
        logger.info("Loading train and test data from FashionMNIST")

        (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

        logger.debug(f"Images befor preprocessing: \
            \n{X_train.shape=}, {X_test.shape=} \
            \n{X_train.min()=}, {X_train.max()=}, \
            \n{X_test.min()=}, {X_test.max()=}")
        logger.debug(f"Labels befor preprocessing: \
            \n{y_train.min()=}, {y_train.max()=}")

        X_train, X_test = [
            cls._images_preprocess(X, 28, 28, 1, 255)
            for X in [X_train, X_test]
        ]
        y_train = cls._labels_preprocess(y_train, 10)

        logger.debug(f"Images after preprocessing: \
            \n{X_train.shape=}, {X_test.shape=} \
            \n{X_train.min()=}, {X_train.max()=}, \
            \n{X_test.min()=}, {X_test.max()=}")
        logger.debug(f"Labels after preprocessing: \
            \n{y_train.min()=}, {y_train.max()=}")

        return cls(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            labels=(
                "T-shirt/top", "Trouser", "Pullover",
                "Dress", "Coat", "Sandal", "Shirt",
                "Sneaker", "Bag", "Ankle boot"
            )
        )

    @classmethod
    def _images_preprocess(
        cls, 
        images: np.ndarray,
        img_height: int,
        img_weight: int,
        n_channels: int,
        px_value: int) -> np.ndarray:
        """
        Method reshapes data images to include channsels,
        then convert pixel values to float and scales them between 0-1.
        """
        logger.info("Images preprocessing")

        return images.reshape(
            -1, img_height, img_weight, n_channels
        ).astype('float32') / px_value

    @classmethod
    def _labels_preprocess(
        cls,
        labels: np.ndarray,
        n_classes: int) -> np.ndarray:
        """
        Method converts data labels to one-hot encoded vector.
        """
        logger.info("Labels preprocessing")

        return to_categorical(labels, n_classes)
