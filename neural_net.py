"""
Neural Net logic.
"""

# System Imports.
import numpy, pandas

# User Class Imports.
from resources import logging


# Initialize logging.
logger = logging.get_logger(__name__)


class Normalizer():
    """
    Handles data normalization.
    """
    def normalize_data(self, data):
        """
        Normalizes and returns provided dataset.
        :param data: A pandas array of housing data to normalize.
        :return: The normalized array of housing data.
        """
        # Print out data.
        # logger.info(data)
        # logger.info(data.columns.values)

        # Address individual columns.

        return data

class BackPropNet():
    """
    Neural Net implementing back propagation.
    """
    def __init__(self, data):
        self.weights = self._initialize_weights(data)

    def _initialize_weights(self, data):
        """
        Initialize weights based of number of passed columns in data.
        Values are initialized to random decimals near 0, using a normal distribution.
        :param data: Data to create weights for.
        :return: Vector of column weights.
        """
        weights = []
        for column in data.columns:
            weights.append(numpy.random.randn() * 0.001)

        # logger.info(weights)
        return weights

    def predict(self):
        pass

    def train(self):
        pass

    def _train_step(self):
        pass

    def _calculate_error(self):
        pass


class ResultTracker():
    """
    Tracks statistics and progress of main Neural Net class.
    """
