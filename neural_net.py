"""
Neural Net logic.
"""

# System Imports.
import math, numpy, pandas

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
        normalized_data = pandas.DataFrame()

        column = 'Lot Area'
        if column in data.columns:
            frame = pandas.DataFrame(data[column])
            normalized_data = normalized_data.join(frame, how='outer')
            data = data.loc[:, data.columns != column]

        column = 'Year Built'
        if column in data.columns:
            frame = pandas.DataFrame(data[column])
            normalized_data = normalized_data.join(frame, how='outer')
            data = data.loc[:, data.columns != column]

        column = 'Year Remod/Add'
        if column in data.columns:
            frame = pandas.DataFrame(data[column])
            normalized_data = normalized_data.join(frame, how='outer')
            data = data.loc[:, data.columns != column]

        column = 'Fireplaces'
        if column in data.columns:
            frame = pandas.DataFrame(data[column])
            normalized_data = normalized_data.join(frame, how='outer')
            data = data.loc[:, data.columns != column]

        column = 'Garage Area'
        if column in data.columns:
            frame = pandas.DataFrame(data[column])
            normalized_data = normalized_data.join(frame, how='outer')
            data = data.loc[:, data.columns != column]

        column = 'Pool Area'
        if column in data.columns:
            frame = pandas.DataFrame(data[column])
            normalized_data = normalized_data.join(frame, how='outer')
            data = data.loc[:, data.columns != column]

        column = 'Yr Sold'
        if column in data.columns:
            frame = pandas.DataFrame(data[column])
            normalized_data = normalized_data.join(frame, how='outer')
            data = data.loc[:, data.columns != column]

        column = 'SalePrice'
        if column in data.columns:
            frame = pandas.DataFrame(data[column])
            normalized_data = normalized_data.join(frame, how='outer')
            data = data.loc[:, data.columns != column]

        return normalized_data

    def squish_values(self):
        """
        Squishes vector values to be between 0 and 1.
        :return:
        """
        pass

    def separate_categories(self):
        pass

class BackPropNet():
    """
    Neural Net implementing back propagation.
    """
    def __init__(self, data):
        self.hidden_layer_size = 3
        self.network = []
        self._create_architecture(data)

    def _create_architecture(self, data):
        """
        Creates neural net architecture.
        Each layer has sets of weights equal to the number of nodes in the layer.
        Each set of weights has x values where x is the number of nodes in the previous layer, plus a bias.
        Weight values are randomized values near 0, using a normal distribution.
        :param data:
        :return:
        """
        # Create first hidden layer.
        hidden_layer_1 = []
        for index in range(self.hidden_layer_size):
            hidden_layer_1.append([
                (numpy.random.randn() * 0.001) for index in range(len(data) + 1)
            ])

        # Create second hidden layer.
        hidden_layer_2 = []
        for index in range(self.hidden_layer_size):
            hidden_layer_2.append([
                (numpy.random.randn() * 0.001) for index in range(self.hidden_layer_size + 1)
            ])

        # Create output layer
        output_layer = [[
            (numpy.random.randn() * 0.001) for index in range(self.hidden_layer_size + 1)
        ]]

        # Add layers to network.
        self.network.append(hidden_layer_1)
        self.network.append(hidden_layer_2)
        self.network.append(output_layer)

        logger.info('Network:')
        index = 0
        for layer in self.network:
            logger.info('Layer {0}: {1}'.format(index, layer))
            index += 1

    def activation(self, weights, inputs):
        """
        Calculate if neuron fires or not, based on inputs and weights being calculated and passed into sigmoid.
        :param weights: Weights of given layer.
        :param inputs: Inputs to calculate with.
        :return: Calculated value, passed through sigmoid.
        """
        # Calculate single value based on inputs and weights.
        value = weights[-1]
        for index in range(len(weights) - 1):
            value += weights[index] * inputs[index]

        # Pass into sigmoid, then return result.
        return self.sigmoid(value)

    def sigmoid(self, value):
        """
        Calculate the sigmoid of the provided value.
        :param value: Single value to calculate.
        :return: Sigmoid of value.
        """
        return ( 1 / (1 + math.exp(-value)) )

    def reverse_sigmoid(self, value):
        """
        Calculate the derivative of sigmoid.
        :param value: Single value to calculate.
        :return: Reverse sigmoid of value.
        """
        return ( self.sigmoid(value) * ( 1 - self.sigmoid(value) ) )

    def forward_propagate(self, inputs):
        """
        Walk forward through the neural network.
        :param inputs: Initial inputs for network.
        :return: Output results of network.
        """
        outputs = None
        for layer in self.network:
            outputs = []
            for neuron in layer:
                outputs = inputs.append(self.activation(neuron[0], inputs))
            inputs = outputs
        return outputs

    def backward_propagate(self, inputs):
        """
        Walk backward through the neural network, using derivatives.
        :param inputs: Original output of network.
        :return: ???
        """

class ResultTracker():
    """
    Tracks statistics and progress of main Neural Net class.
    """
