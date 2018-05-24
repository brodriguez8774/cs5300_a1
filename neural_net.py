"""
Neural Net logic.
"""

# System Imports.
from sklearn import preprocessing
import math, numpy, pandas

# User Class Imports.
from resources import logging


# Initialize logging.
logger = logging.get_logger(__name__)


# Disable false positive warning. Occurred when trying to remove NaN values from dataset.
pandas.options.mode.chained_assignment = None


class Normalizer():
    """
    Handles data normalization.
    """
    def normalize_data(self, orig_data):
        """
        Normalizes and returns provided dataset.
        :param data: A pandas array of housing data to normalize.
        :return: The normalized array of housing data.
        """
        # Print out data.
        # logger.info(orig_data)
        # logger.info(orig_data.columns.values)
        # logger.info(orig_data['Alley'])

        # Remove NaN references.
        # orig_data = orig_data.fillna(value='NaN')

        # Address individual columns.
        normalized_data = pandas.DataFrame()

        continuous_columns = [
            'Lot Frontage', 'Lot Area', 'Mas Vnr Area', 'BsmtFin SF 1', 'BsmtFin SF 2', 'Bsmt Unf SF', 'Total Bsmt SF',
            # '1st Flr SF', '2nd Flr SF', 'Low Qual Fin SF', 'Gr Liv Area', 'Garage Area', 'Wood Deck SF',
            # 'Open Porch SF', 'Enclosed Porch', '3Ssn Porch', 'Screen Porch', 'Pool Area', 'Misc Val',
        ]
        discrete_columns = [
            # 'Year Built', 'Year Remod/Add', 'Bsmt Full Bath', 'Bsmt Half Bath', 'Full Bath', 'Half Bath',
            # 'Bedroom AbvGr', 'Kitchen AbvGr', 'TotRms AbvGrd', 'Fireplaces', 'Garage Yr Blt', 'Garage Cars', 'Mo Sold',
            # 'Yr Sold',
        ]
        categorical_columns = [
            # 'MS SubClass', 'MS Zoning', 'Street', 'Alley', 'Land Contour', 'Lot Config', 'Neighborhood', 'Condition 1',
            # 'Condition 2', 'Bldg Type', 'House Style', 'Roof Style', 'Roof Matl', 'Exterior 1st', 'Exterior 2nd',
            # 'Mas Vnr Type', 'Foundation', 'Heating', 'Central Air', 'Garage Type', 'Misc Feature', 'Sale Type',
            # 'Sale Condition',
        ]
        categorical_dict = {}
        ordinal_columns = [
            # 'Lot Shape', 'Land Slope', 'Overall Qual', 'Overall Cond', 'Exter Qual', 'Exter Cond', 'Bsmt Qual',
            # 'Bsmt Cond', 'Bsmt Exposure', 'BsmtFin Type 1', 'BsmtFin Type 2', 'Heating QC', 'Electrical',
            # 'Kitchen Qual', 'Functional', 'Fireplace Qu', 'Garage Finish', 'Garage Qual', 'Garage Cond', 'Paved Drive',
            # 'Pool QC', 'Fence',
        ]
        ordinal_dict = {}
        ignored_columns = ['Utilities',]
        target_column = ['SalePrice',]

        # Process continuous data.
        for column in continuous_columns:
            if column in orig_data.columns:
                # Normalize.
                self.squish_values(orig_data, column)

                # Remove NaN references.
                orig_data[column] = orig_data[column].fillna(value=0)

                # Add column to normalized data list.
                frame = pandas.DataFrame(orig_data[column])
                normalized_data = normalized_data.join(frame, how='outer')
                orig_data = orig_data.loc[:, orig_data.columns != column]

        # Process discreet data. Currently handles as if it were continuous.
        for column in discrete_columns:
            if column in orig_data.columns:
                # Normalize.
                self.squish_values(orig_data, column)

                # Remove NaN references.
                orig_data[column] = orig_data[column].fillna(value=0)

                # Add column to normalized data list.
                frame = pandas.DataFrame(orig_data[column])
                normalized_data = normalized_data.join(frame, how='outer')
                orig_data = orig_data.loc[:, orig_data.columns != column]

        # Process categorical data.
        for column in categorical_columns:
            if column in orig_data.columns:
                # Remove NaN references.
                orig_data[column] = orig_data[column].fillna(value='NaN')

                # Turn single column into onehot matrix.
                onehot_tuple = self.create_onehot(orig_data[column])
                # Add onehot matrix to normalized data list.
                frame = pandas.DataFrame(onehot_tuple[0])
                normalized_data = normalized_data.join(frame, how='outer')
                orig_data = orig_data.loc[:, orig_data.columns != column]

                # Save newly created columns associated with the original column.
                categorical_dict[column] = onehot_tuple[1]

        # Process ordinal data. Currently handles as categorical. Perhaps a better way?
        for column in ordinal_columns:
            if column in orig_data.columns:
                # Remove NaN references.
                orig_data[column] = orig_data[column].fillna(value='NaN')

                # Turn single column into onehot matrix.
                onehot_tuple = self.create_onehot(orig_data[column])
                # Add onehot matrix to normalized data list.
                frame = pandas.DataFrame(onehot_tuple[0])
                normalized_data = normalized_data.join(frame, how='outer')
                orig_data = orig_data.loc[:, orig_data.columns != column]

                # Save newly created columns associated with the original column.
                categorical_dict[column] = onehot_tuple[1]

        # Columns to be ignored.
        for column in ignored_columns:
            orig_data = orig_data.loc[:, orig_data.columns != column]

        # Target Data. Should this be normalized as well?
        # Add column to normalized data list.
        frame = pandas.DataFrame(orig_data[target_column])
        normalized_data = normalized_data.join(frame, how='outer')

        return normalized_data

    def squish_values(self, orig_data, column):
        """
        Squishes vector values to be between 0 and 1.
        Referenced from http://lamda.nju.edu.cn/weixs/project/CNNTricks/CNNTricks.html
        """
        # logger.info('Pre Normalization: {0}'.format(orig_data[column]))
        x_value = orig_data[column]
        x_value -= numpy.mean(x_value, axis=0)  # Zero-center.
        x_value /= numpy.std(x_value, axis=0)   # Normalize.
        # logger.info('Post Normalization: {0}'.format(orig_data[column]))


    def create_onehot(self, column):
        """
        Creates a onehot of data based on the given column.
        Each unique value is turned into a new column.
        The index will have a 1 on the respective valid column, and 0 for all others.
        :param column: Column of data to onehot.
        :return: Onehot of data. Columns are denoted by "columnName__value".
        """
        # Used Dillon's magic code as reference to create onehot.
        label_enc = preprocessing.LabelEncoder()
        label_enc.fit(column)
        int_label = label_enc.transform(column)
        int_label = int_label.reshape(-1,1)
        column_enc = preprocessing.OneHotEncoder(sparse=False)
        column_onehot = column_enc.fit_transform(int_label)

        # Create meaningful labels for onehot, using initial category values.
        new_labels = []
        for label in label_enc.classes_:
            new_labels.append(str(column.name) + '__' + str(label))
        column_array = pandas.DataFrame(column_onehot, columns=new_labels)

        return [column_array, label_enc.classes_]


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
        :param data: Data to reference for input layer count.
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

        # logger.info('Network:')
        index = 0
        for layer in self.network:
            # logger.info('Layer {0}: {1}'.format(index, layer))
            index += 1

    def _activation(self, weights, inputs):
        """
        Calculate if neuron fires or not, based on inputs and weights being calculated and passed into sigmoid.
        :param weights: Weights of given layer.
        :param inputs: Inputs to calculate with.
        :return: Calculated value, passed through sigmoid.
        """
        # Calculate single value based on inputs and weights.
        activation_value = weights[-1]
        for index in range(len(weights) - 1):
            activation_value += (weights[index] * inputs[index])

        # Pass into sigmoid, then return result.
        return self._sigmoid(activation_value)

    def _sigmoid(self, value):
        """
        Calculate the sigmoid of the provided value.
        :param value: Single value to calculate.
        :return: Sigmoid of value.
        """
        return ( 1 / (1 + math.exp(-value)) )

    def _reverse_sigmoid(self, value):
        """
        Calculate the derivative of sigmoid.
        :param value: Single value to calculate.
        :return: Reverse sigmoid of value.
        """
        return ( self._sigmoid(value) * ( 1 - self._sigmoid(value) ) )

    def _forward_propagate(self, inputs):
        """
        Walk forward through the neural network.
        :param inputs: Initial inputs for network.
        :return: Output results of network.
        """
        outputs = None
        # Iterate through each value in network, using previous outputs as new inputs.
        for index in range(len(self.network)):
            outputs = []
            for neuron in self.network[index]:
                outputs.append(self._activation(neuron, inputs))
            inputs = outputs
        return outputs

    def _backward_propagate(self, inputs):
        """
        Walk backward through the neural network, using derivatives.
        :param inputs: Original output of network.
        :return: ???
        """
        pass

    def _calculate_delta(self, prediction, targets):
        """
        Calculates an error delta.
        :param prediction: Current prediction.
        :param targets: Desired prediction.
        :return: Delta of error difference.
        """
        return ( (targets - prediction) ** 2)

    def train(self, features, targets):
        """
        Trains net based on provided data.
        :param data: Data to train on.
        """
        prediction = []
        for index in range(len(features)):
            prediction.append(self._forward_propagate(features[index]))
        delta_error = self._calculate_delta(prediction, targets)
        logger.info('Delta Error: {0}'.format(delta_error))

    def predict(self, data):
        """
        Makes a prediction with the given data.
        :param data: Data to predict with.
        :return: Prediction of values.
        """
        return self._forward_propagate(data)

class ResultTracker():
    """
    Tracks statistics and progress of main Neural Net class.
    """
