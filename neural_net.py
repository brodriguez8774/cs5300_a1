"""
Neural Net logic.
"""

# System Imports.
from sklearn import preprocessing
import numpy, pandas

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
            '1st Flr SF', '2nd Flr SF', 'Low Qual Fin SF', 'Gr Liv Area', 'Garage Area', 'Wood Deck SF',
            'Open Porch SF', 'Enclosed Porch', '3Ssn Porch', 'Screen Porch', 'Pool Area', 'Misc Val',
        ]
        discrete_columns = [
            'Year Built', 'Year Remod/Add', 'Bsmt Full Bath', 'Bsmt Half Bath', 'Full Bath', 'Half Bath',
            'Bedroom AbvGr', 'Kitchen AbvGr', 'TotRms AbvGrd', 'Fireplaces', 'Garage Yr Blt', 'Garage Cars', 'Mo Sold',
            'Yr Sold',
        ]
        categorical_columns = [
            'MS SubClass', 'MS Zoning', 'Street', 'Alley', 'Land Contour', 'Lot Config', 'Neighborhood', 'Condition 1',
            'Condition 2', 'Bldg Type', 'House Style', 'Roof Style', 'Roof Matl', 'Exterior 1st', 'Exterior 2nd',
            'Mas Vnr Type', 'Foundation', 'Heating', 'Central Air', 'Garage Type', 'Misc Feature', 'Sale Type',
            'Sale Condition',
        ]
        categorical_dict = {}
        ordinal_columns = [
            'Lot Shape', 'Land Slope', 'Overall Qual', 'Overall Cond', 'Exter Qual', 'Exter Cond', 'Bsmt Qual',
            'Bsmt Cond', 'Bsmt Exposure', 'BsmtFin Type 1', 'BsmtFin Type 2', 'Heating QC', 'Electrical',
            'Kitchen Qual', 'Functional', 'Fireplace Qu', 'Garage Finish', 'Garage Qual', 'Garage Cond', 'Paved Drive',
            'Pool QC', 'Fence',
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

        # Process target data. Handles same as continous.
        for column in target_column:
            if column in orig_data.columns:
                # Normalize.
                self.squish_values(orig_data, column)

                # Remove NaN references.
                orig_data[column] = orig_data[column].fillna(value=0)

                # Add column to normalized data list.
                frame = pandas.DataFrame(orig_data[column])
                normalized_data = normalized_data.join(frame, how='outer')
                orig_data = orig_data.loc[:, orig_data.columns != column]

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
        self.hidden_layer_size = 10
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
        for neuron_index in range(self.hidden_layer_size):
            hidden_layer_1.append({})
            dict_values = []
            for input_index in range(len(data.columns)):
                 dict_values += [numpy.random.randn() * 0.001]
            hidden_layer_1[neuron_index]['weights'] = dict_values

        # Create second hidden layer.
        hidden_layer_2 = []
        for neuron_index in range(self.hidden_layer_size):
            hidden_layer_2.append({})
            dict_values = []
            for input_index in range(self.hidden_layer_size + 1):
                dict_values += [numpy.random.randn() * 0.001]
            hidden_layer_2[neuron_index]['weights'] = dict_values

        # Create output layer
        output_layer = []
        for index in range(len(data.values)):
            output_layer.append({})
            dict_values = []
            for input_index in range(self.hidden_layer_size + 1):
                dict_values += [numpy.random.randn() * 0.001]
            output_layer[index]['weights'] = dict_values

        # Add layers to network.
        self.network.append(hidden_layer_1)
        self.network.append(hidden_layer_2)
        self.network.append(output_layer)

        logger.info('')
        logger.info('Network:')
        index = 0
        for layer in self.network:
            logger.info('Layer {0} has {1} neurons: {2}'.format(index, len(layer), layer))
            index += 1
        logger.info('')
        logger.info('')

    def _activation(self, weights, inputs):
        """
        Calculate how strongly neuron fires, based on inputs and weights being calculated and passed into sigmoid.
        Multiplies inputs and weights, adds bias, passes into sigmoid, then returns result.
        :param weights: Weights of given layer.
        :param inputs: Inputs to calculate with.
        :return: Calculated value, passed through sigmoid.
        """
        bias = weights[-1]
        return self._sigmoid(numpy.dot(inputs, weights[:-1].copy()) + bias)

    def _sigmoid(self, value):
        """
        Calculate the sigmoid of the provided value.
        :param value: Value to calculate with.
        :return: Sigmoid of value.
        """
        return ( 1 / (1 + numpy.exp(-value)) )

    def _reverse_sigmoid(self, value):
        """
        Calculate the derivative of sigmoid.
        :param value: Value to calculate with.
        :return: Reverse sigmoid of value.
        """
        return (value * (1 - value))

    def _forward_propagate(self, inputs):
        """
        Walk forward through the neural network.
        :param inputs: Initial inputs for network.
        :return: Output results of network.
        """
        outputs = None
        # Iterate through each layer in network, using previous outputs as new inputs.
        for index in range(len(self.network)):
            outputs = []
            # Iterate through each neuron in the given layer, determining activation and saving as the layer output.
            for neuron in self.network[index]:
                neuron['output'] = self._activation(neuron['weights'], inputs)
                outputs.append(neuron['output'])
            inputs = outputs
        return outputs

    def _backward_propagate_error(self, targets):
        """
        Walk backward through the neural network, using derivatives.

        Due to issues implementing myself, this function is heavily referenced from
        https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/

        :param targets: Desired output of network.
        """
        # Iterate backward through all network layers.
        for layer_index in reversed(range(len(self.network))):
            layer = self.network[layer_index]
            errors = []

            # Check if starting backprop.
            if layer_index == (len(self.network) - 1):
                # Backprop already started.
                for neuron_index in range(len(layer)):
                    neuron = layer[neuron_index]
                    errors.append(targets[neuron_index] - neuron['output'])
            else:
                # Backprop just starting. Multiply neuron weights and delta to establish error.
                for neuron_index in range(len(layer)):
                    error = 0
                    for neuron in self.network[layer_index + 1]:
                        error += (neuron['weights'][neuron_index] * neuron['delta'])
                    errors.append(error)

            # Calculate the delta error of the current layer.
            for neuron_index in range(len(layer)):
                neuron = layer[neuron_index]
                neuron['delta'] = errors[neuron_index] * self._reverse_sigmoid(neuron['output'])

    def _calculate_delta(self, prediction, targets):
        """
        Calculates an error delta.
        :param prediction: Current prediction.
        :param targets: Desired prediction.
        :return: Delta of error difference.
        """
        return ( (targets - prediction) ** 2)

    def _update_weights(self, row, learn_rate):
        """
        Iterate through the network and all neurons, using the delta and learn rate to update weights and biases.
        :param row: A single row/record of feature inputs.
        :param learn_rate: The rate for the weights to learn by.
        """
        # Iterate through all network layers once more.
        for layer_index in range(len(self.network)):
            inputs = row[:-1]
            # If not first layer.
            if layer_index != 0:
                inputs = []
                # Add up all neuron inputs of previous layer?
                for neuron in self.network[layer_index - 1]:
                    inputs += neuron['output']

            # Iterate through all neurons and neuron inputs.
            # Multiplies the input, delta, and learn rate to update weights.
            for neuron in self.network[layer_index]:
                for input_index in range(len(inputs)):
                    neuron['weights'][input_index] += learn_rate * neuron['delta'] * inputs[input_index]
                # Also update the bias for the layer.
                neuron['weights'][-1] += learn_rate * neuron['delta']

    def train(self, features, targets, learn_rate=0.5):
        """
        Trains net based on provided data.
        :param features: Set of inputs to use for training.
        :param targets: Desired output after prosessing features.
        :param learn_rate: The rate of learning based off of errors.
        :return: The total accumulated error between predictions and values, while training.
        """
        total_error = 0
        # Iterate through each row of inputs in features.
        for index in range(len(features)):
            # Pass inputs through neural net to create predictions.
            outputs = self.predict(features[index])

            # Determine error value between predicted output and targets.
            delta_error = 0
            for error_index in range(len(targets)):
                delta_error += self._calculate_delta(outputs[error_index], targets[error_index])
            total_error += delta_error

            # Backstep through network to correct and modify weights for future predictions.
            self._backward_propagate_error(targets)
            self._update_weights(features[index], learn_rate)
        return total_error

    def predict(self, data):
        """
        Makes a prediction with the given data.
        :param data: Data to predict with.
        :return: Prediction of values.
        """
        return self._forward_propagate(data)
