"""
Logic to track results of neural nets.
"""

# User Class Imports.
from resources import logging


# Initialize logging.
logger = logging.get_logger(__name__)


class ResultTracker():
    """
    Tracks statistics and progress of Neural Net.
    """
    def __init__(self):
        self.iterations = []
        self.best_iteration_index = 0

    def add_iteration(self, weights, error_margin):
        """
        Adds a new set of results to track.
        :param weights: Weights of current iteration.
        :param error_margin: Error margin of current iteration.
        """
        new_iteration = [weights, error_margin]
        self.iterations.append(new_iteration)
        # logger.info('Iteration {0}: {1}'.format(len(self.iterations) - 1, new_iteration))
        #
        # logger.info('Previous Best: {0}   New Value: {1}'
        #             .format(self.iterations[self.best_iteration_index][1], error_margin))

        # Calculate best iteration thus far. Based on total error margin.
        if error_margin < self.iterations[self.best_iteration_index][1]:
            self.best_iteration_index = len(self.iterations) - 1

    def continue_training_check(self):
        """
        Determines if Neural Net should continue training.
        :return: True on continued training. False on training complete.
        """
        total_iterations = len(self.iterations)

        # Make Neural Net iterate at least 10 times.
        if total_iterations <= 10:
            return True

        # Check if Neural Net is still improving. Continue if progress has made in last 5 iterations.
        if self.best_iteration_index > (total_iterations - 5):
            return True

        return False
