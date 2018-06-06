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
    def __init__(self, min_iterations):
        self.iterations = []
        self.best_iteration_index = 0
        self.min_iterations = min_iterations

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

    def continue_training_check(self, max_epochs=None, iteration_diff=10):
        """
        Determines if Neural Net should continue training.
        :param max_epochs: Optional arg to set a max epoch count on training iterations.
        :return: True on continued training. False on training complete.
        """
        exit_training = False
        total_iterations = len(self.iterations)

        # Make Neural Net iterate a minimum set of times.
        if total_iterations <= self.min_iterations:
            exit_training = True

        # Check if Neural Net is still improving.
        if max_epochs is None:
            # Continue if progress has made in last x iterations.
            if self.best_iteration_index > (total_iterations - iteration_diff):
                exit_training = True
        else:
            # Check if under epoch count.
            if total_iterations < max_epochs:
                # Continue if progress has been made in last x iterations.
                if self.best_iteration_index > (total_iterations - iteration_diff):
                    exit_training = True

        return exit_training
