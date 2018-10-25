"""
Created on 08.11.2017

@author: rmueller
"""
from typing import List

import numpy
from keras.layers import Dense

from model.IPredictor import IPredictor

from model.StockData import StockData
from utils import load_keras_sequential, save_keras_sequential, read_stock_market_data
from model.CompanyEnum import CompanyEnum
from logger import logger
from matplotlib import pyplot as plt
from keras.models import Sequential
from definitions import PERIOD_1, PERIOD_2
from keras.callbacks import History

TEAM_NAME = "team_green"

RELATIVE_PATH = 'predicting/predictor/' + TEAM_NAME + '/' + TEAM_NAME + '_predictor_data'
MODEL_FILE_NAME_STOCK_A = TEAM_NAME + '_predictor_stock_a_network'
MODEL_FILE_NAME_STOCK_B = TEAM_NAME + '_predictor_stock_b_network'

# Neural network configuration -> TODO see Keras Documentation
INPUT_SIZE = 42  # TODO

BATCH_SIZE = 100
EPOCHS = 150

WINDOW_SIZE = 35
HIDDEN_SIZE = 50


class TeamGreenBasePredictor(IPredictor):
    """
    Predictor based on an already trained neural network.
    """

    def __init__(self, nn_filename: str):
        """
        Constructor: Load the trained and stored neural network.

        Args:
            nn_filename: The filename to load the trained data from
        """
        # Try loading a stored trained neural network...
        self.model = load_keras_sequential(RELATIVE_PATH, nn_filename)
        assert self.model is not None

        self.model.compile(loss='mean_squared_error', optimizer='sgd')

    def doPredict(self, data: StockData) -> float:
        """
        Use the loaded trained neural network to predict the next stock value.
    
        Args:
          data: historical stock values of a company

        Returns:
          predicted next stock value for that company
        """
        input = numpy.array([data.get_values()[-WINDOW_SIZE:]])
        output = self.model.predict(input)

        print("predicted %f for price %f" % (output[0], data.get_last()[1]))
        return output[0] + data.get_last()[1]


class TeamGreenStockAPredictor(TeamGreenBasePredictor):
    """
    Predictor for stock A based on an already trained neural network.
    """

    def __init__(self):
        """
        Constructor: Load the trained and stored neural network.
        """
        super().__init__(MODEL_FILE_NAME_STOCK_A)


class TeamGreenStockBPredictor(TeamGreenBasePredictor):
    """
    Predictor for stock B based on an already trained neural network.
    """

    def __init__(self):
        """
        Constructor: Load the trained and stored neural network.
        """
        super().__init__(MODEL_FILE_NAME_STOCK_B)


###############################################################################
# The following code trains and stores the corresponding neural network
###############################################################################


def learn_nn_and_save(training_data: StockData, test_data: StockData, filename_to_save: str):
    network = create_model()

    network.compile(loss='mean_squared_error', optimizer='sgd')


    values = training_data.get_values()
    setCount = len(values) - (WINDOW_SIZE + 1)



    xtrain = []
    for element in range(0, setCount):
        xtrain.append(numpy.array(values[element:WINDOW_SIZE+element]))

    X_TRAIN = numpy.array(xtrain)

    Y_TRAIN = numpy.empty(setCount, dtype=numpy.float)

    offset = WINDOW_SIZE - 1
    for element in range(0, setCount):
        current = values[element + offset]
        next = values[element + offset + 1]
        Y_TRAIN[element] = 1.0 if next > current else -1.0

    history = network.fit(X_TRAIN, Y_TRAIN, epochs=EPOCHS, batch_size=BATCH_SIZE)
    draw_history(history)

    # Save trained model: separate network structure (stored as JSON) and trained weights (stored as HDF5)
    save_keras_sequential(network, RELATIVE_PATH, filename_to_save)


def create_model() -> Sequential:
    network = Sequential()

    network.add(Dense(HIDDEN_SIZE, input_dim=WINDOW_SIZE, activation='relu'))
    network.add(Dense(1, activation='tanh'))

    return network


def draw_history(history: History):
    plt.figure()
    plt.plot(history.history['loss'])
    plt.title('training loss / testing loss by epoch')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['training', 'testing'], loc='best')

    plt.show()


def draw_prediction(dates: list, awaited_results: list, prediction_results: list):
    plt.figure()

    plt.plot(dates[INPUT_SIZE:], awaited_results, color="black")  # current prices in reality
    plt.plot(dates[INPUT_SIZE:], prediction_results, color="green")  # predicted prices by neural network
    plt.title('current prices / predicted prices by date')
    plt.ylabel('price')
    plt.xlabel('date')
    plt.legend(['current', 'predicted'], loc='best')

    plt.show()


if __name__ == "__main__":
    logger.debug("Data loading...")
    training_stock_market_data = read_stock_market_data([CompanyEnum.COMPANY_A, CompanyEnum.COMPANY_B], [PERIOD_1])
    test_stock_market_data = read_stock_market_data([CompanyEnum.COMPANY_A, CompanyEnum.COMPANY_B], [PERIOD_2])

    company_a_training_stock_data: StockData = training_stock_market_data[CompanyEnum.COMPANY_A]
    company_a_test_stock_data: StockData = test_stock_market_data[CompanyEnum.COMPANY_A]

    logger.debug(f"Data for Stock A loaded")
    learn_nn_and_save(company_a_training_stock_data, company_a_test_stock_data, MODEL_FILE_NAME_STOCK_A)

    company_b_training_stock_data: StockData = training_stock_market_data[CompanyEnum.COMPANY_B]
    company_b_test_stock_data: StockData = test_stock_market_data[CompanyEnum.COMPANY_B]

    logger.debug(f"Data for Stock B loaded")
    learn_nn_and_save(company_b_training_stock_data, company_b_test_stock_data, MODEL_FILE_NAME_STOCK_B)
