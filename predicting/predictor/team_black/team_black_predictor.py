"""
Created on 08.11.2017

@author: rmueller
"""
from model.IPredictor import IPredictor

from model.StockData import StockData
from utils import load_keras_sequential, save_keras_sequential, read_stock_market_data
from model.CompanyEnum import CompanyEnum
from logger import logger
from matplotlib import pyplot as plt
from keras.models import Sequential
from definitions import PERIOD_1, PERIOD_2
from keras.callbacks import History
from keras.layers import Dense
import numpy as np

TEAM_NAME = "team_black"

RELATIVE_PATH = 'predicting/predictor/' + TEAM_NAME + '/' + TEAM_NAME + '_predictor_data'
MODEL_FILE_NAME_STOCK_A = TEAM_NAME + '_predictor_stock_a_network'
MODEL_FILE_NAME_STOCK_B = TEAM_NAME + '_predictor_stock_b_network'

# Neural network configuration -> TODO see Keras Documentation
# INPUT_SIZE = 42  # TODO

MODEL_LENGTH=10


class TeamBlackBasePredictor(IPredictor):
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

        # TODO DONE compile loaded model
        self.model.compile(loss='mean_squared_error', optimizer='sgd')


    def doPredict(self, data: StockData) -> float:
        """
        Use the loaded trained neural network to predict the next stock value.
    
        Args:
          data: historical stock values of a company

        Returns:
          predicted next stock value for that company
        """

        # self.model.predict_classes

        stocks = data.get_values()
        length = len(stocks)

        last = stocks[length - (MODEL_LENGTH + 1):]

        xtrain = []

        tuple = []
        for j in range(MODEL_LENGTH):
            increase = (last[j + 1] - last[j]) / last[j]
            if increase > 1:
                increase = 1

            if increase < -1:
                increase = -1

            tuple.append(increase)

        xtrain.append(tuple)


        np_xtrain = np.array(xtrain)

        result = self.model.predict(np_xtrain)


        value = result[0][0]


        absolute_last = data.get_values()[-1]
        return absolute_last + absolute_last * value



class TeamBlackStockAPredictor(TeamBlackBasePredictor):
    """
    Predictor for stock A based on an already trained neural network.
    """

    def __init__(self):
        """
        Constructor: Load the trained and stored neural network.
        """
        super().__init__(MODEL_FILE_NAME_STOCK_A)


class TeamBlackStockBPredictor(TeamBlackBasePredictor):
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

    stocks = training_data.get_values()

    xtrain = []
    ytrain = []

    for i in range(len(stocks) - (MODEL_LENGTH + 2)):
        tuple = []
        for j in range(MODEL_LENGTH):
            increase = (stocks[i + j + 1] - stocks[i + j]) / stocks[i + j]
            if increase > 1:
                increase = 1

            if increase < -1:
                increase = -1

            tuple.append(increase)

        xtrain.append(tuple)

        j = MODEL_LENGTH
        increase = (stocks[i + j + 1] - stocks[i + j]) / stocks[i + j]
        if increase > 1:
            increase = 1

        if increase < -1:
            increase = -1

        tuple2 = []
        tuple2.append(increase)
        ytrain.append(tuple2)

    np_xtrain = np.array(xtrain)
    np_ytrain = np.array(ytrain)


    BATCH_SIZE = 100
    EPOCHS = 50

    network.fit(np_xtrain, np_ytrain, epochs=EPOCHS, batch_size=BATCH_SIZE)


    stocks = test_data.get_values()

    xtrain = []
    ytrain = []

    for i in range(len(stocks) - (MODEL_LENGTH + 2)):
        tuple = []
        for j in range(MODEL_LENGTH):
            increase = (stocks[i + j + 1] - stocks[i + j]) / stocks[i + j]
            if increase > 1:
                increase = 1

            if increase < -1:
                increase = -1

            tuple.append(increase)

        xtrain.append(tuple)

        j = MODEL_LENGTH
        increase = (stocks[i + j + 1] - stocks[i + j]) / stocks[i + j]
        if increase > 1:
            increase = 1

        if increase < -1:
            increase = -1

        tuple2 = []
        tuple2.append(increase)
        ytrain.append(tuple2)

    np_xtrain = np.array(xtrain)
    np_ytrain = np.array(ytrain)

    score = network.evaluate(np_xtrain, np_ytrain, batch_size=BATCH_SIZE)

    # Save trained model: separate network structure (stored as JSON) and trained weights (stored as HDF5)
    save_keras_sequential(network, RELATIVE_PATH, filename_to_save)


def create_model() -> Sequential:
    model = Sequential()

    HIDDEN_SIZE=MODEL_LENGTH - 1
    INPUT_SIZE=MODEL_LENGTH
    OUTPUT_SIZE=1

    model.add(Dense(HIDDEN_SIZE, input_dim=INPUT_SIZE, activation='tanh'))
    model.add(Dense(HIDDEN_SIZE, activation='relu'))
    model.add(Dense(HIDDEN_SIZE, activation='relu'))
    model.add(Dense(HIDDEN_SIZE, activation='relu'))
    model.add(Dense(OUTPUT_SIZE, activation='linear'))

    model.compile(loss='mean_squared_error', optimizer='sgd')

    return model


def draw_history(history: History):
    plt.figure()
    plt.plot(history.history['loss'])
    plt.title('training loss / testing loss by epoch')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['training', 'testing'], loc='best')


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
