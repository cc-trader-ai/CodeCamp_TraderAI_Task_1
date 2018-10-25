"""
Created on 08.11.2017

@author: jtymoszuk
"""
import math

from model.CompanyEnum import CompanyEnum
from model.Portfolio import Portfolio
from model.StockMarketData import StockMarketData
from model.ITrader import ITrader
from model.Order import OrderList
from model.IPredictor import IPredictor


class TeamGreenSimpleTrader(ITrader):
    """
    Simple Trader generates Order based on simple logic, input data and prediction from NN-Engine
    """

    def __init__(self, stock_a_predictor: IPredictor, stock_b_predictor: IPredictor):
        """
        Constructor
        """
        self.stock_a_predictor = stock_a_predictor
        self.stock_b_predictor = stock_b_predictor

    def doTrade(self, portfolio: Portfolio, current_portfolio_value: float,
                stock_market_data: StockMarketData) -> OrderList:
        """
        Generate action to be taken on the "stock market"
    
        Args:
          portfolio : current Portfolio of this trader
          current_portfolio_value : value of Portfolio at given Momemnt
          stock_market_data : StockMarketData for evaluation

        Returns:
          A OrderList instance, may be empty never None
        """

        result = OrderList()

        predictions = {
            CompanyEnum.COMPANY_A: self.stock_a_predictor.doPredict(stock_market_data[CompanyEnum.COMPANY_A]),
            CompanyEnum.COMPANY_B: self.stock_b_predictor.doPredict(stock_market_data[CompanyEnum.COMPANY_B])
        }

        zuwachs = {}

        # sell companies which get worse
        for company in CompanyEnum:
            prediction = predictions[company]
            anzahl = portfolio.get_amount(company)
            current = stock_market_data.get_most_recent_price(company)

            zuwachs[company] = prediction / current

            # sell if getting worse
            if anzahl > 0:
                if current > prediction:
                    result.sell(company, anzahl)

        best = sorted(zuwachs, key=zuwachs.__getitem__)[::-1]

        currentCash = portfolio.cash

        for b in best:
            prediction = predictions[b]
            if prediction > 1:
                current = stock_market_data.get_most_recent_price(b)
                count = math.floor(currentCash / current)
                result.buy(b, count)
                currentCash -= count * current

        return result
