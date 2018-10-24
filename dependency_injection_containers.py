'''
Created on 09.11.2017

@author: jtymoszuk
'''
import dependency_injector.containers as containers
import dependency_injector.providers as providers

from model.CompanyEnum import CompanyEnum
from predicting.predictor.reference.random_predictor import RandomPredictor
from predicting.predictor.reference.perfect_predictor import PerfectPredictor
from trading.trader.reference.simple_trader import SimpleTrader
from trading.trader.reference.buy_and_hold_trader import BuyAndHoldTrader
from predicting.predictor.team_blue.team_blue_predictor import TeamBlueStockAPredictor, \
    TeamBlueStockBPredictor
from predicting.predictor.team_green.team_green_predictor import TeamGreenStockAPredictor, \
    TeamGreenStockBPredictor
from predicting.predictor.team_black.team_black_predictor import TeamBlackStockBPredictor, \
    TeamBlackStockAPredictor
from predicting.predictor.team_red.team_red_predictor import TeamRedStockBPredictor, \
    TeamRedStockAPredictor
from trading.trader.team_blue.team_blue_simple_trader import TeamBlueSimpleTrader
from trading.trader.team_green.team_green_simple_trader import TeamGreenSimpleTrader
from trading.trader.team_black.team_black_simple_trader import TeamBlackSimpleTrader
from trading.trader.team_red.team_red_simple_trader import TeamRedSimpleTrader


class Predictors(containers.DeclarativeContainer):
    """IoC container of predictor providers."""
 
    """ Random predictor delivering value of last share +- Random[0,1]"""
    RandomPredictor = providers.Factory(RandomPredictor)
    
    """ Perfect predictors knowing future"""
    # Task 0 and Task 2
    PerfectPredictor_stock_a = providers.Factory(PerfectPredictor, CompanyEnum.COMPANY_A)    
    PerfectPredictor_stock_b = providers.Factory(PerfectPredictor, CompanyEnum.COMPANY_B)
   
    """Initial empty Predictors for training purposes"""
    """Team Blue Predictors"""
    # Task 1
    TeamBlueStockAPredictor = providers.Factory(TeamBlueStockAPredictor)
    TeamBlueStockBPredictor = providers.Factory(TeamBlueStockBPredictor)
    
    """Team Green Predictors"""
    # Task 1
    TeamGreenStockAPredictor = providers.Factory(TeamGreenStockAPredictor)
    TeamGreenStockBPredictor = providers.Factory(TeamGreenStockBPredictor)
        
    """Team Black Predictors"""
    # Task 1
    TeamBlackStockAPredictor = providers.Factory(TeamBlackStockAPredictor)
    TeamBlackStockBPredictor = providers.Factory(TeamBlackStockBPredictor)
            
    """Team Red Predictors"""
    # Task 1
    TeamRedStockAPredictor = providers.Factory(TeamRedStockAPredictor)
    TeamRedStockBPredictor = providers.Factory(TeamRedStockBPredictor)

 
class Traders(containers.DeclarativeContainer):
    """IoC container of trader providers."""
    
    """Simple Trader"""
    SimpleTrader_with_perfect_prediction = providers.Factory(
        SimpleTrader,
        stock_a_predictor=Predictors.PerfectPredictor_stock_a,
        stock_b_predictor=Predictors.PerfectPredictor_stock_b
        )
    
    SimpleTrader_with_random_prediction = providers.Factory(
        SimpleTrader,
        stock_a_predictor=Predictors.RandomPredictor,
        stock_b_predictor=Predictors.RandomPredictor
        )
    
    """Buy and Hold Trader"""
    BuyAndHoldTrader = providers.Factory(
        BuyAndHoldTrader
        )
   
    """Traders for training purposes"""
    """Team Blue Traders"""
    # Task 0
    TeamBlueSimpleTrader_with_perfect_prediction = providers.Factory(
        TeamBlueSimpleTrader,
        stock_a_predictor=Predictors.PerfectPredictor_stock_a,
        stock_b_predictor=Predictors.PerfectPredictor_stock_b
        )
    
    # Task 1
    SimpleTrader_with_team_blue_prediction = providers.Factory(
        SimpleTrader,
        stock_a_predictor=Predictors.TeamBlueStockAPredictor,
        stock_b_predictor=Predictors.TeamBlueStockBPredictor
        )
    
    """Team Green Traders"""
    # Task 0
    TeamGreenSimpleTrader_with_perfect_prediction = providers.Factory(
        TeamGreenSimpleTrader,
        stock_a_predictor=Predictors.PerfectPredictor_stock_a,
        stock_b_predictor=Predictors.PerfectPredictor_stock_b
        )

    # Task 1
    SimpleTrader_with_team_green_prediction = providers.Factory(
        SimpleTrader,
        stock_a_predictor=Predictors.TeamGreenStockAPredictor,
        stock_b_predictor=Predictors.TeamGreenStockBPredictor
        )
        
    """Team Black Traders"""
    # Task 0
    TeamBlackSimpleTrader_with_perfect_prediction = providers.Factory(
        TeamBlackSimpleTrader,
        stock_a_predictor=Predictors.PerfectPredictor_stock_a,
        stock_b_predictor=Predictors.PerfectPredictor_stock_b
        )
    
    # Task 1
    SimpleTrader_with_team_black_prediction = providers.Factory(
        SimpleTrader,
        stock_a_predictor=Predictors.TeamBlackStockAPredictor,
        stock_b_predictor=Predictors.TeamBlackStockBPredictor
        )
            
    """Team Red Traders"""
    # Task 0
    TeamRedSimpleTrader_with_perfect_prediction = providers.Factory(
        TeamRedSimpleTrader,
        stock_a_predictor=Predictors.PerfectPredictor_stock_a,
        stock_b_predictor=Predictors.PerfectPredictor_stock_b
        )
    
    # Task 1
    SimpleTrader_with_team_red_prediction = providers.Factory(
        SimpleTrader,
        stock_a_predictor=Predictors.TeamRedStockAPredictor,
        stock_b_predictor=Predictors.TeamRedStockBPredictor
        )
    
