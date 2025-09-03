import numpy as np
import pandas as pd
from trading.Position import Position
from typing import List
from performance_metrics import performance_metrics_new


class BaseStrategy:
    name = "BaseStrategy"

    # Флаг ведется ли оптимизация
    isOptimization:bool = False

    bh: list = [] # buy&hold

    # Объект для расчета метрик
    pmn = performance_metrics_new
    
    def __init__(self, start_capital, rel_commission, is_optimization) -> None:
        self.start_capital = start_capital
        self.rel_comission = rel_commission
        self.positions: List[Position] = []
        
        self.is_optimization = is_optimization

        # Объекты для хранения позиций
        self.LastActivePositionLong: Position = None
        self.LastActivePositionShort: Position = None

        # Сигналы стратегии
        self.SignalEntryLong: bool = None
        self.SignalEntryShort: bool = None
        self.SignalExitLong: bool = None
        self.SignalExitShort: bool = None

        # Значения take-profit для long и short позиций
        self.TakeProfitLong: float = 0
        self.TakeProfitShort: float = 0

        # Объект для хранения основных кривых
        self.net_profit: list = None
        self.net_profit_fixed: list = None
    
    def run(self, bars_df, interval=[0, 1]):
        pass

    def CalculateNetProfit(self):
        """
        Считаем Net Profit по истории всех позиций
        """

        data_len = self.bars.shape[0]
        self.net_profit = np.zeros(data_len)

        for pos in self.positions:
            pos_net_profit = np.zeros(data_len)

            end = data_len if pos.IsActive else pos.ExitBarNum

            for bar in range(pos.EntryBarNum, end, 1):
                pos_net_profit[bar] += pos.CurrentProfit(self.Close[bar])
                
            pos_net_profit[bar+1:] += pos.Profit()

            self.net_profit += pos_net_profit

        # FIXED
        self.net_profit_fixed = np.zeros(data_len)
        for pos in self.positions:
            self.net_profit_fixed[pos.ExitBarNum:] += pos.Profit()
    
    def GetActivePositionsForBar(self):

        # LONG
        longActivePositions = [pos for pos in self.positions if pos.IsLong and pos.IsActive]
        self.LastActivePositionLong = longActivePositions[-1] if len(longActivePositions) > 0 else None

        # SHORT
        shortActivePositions = [pos for pos in self.positions if not pos.IsLong and pos.IsActive]
        self.LastActivePositionShort = shortActivePositions[-1] if len(shortActivePositions) > 0 else None

    def CalculateBH(self):
        if self.is_optimization: return
        
        for bar in range(len(self.Close)):
            self.bh.append((self.Close[bar] - self.Close[0])/self.Close[0])

    def savePositionsToCsv(self):
        if self.is_optimization: return

        data = []

        for pos in self.positions:
            data.append({
                "Type": "LONG" if pos.IsLong else "SHORT",
                "Lots": pos.Lots,
                "EntryBarNum": pos.EntryBarNum,
                "ExitBarNum": pos.ExitBarNum,
                "OrderPrice": pos.OrderPrice,
                "ExitLimitPrice": pos.ExitLimitPrice
            })
        
        print("Saving trades...")
        pd.DataFrame(data).to_csv("tmp/trades.csv", index=False)
        