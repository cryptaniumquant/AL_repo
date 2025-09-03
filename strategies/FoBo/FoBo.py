import talib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import List
import time
import sys
sys.path.append("..")
from trading.Position import Position
from strategies.BaseStrategy import BaseStrategy
from trading.Indicators import SMA, ATR
from trading.PosSizers import MaxPctRiskBinance
import os


class FoBo(BaseStrategy):

    # Имя стратегии
    name = "FoBo"

    # Объекты для хранения позиций
    LastActivePositionLong: Position = None
    LastActivePositionShort: Position = None

    # Сигналы стратегии
    SignalEntryLong: bool = None
    SignalEntryShort: bool = None
    SignalExitLong: bool = None
    SignalExitShort: bool = None

    # Значения take-profit для long и short позиций
    TakeProfitLong: float = 0
    TakeProfitShort: float = 0

    # Объект для хранения основных кривых
    net_profit: list = None
    net_profit_fixed: list = None
        
    def params_names():
        return ["atrPeriod", "smaPeriod", "skipValue", "maxPercentRisk"]

    def run(self, params, bars_df, interval=[0, 1], metrics=[]):
        self.positions = []
        # Распаковываем параметры
        self.atrPeriod = params["atrPeriod"]["v"]
        self.smaPeriod = params["smaPeriod"]["v"]
        self.skipValue = params["skipValue"]["v"]
        self.maxPercentRisk = params["maxPercentRisk"]["v"]

        # Вытаскиваем информацию о свечках
        self.bars = bars_df
        
        self.date  = self.bars["Date_dt"].to_numpy()
        self.Open  = self.bars["Open"].to_numpy()
        self.Close = self.bars["Close"].to_numpy()
        self.High  = self.bars["High"].to_numpy()
        self.Low   = self.bars["Low"].to_numpy()

        self.net_profit = np.zeros(self.bars.shape[0])
        self.net_profit_fixed = np.zeros(self.bars.shape[0])

        # ATR
        self.atrSeries = ATR(
            close = self.Close,
            high = self.High,
            low = self.Low,
            period = self.atrPeriod
        )

        # SMA
        self.smaSeries = SMA(x=self.Close, period=self.smaPeriod)
        
        # Interval
        start = max(
            max(self.atrPeriod, self.smaPeriod, self.skipValue),
            int(interval[0] * len(self.Close))
        ) 
        end = int(interval[1] * len(self.Close))
        # print(f"\nINTERVAL: [{start} : {end}/{len(self.Close)}]")
        for bar in range(start, end-1, 1):
            
            # Получаем список активных позиций Long и Short
            self.GetActivePositionsForBar()

            # Генерируем торговые сигналы на вход и выход из позиции и определяем первоначальный стоп
            self.GenerateSignals(bar)

            # Управляем позициями (выставляем ордера на вход и выход из позиции) на основе сгенерированных сигналов
            self.ManagePositions(bar)

        # Считаем net_profit
        self.CalculateNetProfit()
        self.CalculateBH()
        self.saveArrays()
        
        # Считаем метрики
        # Считаем метрики
        Date_np = bars_df['Date_dt'].to_numpy()
        Date_pd = bars_df['Date_dt']
        Date_dt = np.array(Date_pd.dt.to_pydatetime())

        metrics_calc = self.pmn.PerformanceMetrics_new(
            start_capital=self.start_capital,
            Date_np=Date_np,
            Date_pd=Date_pd,
            Date_dt=Date_dt,
            net_profit_punkt_arr=self.net_profit,
            net_profit_punkt_fixed_arr=self.net_profit_fixed,
            trades_count=len(self.positions)
        )

        # Сохраняем метрики
        metrics_ret = {}
        for metric_name in metrics:
            metrics_ret[metric_name] = getattr(metrics_calc, metric_name)
        return metrics_ret

    def GenerateSignals(self, bar):
        
        # Entry LONG
        SignalEntryLong = True
        SignalEntryLong = SignalEntryLong and (self.LastActivePositionLong is None)
        SignalEntryLong = SignalEntryLong and (self.Close[bar] > self.smaSeries[bar])
        SignalEntryLong = SignalEntryLong and (self.atrSeries[bar] > np.mean(self.atrSeries[bar-self.skipValue: bar]))
        self.SignalEntryLong = SignalEntryLong
        
        # Entry SHORT
        SignalEntryShort = True
        SignalEntryShort = SignalEntryShort and (self.LastActivePositionShort is None)
        SignalEntryShort = SignalEntryShort and (self.Close[bar] < self.smaSeries[bar])
        SignalEntryShort = SignalEntryShort and (self.atrSeries[bar] > np.mean(self.atrSeries[bar-self.skipValue: bar]))
        self.SignalEntryShort = SignalEntryShort

        # Exit LONG
        SignalExitLong = False
        if (self.LastActivePositionLong is not None):
            SignalExitLong = SignalExitLong or (self.Close[bar] < (self.smaSeries[bar] - 1.5 * self.atrSeries[bar])) # stoploss
            SignalExitLong = SignalExitLong or (self.Close[bar] > (self.TakeProfitLong + 2.5 * self.atrSeries[bar])) # takeprofit
        self.SignalExitLong = SignalExitLong

        # Exit SHORT
        SignalExitShort = False
        if (self.LastActivePositionShort is not None):
            SignalExitShort = SignalExitShort or (self.Close[bar] > (self.smaSeries[bar] + 1.5 * self.atrSeries[bar]))
            SignalExitShort = SignalExitShort or (self.Close[bar] < (self.TakeProfitShort - 2.5 * self.atrSeries[bar]))
        self.SignalExitShort = SignalExitShort

    def ManagePositions(self, bar):
        # Entry LONG
        if (self.SignalEntryLong): # Проверка сигнала на открытие лонга

            OrderPriceEntryLong = self.Close[bar]
            StopPriceLong = self.smaSeries[bar] - 1.5 * self.atrSeries[bar]

            # Открытие длинной позиции по цене OrderPriceEntryLong на следующем баре
            OrderLotsLong = MaxPctRiskBinance(
                SummForSystem    = self.start_capital,
                maxPctRisk       = self.maxPercentRisk,
                TargetEntryPrice = OrderPriceEntryLong,
                StartStopLoss    = StopPriceLong,
                minLotSizeCrypta = 0.01
            )

            # Open LONG position
            self.positions.append(Position(
                IsActive=True,
                IsLong=True,
                EntryBarNum=bar+1,
                Lots=OrderLotsLong,
                OrderPrice=OrderPriceEntryLong,
                EntryDate=self.date[bar+1]
            ))
            # print(f"Open LONG {bar+1}, {OrderLotsLong}, {OrderPriceEntryLong}")
            self.TakeProfitLong = OrderPriceEntryLong


        # Entry Short
        if (self.SignalEntryShort): # Проверка сигнала на открытие лонга
            
            # Открытие short позиции по цене OrderPriceEntryShort на следующем баре
            OrderPriceEntryShort = self.Close[bar]
            StopPriceShort = self.smaSeries[bar] + 1.5 * self.atrSeries[bar]

            OrderLotsShort = MaxPctRiskBinance(
                SummForSystem    = self.start_capital,
                maxPctRisk       = self.maxPercentRisk,
                TargetEntryPrice = OrderPriceEntryShort,
                StartStopLoss    = StopPriceShort,
                minLotSizeCrypta = 0.01,

            )

            # Open SHORT position
            self.positions.append(Position(
                IsActive=True,
                IsLong=False,
                EntryBarNum=bar+1,
                Lots=OrderLotsShort,
                OrderPrice=OrderPriceEntryShort,
                EntryDate=self.date[bar+1]
            ))
            # print(f"Open SHORT {bar+1}, {OrderLotsShort}, {OrderPriceEntryShort}")
            self.TakeProfitShort = OrderPriceEntryShort
        

        # Exit LONG
        if (self.SignalExitLong): # Проверка сигнала на закрытие лонга
            # Закрытие позиции на следующем баре
            self.LastActivePositionLong.CloseAtPrice(
                bar + 1,
                self.Close[bar],
                exitDate=self.date[bar+1])

        # Exit SHORT
        if (self.SignalExitShort): # Проверка сигнала на закрытие лонга
            # Закрытие позиции на следующем баре
            self.LastActivePositionShort.CloseAtPrice(
                bar + 1,
                self.Close[bar],
                exitDate=self.date[bar+1])

    def saveArrays(self):
        if self.is_optimization: return

        # Создаем папку ./tmp
        if not os.path.isdir("./tmp"):
            os.mkdir("./tmp")

        # Сохраняем в неё все файлы
        print("Saving tmp files into ./tmp")
        np.save("tmp/sma.npy", self.smaSeries)
        
        np.save("tmp/atr.npy", self.atrSeries)

        np.save("tmp/net_profit.npy", self.net_profit)
        np.save("tmp/net_profit_fixed.npy", self.net_profit_fixed)
        np.save("tmp/date.npy", self.date)
        np.save("tmp/bh.npy", self.bh)

        self.savePositionsToCsv()
