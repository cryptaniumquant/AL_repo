from strategies.BaseStrategy import BaseStrategy
import numpy as np
from trading.Indicators import ATR, EMA
from trading.Position import Position
from trading.PosSizers import MaxPctRiskBinance


class OneP(BaseStrategy):

    # Имя стратегии
    name = "OneP"
    
    # Цены входа в позицию
    EntryPriceLong  = 0
    EntryPriceShort = 0

    # Цены для выхода из позиции
    StopPriceLong = 0
    StopPriceShort = 0

    def params_names():
        return ["multiplier", "maxPercentRisk"]

    def run(self, params, bars_df, interval=[0, 1], metrics=[]):
        self.positions = []

        # Распаковываем параметры
        self.multiplier = params["multiplier"]["v"]
        self.maxPercentRisk = params["maxPercentRisk"]["v"]
        self.atrPeriod = 20
        self.emaPeriod = 100

        # Вытаскиваем информацию о свечках
        self.bars = bars_df
        
        self.date = self.bars["Date_dt"].to_numpy()
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

        # EMA
        self.emaSeries = EMA(x=self.Close, period=self.emaPeriod)
        
        self.upper_band = self.emaSeries + (self.multiplier * self.atrSeries)
        self.lower_band = self.emaSeries - (self.multiplier * self.atrSeries)

        # Interval
        start = int(interval[0] * len(self.Close)) + max(self.atrPeriod, self.emaPeriod)
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
        SignalEntryLong = SignalEntryLong and (self.Close[bar] > self.emaSeries[bar])
        SignalEntryLong = SignalEntryLong and (self.Low[bar] < self.lower_band[bar])
        self.SignalEntryLong = SignalEntryLong
        
        # Entry SHORT
        SignalEntryShort = True
        SignalEntryShort = SignalEntryShort and (self.LastActivePositionShort is None)
        SignalEntryShort = SignalEntryShort and (self.Close[bar] < self.emaSeries[bar])
        SignalEntryShort = SignalEntryShort and (self.High[bar] > self.upper_band[bar])
        self.SignalEntryShort = SignalEntryShort

        # Exit LONG
        SignalExitLong = False
        if (self.LastActivePositionLong is not None):
            SignalExitLong = SignalExitLong or (self.Close[bar] < (self.EntryPriceLong - 1.5 * self.atrSeries[bar])) # stoploss
            SignalExitLong = SignalExitLong or (self.Close[bar] > (self.EntryPriceLong + 3.0 * self.atrSeries[bar])) # takeprofit
        self.SignalExitLong = SignalExitLong

        # Exit SHORT
        SignalExitShort = False
        if (self.LastActivePositionShort is not None):
            SignalExitShort = SignalExitShort or (self.Close[bar] > (self.EntryPriceShort + 1.5 * self.atrSeries[bar])) # stoploss
            SignalExitShort = SignalExitShort or (self.Close[bar] < (self.EntryPriceShort - 3.0 * self.atrSeries[bar])) # takeprofit
        self.SignalExitShort = SignalExitShort

    def ManagePositions(self, bar):
        # Entry LONG
        if (self.SignalEntryLong): # Проверка сигнала на открытие лонга

            self.EntryPriceLong = self.Close[bar]

            # Открытие long позиции по цене OrderPriceEntryLong на следующем баре
            OrderLotsLong = MaxPctRiskBinance(
                SummForSystem    = self.start_capital,
                maxPctRisk       = self.maxPercentRisk,
                TargetEntryPrice = self.Close[bar],
                StartStopLoss    = self.EntryPriceLong - 1.5 * self.atrSeries[bar],
                minLotSizeCrypta = 0.01
            )

            # Open LONG position
            self.positions.append(Position(
                IsActive=True,
                IsLong=True,
                EntryBarNum=bar+1,
                Lots=OrderLotsLong,
                OrderPrice=self.Close[bar],
                EntryDate=self.date[bar+1]
            ))


        # Entry Short
        if (self.SignalEntryShort): # Проверка сигнала на открытие лонга
            
            self.EntryPriceShort = self.Close[bar]

            # Открытие short позиции по цене OrderPriceEntryShort на следующем баре
            OrderLotsShort = MaxPctRiskBinance(
                SummForSystem    = self.start_capital,
                maxPctRisk       = self.maxPercentRisk,
                TargetEntryPrice = self.Close[bar],
                StartStopLoss    = self.Close[bar] + 1.5 * self.atrSeries[bar],
                minLotSizeCrypta = 0.01,

            )

            # Open SHORT position
            self.positions.append(Position(
                IsActive=True,
                IsLong=False,
                EntryBarNum=bar+1,
                Lots=OrderLotsShort,
                OrderPrice=self.Close[bar],
                EntryDate=self.date[bar+1]
            ))
        

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

        print("Saving tmp/net_profit.npy")
        np.save("tmp/ema.npy", self.emaSeries)
        np.save("tmp/atr.npy", self.atrSeries)

        np.save("tmp/net_profit.npy", self.net_profit)
        np.save("tmp/net_profit_fixed.npy", self.net_profit_fixed)
        np.save("tmp/date.npy", self.date)
        np.save("tmp/bh.npy", self.bh)

        self.savePositionsToCsv()
