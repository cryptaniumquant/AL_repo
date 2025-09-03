import numpy as np
from strategies.BaseStrategy import BaseStrategy
from trading.Indicators import VWAP_indicator
from trading.PosSizers import MaxPctRiskBinance
from trading.Position import Position


class VWAP(BaseStrategy):
    # Импульсный пробой объема
    # Имя стратегии
    name = "VWAP"

    # TakeProfit & StopLoss
    TakeProfitLong = 0
    TakeProfitShort = 0

    StopLossLong = 0
    StopLossShort = 0

    def params_names():
        return ["N", "vwapPeriod", "maxPercentRisk"]

    def run(self, params, bars_df, interval=[0, 1], metrics=[]):
        self.positions = []

        # Распаковываем параметры
        self.N = params["N"]["v"]
        self.vwapPeriod = params["vwapPeriod"]["v"]
        self.maxPercentRisk = params["maxPercentRisk"]["v"]

        # Вытаскиваем информацию о свечках
        self.bars = bars_df
        
        self.date   = self.bars["Date_dt"].to_numpy()
        self.Open   = self.bars["Open"].to_numpy()
        self.Close  = self.bars["Close"].to_numpy()
        self.High   = self.bars["High"].to_numpy()
        self.Low    = self.bars["Low"].to_numpy()
        self.Volume = self.bars["Volume"].to_numpy()

        self.net_profit = np.zeros(self.bars.shape[0])
        self.net_profit_fixed = np.zeros(self.bars.shape[0])

        # VWPA
        self.vwapSeries = VWAP_indicator(
            high   = self.High,
            low    = self.Low,
            close  = self.Close,
            volume = self.Volume,
            window = self.vwapPeriod
        )
        
        # Interval
        start = int(interval[0] * len(self.Close)) + max(self.vwapPeriod, self.N)
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
        SignalEntryLong = SignalEntryLong and (self.Volume[bar] == np.max(self.Volume[bar-self.N: bar+1]))
        SignalEntryLong = SignalEntryLong and (self.Close[bar] > self.Open[bar])
        SignalEntryLong = SignalEntryLong and (self.Close[bar] > self.vwapSeries[bar])
        self.SignalEntryLong = SignalEntryLong
        
        # Entry SHORT
        SignalEntryShort = True
        SignalEntryShort = SignalEntryShort and (self.LastActivePositionShort is None)
        SignalEntryShort = SignalEntryShort and (self.Volume[bar] == np.max(self.Volume[bar-self.N: bar+1]))
        SignalEntryShort = SignalEntryShort and (self.Close[bar] < self.Open[bar])
        SignalEntryShort = SignalEntryShort and (self.Close[bar] < self.vwapSeries[bar])
        self.SignalEntryShort = SignalEntryShort

        # Exit LONG
        SignalExitLong = False
        if (self.LastActivePositionLong is not None):
            SignalExitLong = SignalExitLong or (self.Close[bar] < self.StopLossLong) # stoploss
            SignalExitLong = SignalExitLong or (self.Close[bar] > self.TakeProfitLong) # takeprofit
        self.SignalExitLong = SignalExitLong

        # Exit SHORT
        SignalExitShort = False
        if (self.LastActivePositionShort is not None):
            SignalExitShort = SignalExitShort or (self.Close[bar] > self.StopLossShort)
            SignalExitShort = SignalExitShort or (self.Close[bar] < self.TakeProfitShort)
        self.SignalExitShort = SignalExitShort

    def ManagePositions(self, bar):
        # Entry LONG
        if (self.SignalEntryLong): # Проверка сигнала на открытие лонга

            self.TakeProfitLong = self.Close[bar] * (1 + self.maxPercentRisk/100)
            self.StopLossLong = self.Close[bar] * 0.98

            # Открытие длинной позиции по цене OrderPriceEntryLong на следующем баре
            OrderLotsLong = MaxPctRiskBinance(
                SummForSystem    = self.start_capital,
                maxPctRisk       = self.maxPercentRisk,
                TargetEntryPrice = self.Close[bar],
                StartStopLoss    = self.StopLossLong,
                minLotSizeCrypta = 0.01
            )

            # Open LONG position
            self.positions.append(Position(
                IsActive    = True,
                IsLong      = True,
                EntryBarNum = bar+1,
                Lots        = OrderLotsLong,
                OrderPrice  = self.Close[bar],
                EntryDate   = self.date[bar+1]
            ))
            # print(f"Open LONG {bar+1}, {OrderLotsLong}, {OrderPriceEntryLong}")
            

        # Entry Short
        if (self.SignalEntryShort): # Проверка сигнала на открытие лонга
            
            # Открытие short позиции по цене OrderPriceEntryShort на следующем баре
            self.TakeProfitShort = self.Close[bar] * (1 - self.maxPercentRisk/100)
            self.StopLossShort = self.Close[bar] * 1.02
            

            OrderLotsShort = MaxPctRiskBinance(
                SummForSystem    = self.start_capital,
                maxPctRisk       = self.maxPercentRisk,
                TargetEntryPrice = self.Close[bar],
                StartStopLoss    = self.StopLossShort,
                minLotSizeCrypta = 0.01,

            )

            # Open SHORT position
            self.positions.append(Position(
                IsActive    = True,
                IsLong      = False,
                EntryBarNum = bar+1,
                Lots        = OrderLotsShort,
                OrderPrice  = self.Close[bar],
                EntryDate   = self.date[bar+1]
            ))
            # print(f"Open SHORT {bar+1}, {OrderLotsShort}, {OrderPriceEntryShort}")
        

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
        np.save("tmp/vwap.npy", self.vwapSeries)
        
        np.save("tmp/net_profit.npy", self.net_profit)
        np.save("tmp/net_profit_fixed.npy", self.net_profit_fixed)
        np.save("tmp/date.npy", self.date)
        np.save("tmp/bh.npy", self.bh)

        self.savePositionsToCsv()
