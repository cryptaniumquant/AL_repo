import numpy as np
import pandas as pd
import talib
from datetime import datetime
from dataclasses import dataclass
from tqdm import tqdm
import matplotlib.pyplot as plt
import inspect
import warnings

# Импорт пользовательских модулей
from tradingMahtematics import indicators, tradingTactics, array_operations
from performance_metrics import performance_metrics_new, strategy_charts

def bb_trend(Bars_df, _symbol, _init_deposit, _maxPctRisk, _pctOfReinvest, *_params_to_optimize):

    #region подготовка
    symbol = _symbol
    position = []
    entry_signal = []
    entry_bar = []
    lots = []
    entry_price = []
    entry_date = []
    exit_signal = []
    exit_bar = []
    exit_price = []
    exit_date = []
    min_step_lot_symbol = 0.00001
    min_lot_size_usd = 0.00
    step_price_symbol = 0.01

    symbol_params_dict = {
        'BtcUsdt': {'min_step_lot_symbol': 0.00001, 'min_lot_size_usd': 0.00, 'step_price_symbol': 0.01},
        'EthUsdt': {'min_step_lot_symbol': 0.0001, 'min_lot_size_usd': 0.00, 'step_price_symbol': 0.01},
        'BnbUsdt': {'min_step_lot_symbol': 0.001, 'min_lot_size_usd': 0.00, 'step_price_symbol': 0.1},
        'XrpUsdt': {'min_step_lot_symbol': 1.00, 'min_lot_size_usd': 0.00, 'step_price_symbol': 0.0001},
        'DogeUsdt': {'min_step_lot_symbol': 1.00, 'min_lot_size_usd': 0.00, 'step_price_symbol': 0.00001},
        'AdaUsdt': {'min_step_lot_symbol': 0.1, 'min_lot_size_usd': 0.00, 'step_price_symbol': 0.0001},
        'SolUsdt': {'min_step_lot_symbol': 0.001, 'min_lot_size_usd': 0.00, 'step_price_symbol': 0.01},
        'TrxUsdt': {'min_step_lot_symbol': 0.1, 'min_lot_size_usd': 0.00, 'step_price_symbol': 0.00001},
        'ImxUsdt': {'min_step_lot_symbol': 0.1, 'min_lot_size_usd': 0.00, 'step_price_symbol': 0.00001},
        'AvaxUsdt': {'min_step_lot_symbol': 0.1, 'min_lot_size_usd': 0.00, 'step_price_symbol': 0.00001},
        'HbarUsdt': {'min_step_lot_symbol': 0.1, 'min_lot_size_usd': 0.00, 'step_price_symbol': 0.00001}
    }

    def get_symbol_params(symbol):
        return symbol_params_dict.get(symbol, None)

    symbol_params = get_symbol_params(symbol)
    if symbol_params:
        min_step_lot_symbol = symbol_params['min_step_lot_symbol']
        min_lot_size_usd = symbol_params['min_lot_size_usd']
        step_price_symbol = symbol_params['step_price_symbol']
    else:
        raise KeyError(f"Параметры для инструмента '{symbol}' не найдены в словаре symbol_params_dict.")

    strategyName = inspect.currentframe().f_code.co_name
    write_to_log = False
    isOptimization = False
    isRealTime = False
    isExecuteOn = True
    if not isExecuteOn:
        return {'Работа прервана': 0}

    @dataclass
    class LongEntryOrder:
        EntryLimitPrice: float = 0.0
        EntryBarNum: int = 0
        LotsInOrder: float = 0.0

    @dataclass
    class LongExitOrder:
        ExitLimitPrice: float = 0.0
        ExitBarNum: int = 0
        LotsInOrder: float = 0.0

    @dataclass
    class ShortEntryOrder:
        EntryLimitPrice: float = 0.0
        EntryBarNum: int = 0
        LotsInOrder: float = 0.0

    @dataclass
    class ShortExitOrder:
        ExitLimitPrice: float = 0.0
        ExitBarNum: int = 0
        LotsInOrder: float = 0.0

    @dataclass
    class LastActivePosition:
        IsNull: bool = True
        IsLong: bool = True
        PosSizeLotShares: float = 0.00
        AverageEntryPrice: float = 0.00
        EntryBarNum: int = 0
        ExitBarNum: int = 0

        def CloseAtPrice(self, bar, exit_limit_price, lots, isLong, orderText):
            if isLong:
                LongExitOrder.ExitLimitPrice = exit_limit_price
                LongExitOrder.ExitBarNum = bar
                LongExitOrder.LotsInOrder = lots
            else:
                ShortExitOrder.ExitLimitPrice = exit_limit_price
                ShortExitOrder.ExitBarNum = bar
                ShortExitOrder.LotsInOrder = lots

    class OrdersOperations:
        @staticmethod
        def BuyAtPrice(order, bar, lots, orderEntryPrice, orderText):
            order.EntryBarNum = bar
            order.LotsInOrder = lots
            order.EntryLimitPrice = orderEntryPrice

        @staticmethod
        def SellAtPrice(order, bar, lots, orderEntryPrice, orderText):
            order.EntryLimitPrice = orderEntryPrice
            order.EntryBarNum = bar
            order.LotsInOrder = lots

    bar_number = Bars_df.index.to_numpy()
    Open_np = Bars_df['Open'].to_numpy()
    High_np = Bars_df['High'].to_numpy()
    Low_np = Bars_df['Low'].to_numpy()
    Close_np = Bars_df['Close'].to_numpy()
    Date_np = Bars_df['Date_dt'].to_numpy()
    Date_pd = Bars_df['Date_dt']
    with warnings.catch_warnings(action="ignore"):
        Date_dt = np.array(Date_pd.dt.to_pydatetime())

    bars_count = len(Bars_df)
    AbsComission_arr = np.zeros(bars_count, dtype=np.float64)
    RelComission_arr = np.zeros(bars_count, dtype=np.float64)
    net_profit_fixed_arr = np.zeros(bars_count, dtype=np.float64)
    net_profit_arr = np.zeros(bars_count, dtype=np.float64)
    positions_arr = np.array([], dtype=np.float64)

    first_valid_value = 0
    isLong = False
    signalBuy = False
    orderEntryLong = 0.00
    stopPriceLong = 0.00
    trailing_stop_long = 0.00
    kontraktLong_mPR = 0.00
    entryPriceLong = 0.00
    isShort = False
    signalShort = False
    orderEntryShort = 0.00
    stopPriceShort = 0.00
    trailing_stop_short = 0.00
    kontraktShort_mPR = 0.00
    exitLimitPrice = 0.00
    lots_in_order = 0.00
    ActiveOrderPrice = 0.00
    LotsInPosition = 0.00

    LongEntryOrder = LongEntryOrder()
    LongExitOrder = LongExitOrder()
    ShortEntryOrder = ShortEntryOrder()
    ShortExitOrder = ShortExitOrder()
    LastActivePosition = LastActivePosition()

    _bbPeriod, _bbStDev, _bbSplitCount, _maxBandsToTrailing = _params_to_optimize
    _bbPeriod = int(_bbPeriod)
    _bbStDev = float(_bbStDev)
    _bbSplitCount = int(_bbSplitCount)
    _maxBandsToTrailing = float(_maxBandsToTrailing)
    init_deposit = float(_init_deposit)
    max_pct_risk = float(_maxPctRisk)
    pct_of_reinvest = float(_pctOfReinvest)

    use_abs_comission = False
    abs_comission = 0.00
    rel_comission = 0.05
    max_percent_risk = float(_maxPctRisk)
    fin_res_reinvest_pct = float(_pctOfReinvest)
    is_fixed_mode = False
    max_count_of_min_lot_steps = 10000000000
    trade_min_lot_size = False

    finres_for_bar = 0.00
    money_for_trading_system = 0.00

    #endregion

    #region indicators
    Bars_df['upBollinger'], Bars_df['midBollinger'], Bars_df['dnBollinger'] = talib.BBANDS(
        real=Bars_df['Close'],
        timeperiod=_bbPeriod,
        nbdevup=_bbStDev,
        nbdevdn=_bbStDev,
        matype=0
    )

    first_valid_value = max(first_valid_value, _bbPeriod)
    upperBand = Bars_df['upBollinger'].to_numpy()
    midBand = Bars_df['midBollinger'].to_numpy()
    lowerBand = Bars_df['dnBollinger'].to_numpy()
    trailing_for_long_np = np.full(bars_count, np.nan, dtype=np.float64)
    trailing_for_short_np = np.full(bars_count, np.nan, dtype=np.float64)

    first_date = Date_dt[0]
    first_date_string = first_date.strftime("%d-%m-%Y")
    last_date = Date_dt[-1]
    last_date_string = last_date.strftime("%d-%m-%Y")
    #endregion

    #region main buy/sell cycle
    with tqdm(total=len(range(int(first_valid_value), bars_count)), desc="Processing", unit="iteration") as pbar:
        for bar in range(int(first_valid_value), bars_count):
            if LastActivePosition.IsNull:
                if max(LongEntryOrder.EntryBarNum, ShortEntryOrder.EntryBarNum) == bar:
                    pass
                if LongEntryOrder.EntryBarNum == bar and Low_np[bar] < LongEntryOrder.EntryLimitPrice:
                    if write_to_log:
                        print(f"BuyLimit at bar:\t\t {bar},\t price:\t {LongEntryOrder.EntryLimitPrice},\t Open[bar]\t = {Open_np[bar]}")
                    LastActivePosition.IsNull = False
                    LastActivePosition.IsLong = True
                    LastActivePosition.PosSizeLotShares = LongEntryOrder.LotsInOrder
                    LastActivePosition.AverageEntryPrice = LongEntryOrder.EntryLimitPrice
                    LastActivePosition.EntryBarNum = bar
                    position.append('Long')
                    entry_signal.append('LongEnter')
                    entry_bar.append(bar)
                    lots.append(LongEntryOrder.LotsInOrder)
                    entry_price.append(LongEntryOrder.EntryLimitPrice)
                    entry_date.append(Date_dt[bar])
                if ShortEntryOrder.EntryBarNum == bar and High_np[bar] > ShortEntryOrder.EntryLimitPrice:
                    if write_to_log:
                        print(f"SellLimit at bar:\t\t {bar},\t price:\t {ShortEntryOrder.EntryLimitPrice},\t Open[bar]\t = {Open_np[bar]}")
                    LastActivePosition.IsNull = False
                    LastActivePosition.IsLong = False
                    LastActivePosition.PosSizeLotShares = ShortEntryOrder.LotsInOrder
                    LastActivePosition.AverageEntryPrice = ShortEntryOrder.EntryLimitPrice
                    LastActivePosition.EntryBarNum = bar
                    position.append('Short')
                    entry_signal.append('ShortEnter')
                    entry_bar.append(bar)
                    lots.append(-1.0 * ShortEntryOrder.LotsInOrder)
                    entry_price.append(ShortEntryOrder.EntryLimitPrice)
                    entry_date.append(Date_dt[bar])
            elif not LastActivePosition.IsNull:
                if LongExitOrder.ExitBarNum == bar and High_np[bar] > LongExitOrder.ExitLimitPrice:
                    if write_to_log:
                        print(f"ExitLongLimit at bar:\t {bar},\t price:\t {LongExitOrder.ExitLimitPrice},\t Open[bar]\t = {Open_np[bar]}")
                    LastActivePosition.IsNull = True
                    LastActivePosition.ExitBarNum = bar
                    exit_signal.append('ExitLong')
                    exit_bar.append(bar)
                    exit_price.append(LongExitOrder.ExitLimitPrice)
                    exit_date.append(Date_dt[bar])
                if ShortExitOrder.ExitBarNum == bar and Low_np[bar] < ShortExitOrder.ExitLimitPrice:
                    if write_to_log:
                        print(f"ExitShortLimit at bar:\t {bar},\t price:\t {ShortExitOrder.ExitLimitPrice},\t Open[bar]\t = {Open_np[bar]}")
                    LastActivePosition.IsNull = True
                    LastActivePosition.ExitBarNum = bar
                    exit_signal.append('ExitShort')
                    exit_bar.append(bar)
                    exit_price.append(ShortExitOrder.ExitLimitPrice)
                    exit_date.append(Date_dt[bar])

            currentProfit = 0.00
            if LastActivePosition.IsNull:
                if LongExitOrder.ExitBarNum == bar:
                    currentProfit = tradingTactics.position_profit_calc(True, LastActivePosition.AverageEntryPrice, LongExitOrder.ExitLimitPrice, LastActivePosition.PosSizeLotShares, rel_comission)
                elif ShortExitOrder.ExitBarNum == bar:
                    currentProfit = tradingTactics.position_profit_calc(False, LastActivePosition.AverageEntryPrice, ShortExitOrder.ExitLimitPrice, LastActivePosition.PosSizeLotShares, rel_comission)
                else:
                    currentProfit = 0
                net_profit_fixed_arr[bar] = net_profit_fixed_arr[bar - 1] + currentProfit
                net_profit_arr[bar] = net_profit_fixed_arr[bar]
                if LongExitOrder.ExitBarNum == bar or ShortExitOrder.ExitBarNum == bar:
                    positions_arr = np.append(positions_arr, net_profit_fixed_arr[bar])
            elif not LastActivePosition.IsNull:
                net_profit_fixed_arr[bar] = net_profit_fixed_arr[bar - 1]
                if LastActivePosition.IsLong:
                    currentProfit = tradingTactics.position_profit_calc(True, LastActivePosition.AverageEntryPrice, Close_np[bar], LastActivePosition.PosSizeLotShares, rel_comission)
                else:
                    currentProfit = tradingTactics.position_profit_calc(False, LastActivePosition.AverageEntryPrice, Close_np[bar], LastActivePosition.PosSizeLotShares, rel_comission)
                net_profit_arr[bar] = net_profit_fixed_arr[bar] + currentProfit

            if LastActivePosition.IsNull:
                LotsInPosition = 0.00
                EntryPrice = float('nan')
            else:
                if LastActivePosition.IsLong:
                    LotsInPosition = LastActivePosition.PosSizeLotShares
                else:
                    LotsInPosition = -1 * LastActivePosition.PosSizeLotShares
                EntryPrice = LastActivePosition.AverageEntryPrice

            orderEntryLong = Close_np[bar]
            orderEntryLong = tradingTactics.RoundPrice(orderEntryLong, step_price_symbol)
            orderEntryShort = Close_np[bar]
            orderEntryShort = tradingTactics.RoundPrice(orderEntryShort, step_price_symbol)
            exitLimitPrice = Close_np[bar]
            exitLimitPrice = tradingTactics.RoundPrice(exitLimitPrice, step_price_symbol)

            signalBuy = (Close_np[bar] > upperBand[bar])
            signalShort = (Close_np[bar] < lowerBand[bar])

            if LastActivePosition.IsNull:
                if signalBuy:
                    stop_price_long = midBand[bar]
                    stop_price_long = tradingTactics.RoundPrice(stop_price_long, step_price_symbol)
                    trailing_stop_long = lowerBand[bar]
                    trailing_for_long_np[bar] = lowerBand[bar]
                    finres_for_bar = net_profit_fixed_arr[bar] * fin_res_reinvest_pct / 100.0
                    money_for_trading_system = init_deposit + finres_for_bar
                    kontraktLong_mPR = tradingTactics.MaxPctRiskBinance(money_for_trading_system, max_percent_risk, orderEntryLong, stop_price_long, min_step_lot_symbol, min_lot_size_usd, max_count_of_min_lot_steps, trade_min_lot_size)
                    kontraktLong_mPR = tradingTactics.RoundToMinLotStep(kontraktLong_mPR, min_step_lot_symbol)
                    if (kontraktLong_mPR >= min_step_lot_symbol) and (kontraktLong_mPR * orderEntryLong > min_lot_size_usd):
                        orderText = "LongEnter"
                        lots_in_order = kontraktLong_mPR
                        ActiveOrderPrice = orderEntryLong
                        OrdersOperations.BuyAtPrice(LongEntryOrder, bar + 1, kontraktLong_mPR, orderEntryLong, orderText)
                    else:
                        kontraktLong_mPR = tradingTactics.MinPosSizeBinanceSpot(orderEntryLong, min_step_lot_symbol, min_lot_size_usd)
                        kontraktLong_mPR = tradingTactics.RoundToMinLotStep(kontraktLong_mPR, min_step_lot_symbol)
                        kontraktLong_mPR = min(kontraktLong_mPR, max_count_of_min_lot_steps * min_step_lot_symbol)
                        orderText = "LongEnter_minKontrakt"
                        lots_in_order = kontraktLong_mPR
                        ActiveOrderPrice = orderEntryLong
                        OrdersOperations.BuyAtPrice(LongEntryOrder, bar + 1, kontraktLong_mPR, orderEntryLong, orderText)
                elif signalShort:
                    stop_price_short = midBand[bar]
                    stop_price_short = tradingTactics.RoundPrice(stop_price_short, step_price_symbol)
                    trailing_stop_short = upperBand[bar]
                    trailing_for_short_np[bar] = upperBand[bar]
                    finres_for_bar = net_profit_fixed_arr[bar] * fin_res_reinvest_pct / 100.0
                    money_for_trading_system = init_deposit + finres_for_bar
                    kontraktShort_mPR = tradingTactics.MaxPctRiskBinance(money_for_trading_system, max_percent_risk, orderEntryShort, stop_price_short, min_step_lot_symbol, min_lot_size_usd, max_count_of_min_lot_steps, trade_min_lot_size)
                    kontraktShort_mPR = tradingTactics.RoundToMinLotStep(kontraktShort_mPR, min_step_lot_symbol)
                    if (kontraktShort_mPR >= min_step_lot_symbol) and (kontraktShort_mPR * orderEntryShort > min_lot_size_usd):
                        orderText = "ShortEnter"
                        lots_in_order = -1.0 * kontraktShort_mPR
                        ActiveOrderPrice = orderEntryShort
                        OrdersOperations.SellAtPrice(ShortEntryOrder, bar + 1, kontraktShort_mPR, orderEntryShort, orderText)
                    else:
                        kontraktShort_mPR = tradingTactics.MinPosSizeBinanceSpot(orderEntryShort, min_step_lot_symbol, min_lot_size_usd)
                        kontraktShort_mPR = tradingTactics.RoundToMinLotStep(kontraktShort_mPR, min_step_lot_symbol)
                        orderText = "ShortEnter_minKontrakt"
                        lots_in_order = -1.0 * kontraktShort_mPR
                        ActiveOrderPrice = orderEntryShort
                        OrdersOperations.SellAtPrice(ShortEntryOrder, bar + 1, kontraktShort_mPR, orderEntryShort, orderText)
            elif not LastActivePosition.IsNull:
                if LastActivePosition.IsLong and LastActivePosition.EntryBarNum != bar:
                    if bar == LastActivePosition.EntryBarNum:
                        prevBar = LastActivePosition.EntryBarNum - 1
                        trailing_stop_long = upperBand[prevBar]
                        trailing_stop_long = tradingTactics.RoundPrice(trailing_stop_long, step_price_symbol)
                    else:
                        bbRangePoints = max(Bars_df['upBollinger'].iloc[bar] - Bars_df['dnBollinger'].iloc[bar], Bars_df.loc[bar, 'High'] - Bars_df['dnBollinger'].iloc[bar], Bars_df['upBollinger'].iloc[bar] - Bars_df.loc[bar, 'Low'])
                        trailing_stop_long = max(trailing_stop_long, trailing_stop_long + bbRangePoints / _bbSplitCount)
                        trailing_stop_long = max(trailing_stop_long, upperBand[bar] - _maxBandsToTrailing * (upperBand[bar] - lowerBand[bar]))
                        trailing_stop_long = tradingTactics.RoundPrice(trailing_stop_long, step_price_symbol)
                    trailing_for_long_np[bar] = trailing_stop_long
                    ExitLong = Close_np[bar] < trailing_stop_long
                    if ExitLong:
                        lots_in_order = -1.0 * LotsInPosition
                        ActiveOrderPrice = exitLimitPrice
                        LastActivePosition.CloseAtPrice(bar + 1, exitLimitPrice, lots_in_order, True, "Exit Long")
                elif not LastActivePosition.IsLong and LastActivePosition.EntryBarNum != bar:
                    if bar == LastActivePosition.EntryBarNum:
                        prevBar = LastActivePosition.EntryBarNum - 1
                        trailing_stop_short = upperBand[prevBar]
                        trailing_stop_short = tradingTactics.RoundPrice(trailing_stop_short, step_price_symbol)
                    else:
                        bbRangePoints = max(Bars_df['upBollinger'].iloc[bar] - Bars_df['dnBollinger'].iloc[bar], Bars_df.loc[bar, 'High'] - Bars_df['dnBollinger'].iloc[bar], Bars_df['upBollinger'].iloc[bar] - Bars_df.loc[bar, 'Low'])
                        trailing_stop_short = min(trailing_stop_short, trailing_stop_short - bbRangePoints / _bbSplitCount)
                        trailing_stop_short = min(trailing_stop_short, lowerBand[bar] + _maxBandsToTrailing * (upperBand[bar] - lowerBand[bar]))
                        trailing_stop_short = tradingTactics.RoundPrice(trailing_stop_short, step_price_symbol)
                    trailing_for_short_np[bar] = trailing_stop_short
                    ExitShort = Close_np[bar] > trailing_stop_short
                    if ExitShort:
                        lots_in_order = -1.0 * LotsInPosition
                        ActiveOrderPrice = exitLimitPrice
                        LastActivePosition.CloseAtPrice(bar + 1, exitLimitPrice, lots_in_order, False, "Exit Short")
            pbar.update(1)
    #endregion

    max_length = max(len(position), len(entry_signal), len(entry_bar), len(lots), len(entry_price), len(entry_date), len(exit_signal), len(exit_bar), len(exit_price), len(exit_date))
    position.extend([None] * (max_length - len(position)))
    entry_signal.extend([None] * (max_length - len(entry_signal)))
    entry_bar.extend([None] * (max_length - len(entry_bar)))
    lots.extend([None] * (max_length - len(lots)))
    entry_price.extend([None] * (max_length - len(entry_price)))
    entry_date.extend([None] * (max_length - len(entry_date)))
    exit_signal.extend([None] * (max_length - len(exit_signal)))
    exit_bar.extend([None] * (max_length - len(exit_bar)))
    exit_price.extend([None] * (max_length - len(exit_price)))
    exit_date.extend([None] * (max_length - len(exit_date)))

    series_dict = {
        'Open_np': Open_np,
        'High_np': High_np,
        'Low_np': Low_np,
        'Close_np': Close_np,
        'Date_np': Date_np,
        'Date_pd': Date_pd,
        'Date_dt': Date_dt,
        'net_profit_arr': net_profit_arr,
        'net_profit_fixed_arr': net_profit_fixed_arr,
        'fluger_np': upperBand,
        'lowerBand': lowerBand,
        'trailing_for_long_np': trailing_for_long_np,
        'trailing_for_short_np': trailing_for_short_np
    }

    positions_dict = {
        'Position': position,
        'Symbol': ([f'{_symbol}'] * len(position)),
        'Lots': lots,
        'Entry Signal': entry_signal,
        'Entry Bar': entry_bar,
        'Entry Price': entry_price,
        'Entry Date': entry_date,
        'Exit Bar': exit_bar,
        'Exit Price': exit_price,
        'Exit Date': exit_date,
        'Exit Signal': exit_signal
    }

    return {'series_dict': series_dict, 'positions_dict': positions_dict}

def resample_candles(df, timeframe):
    df.index = pd.to_datetime(df.Date)
    return df.resample(timeframe).agg({
        'Open': 'first',
        'Close': 'last',
        'High': 'max',
        'Low': 'min',
        'Volume': 'sum'
    }).dropna()

def main():
    bars_df = pd.read_csv('/Users/mishulil/PycharmProjects/backtestingGIT/data_test/data/HBARUSDT.csv')
    bars_df = bars_df.rename(columns={'open_price': 'Open', 'close_price': 'Close', 'high_price': 'High', 'low_price': 'Low', 'close_time': 'Date', 'volume': 'Volume'})
    bars_df = resample_candles(bars_df, '5min')
    bars_df['Date_dt'] = pd.to_datetime(bars_df.index)
    bars_df = bars_df.reset_index(drop=True)

    params_to_optimize = [580, 0.9, 960, 12]
    _symbol = "HbarUsdt"
    _init_deposit = 100_000
    _max_pct_risk = 0.54
    _pct_of_reinvest = 0
    must_plot = True

    strategy_name = 'bb trend'
    current_datetime = datetime.now()
    cur_dt_string = current_datetime.strftime("%d-%m-%Y %H:%M:%S")
    print(f"{cur_dt_string}: запускаем стратегию {strategy_name}")

    strategy_results = bb_trend(bars_df, _symbol, _init_deposit, _max_pct_risk, _pct_of_reinvest, *params_to_optimize)

    minCalmar = 1.5
    minTradesInYear = 15

    strategy_series_dict = strategy_results['series_dict']
    strategy_positions_dict = strategy_results['positions_dict']

    strategy_series_df = pd.DataFrame(strategy_series_dict)
    strategy_positions_df = pd.DataFrame(strategy_positions_dict)

    metrics_calc = performance_metrics_new.PerformanceMetrics_new(
        start_capital=_init_deposit,
        Date_np=strategy_series_dict['Date_np'],
        Date_pd=strategy_series_dict['Date_pd'],
        Date_dt=strategy_series_dict['Date_dt'],
        net_profit_punkt_arr=strategy_series_dict['net_profit_arr'],
        net_profit_punkt_fixed_arr=strategy_series_dict['net_profit_fixed_arr'],
        trades_count=len(strategy_positions_dict['Position'])
    )

    metrics_values_dict = {}
    metrics_series_dict = {}

    metrics_series_dict['date_dt'] = metrics_calc.Date_dt
    metrics_series_dict['hourly_net_profit_punkt'] = metrics_calc.hourly_net_profit_punkt
    metrics_series_dict['daily_net_profit_punkt'] = metrics_calc.daily_net_profit_punkt
    metrics_series_dict['monthly_net_profit_punkt'] = metrics_calc.monthly_net_profit_punkt
    metrics_series_dict['monthly_net_profit_pct'] = metrics_calc.monthly_net_profit_pct
    metrics_series_dict['quartal_net_profit_punkt'] = metrics_calc.quartal_net_profit_punkt
    metrics_series_dict['equity_punkt_arr'] = metrics_calc.equity_punkt_arr
    metrics_series_dict['net_profit_pct_arr'] = metrics_calc.net_profit_pct_arr

    metrics_values_dict['COIN'] = _symbol
    metrics_values_dict['timeframe'] = metrics_calc.timeframe_string
    metrics_values_dict['equity_start_punkt'] = metrics_calc.equity_start_punkt
    metrics_values_dict['start_time_strategy'] = metrics_calc.start_time_strategy
    metrics_values_dict['end_time_strategy'] = metrics_calc.end_time_strategy
    metrics_values_dict['start_time'] = metrics_calc.start_time_str
    metrics_values_dict['end_time'] = metrics_calc.end_time_str
    metrics_values_dict['_max_pct_risk'] = _max_pct_risk
    metrics_values_dict['_pct_of_reinvest'] = _pct_of_reinvest
    metrics_values_dict['_bbPeriod'] = params_to_optimize[0]
    metrics_values_dict['_bbStDev'] = params_to_optimize[1]
    metrics_values_dict['_bbSplitCount'] = params_to_optimize[2]
    metrics_values_dict['_maxBandsToTrailing'] = params_to_optimize[3]
    metrics_values_dict['net_profit_end_punkt'] = metrics_calc.net_profit_end_punkt
    metrics_values_dict['net_profit_end_pct'] = metrics_calc.net_profit_end_pct
    metrics_values_dict['apr_pct'] = metrics_calc.apr_pct
    metrics_values_dict['recovery_factor_punkt'] = metrics_calc.recovery_factor_punkt
    metrics_values_dict['sharpe_month_days'] = metrics_calc.sharpe_month_days
    metrics_values_dict['sortino_month_days'] = metrics_calc.sortino_month_days
    metrics_values_dict['calmar_coeff_max_eqty'] = metrics_calc.calmar_coeff_max_eqty
    metrics_values_dict['recovery_pct_capital'] = metrics_calc.recovery_factor_pct_start_capital
    metrics_values_dict['recovery_pct_eqty'] = metrics_calc.recovery_factor_pct_max_equity
    metrics_values_dict['calmar_coeff_start_capital'] = metrics_calc.calmar_coeff_start_capital
    metrics_values_dict['beard_coeff_daily'] = metrics_calc.beard_coeff_daily
    metrics_values_dict['daily_beards_per_year'] = metrics_calc.daily_beards_per_year
    metrics_values_dict['daily_beard_max'] = metrics_calc.daily_beard_max
    metrics_values_dict['dd_worst_start_capital'] = metrics_calc.drawdown_worst_from_start_capital_pct
    metrics_values_dict['dd_worst_max_eqty'] = metrics_calc.drawdown_worst_from_max_eqty_pct
    metrics_values_dict['trades_per_year'] = metrics_calc.trades_per_year
    metrics_values_dict['months_plus_pct'] = metrics_calc.months_plus_pct
    metrics_values_dict['quartals_plus_pct'] = metrics_calc.quartals_plus_pct
    metrics_values_dict['days_plus_pct'] = metrics_calc.days_plus_pct
    metrics_values_dict['GraalMetr_NoReinvest'] = metrics_calc.graal_metr_no_reinvest
    metrics_values_dict['GraalMetr_WithReinvest'] = metrics_calc.graal_metr_with_reinvest

    netprof_df = pd.DataFrame(metrics_series_dict['net_profit_pct_arr'])
    netprof_df.index = bars_df['Date_dt']
    netprof_df.to_csv(f'/Users/mishulil/PycharmProjects/backtestingGIT/trend_backtest/netprof_{_symbol}_bbtrend.csv')

    if (metrics_values_dict["calmar_coeff_max_eqty"] > minCalmar) and (metrics_values_dict["trades_per_year"] > minTradesInYear):
        print(f' Данная комбинация параметров является хорошей')
        print(metrics_values_dict)
        metrics_current_strategy_df = pd.DataFrame([metrics_values_dict]).T
        print(metrics_current_strategy_df)
    else:
        print(f' Данная комбинация параметров является плохой:\n'
              f'\t calmar (start_capital) = {metrics_values_dict["calmar_coeff_start_capital"]:.2f}')

    if must_plot:
        strategy_charts.plot_graph(
            metrics_series_dict['date_dt'],
            metrics_series_dict['net_profit_pct_arr'],
            x_label="Time",
            y_label="NetProfit (%)",
            title="NetProfit (%)",
            legend_label="NetProfit, %",
            color="green"
        )

if __name__ == '__main__':
    main()
