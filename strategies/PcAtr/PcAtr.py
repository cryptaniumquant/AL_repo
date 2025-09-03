#для синхронизации с сервером VDV
from dataclasses import dataclass
from datetime import datetime
import sys

sys.path.append("..")
# Собственные классы
from tradingMahtematics import indicators
from tradingMahtematics import metrics_punkt
from tradingMahtematics import tradingTactics
from tradingMahtematics import array_operations
from performance_metrics import performance_metrics_new, strategy_charts
import numpy as np  # Для эффективной работы с массивами
import quantstats as qs  # Для расчета финансовых показателей и статистики

import pandas as pd
import inspect  # Для получения навзания функции внутри этой функции
from trailing_stops import trailing_stops
from strategies.PcAtr.symbol_params import symbol_params_dict
from strategies.BaseStrategy import BaseStrategy
from trading.Position import Position


class PcAtr(BaseStrategy):

    # Имя стратегии
    name = "PcAtr"

    # Объекты для хранения позиций
    LastActivePositionLong: Position = None
    LastActivePositionShort: Position = None

    def __init__(self, start_capital, rel_commission, params, is_optimization) -> None:
        self.start_capital = start_capital
        self.rel_commission = rel_commission

        self.koeff = params["koeff"]
        self.dividerAtr = params["dividerAtr"]
        self.periodAtr = params["periodAtr"]
        self.periodEnterPC = params["periodEnterPC"]
        self.periodLowEnter = self.periodEnterPC
        self.periodHighEnter = self.periodEnterPC
        self.maxLeverage = params["maxLeverage"]
    
        pass
    
    def run(self, bars_df, interval=[0, 1]):
        
        self.bars = bars_df
        self.Open  = self.bars["Open"].to_numpy()
        self.Close = self.bars["Close"].to_numpy()
        self.High  = self.bars["High"].to_numpy()
        self.Low   = self.bars["Low"].to_numpy()

        
        high_level_enter_pd = indicators.Highest(self.bars["High"], self.periodHighEnter)
        high_level_enter_pd = high_level_enter_pd.shift(1)  # сдвигаем индикатор highLevel на один бар вправо (с помощью Panda)
        
        low_level_enter_pd = indicators.Lowest(self.bars["Low"], self.periodLowEnter)
        low_level_enter_pd = low_level_enter_pd.shift(1)  # сдвигаем индикатор highLevel на один бар вправо (с помощью Panda)
        
        atr_pd = indicators.ATR_tslab_calc(self.bars["Close"], self.bars["High"], self.bars["Low"], self.periodAtr)
        
        # Рассчитываем предварительные стоп-лоссы
        stop_price_long_pd = tradingTactics.stopPriceLong_calc(high_level_enter_pd, low_level_enter_pd, self.koeff)
        stop_price_short_pd = tradingTactics.stopPriceShort_calc(high_level_enter_pd, low_level_enter_pd, self.koeff)

        # interval
        start = max(self.periodAtr, self.periodLowEnter, self.periodHighEnter)
        end = len(self.Close)

        # region Индикаторы: преобразуем в numPy

        highLevelEnter_np = high_level_enter_pd.to_numpy()
        lowLevelEnter_np = low_level_enter_pd.to_numpy()
        atrSeries_np = atr_pd.to_numpy()
        stopPriceLong_np = stop_price_long_pd.to_numpy()
        stopPriceShort_np = stop_price_short_pd.to_numpy()

        # Пустой массив (контейнер) для хранения трейлинга для сопровождения Long
        trailing_for_long_np = np.zeros(self.Close.shape, dtype=np.float64)

        # Пустой массив (контейнер) для хранения трейлинга для сопровождения Short
        trailing_for_short_np = np.zeros(self.Close.shape, dtype=np.float64)

        for bar in range(start, end, 1):
            # Получаем список активных позиций Long и Short
            self.GetActivePositionsForBar()


    def GetActivePositionsForBar(self):

        # LONG
        longActivePositions = [pos for pos in self.positions if pos.IsLong and pos.IsActive]
        self.LastActivePositionLong = longActivePositions[-1] if len(longActivePositions) > 0 else None

        # SHORT
        shortActivePositions = [pos for pos in self.positions if not pos.IsLong and pos.IsActive]
        self.LastActivePositionShort = shortActivePositions[-1] if len(shortActivePositions) > 0 else None

def PcAtr_BTC_60m_03(
        Bars_df: pd.DataFrame,
        _symbol: str,
        _init_deposit: float,
        maxPctRisk: float,
        pctOfReinvest: float,
        *params_to_optimize
) -> dict:

    min_step_lot_symbol: float = 0.00001  # Величина шага лота (минимум количества, которое можно указывать в заявке)
    min_lot_size_usd: float = 0.00  # Минимальная стоимость заявки в долларах (дешевле не допускает биржа)
    step_price_symbol: float = 0.01  # шаг изменения цены для финансового инструмента

    #region Словарь для определения параметров для каждого финансового инструмента
    symbol_params_dict = symbol_params_dict
    #endregion

    # region Собственные классы ордеров

    @dataclass
    class LongEntryOrder:
        EntryLimitPrice: float = 0.0  # Цена входа в позицию
        EntryBarNum: int = 0  # Целевой бар для входа в позицию если не равен, то не будем входить
        LotsInOrder: float = 0.0  # Количество лотов в позиции

    @dataclass
    class LongExitOrder:
        ExitLimitPrice: float = 0.0  # Цена выхода из позиции
        ExitBarNum: int = 0  # Целевой бар для выхода из позиции если не равен, то не будем выходить
        LotsInOrder: float = 0.0  # Количество лотов в позиции

    @dataclass
    class ShortEntryOrder:
        EntryLimitPrice: float = 0.0  # Цена входа в позицию
        EntryBarNum: int = 0  # Целевой бар для входа в позицию если не равен, то не будем входить
        LotsInOrder: float = 0.0  # Количество лотов в позиции

    @dataclass
    class ShortExitOrder:
        ExitLimitPrice: float = 0.0  # Цена выхода из позиции
        ExitBarNum: int = 0  # Целевой бар для выхода из позиции если не равен, то не будем выходить
        LotsInOrder: float = 0.0  # Количество лотов в позиции

    # endregion

    # region Статический Класс (контейнер методов) для работы с ордерами

    class OrdersOperations:
        @staticmethod
        def BuyAtPrice(order: LongEntryOrder, bar: int, lots: float, orderEntryPrice: float, orderText: str):
            order.EntryBarNum = bar
            order.LotsInOrder = lots
            order.EntryLimitPrice = orderEntryPrice
            # print(f"Buy Order: {orderText}, Bar: {bar}, Lots: {lots}, Price: {orderEntryPrice}")

        @staticmethod
        def SellAtPrice(order: ShortEntryOrder, bar: int, lots: float, orderEntryPrice: float, orderText: str):
            order.EntryLimitPrice = orderEntryPrice
            order.EntryBarNum = bar
            order.LotsInOrder = lots
            # print(f"Sell Order: {orderText}, Bar: {bar}, Lots: {lots}, Price: {orderEntryPrice}")

    # endregion

    # region Собственый класс LastActivePosition - класс позиции

    @dataclass
    class LastActivePosition:
        IsNull: bool = True  # Если True, то Позиция равна нулю (не существует)
        IsLong: bool = True  # Длинная позиция
        PosSizeLotShares: float = 0.00  # Сколько лотов в позиции
        AverageEntryPrice: float = 0.00  # Какая средняя цена входа в позицию
        EntryBarNum: int = 0  # Номер входа в открытую позицию
        ExitBarNum: int = 0  # Номер выхода из позиции

        # Определение функции: Когда вызывается - изменяет ордер на покупку или продажу
        def CloseAtPrice(self, bar, exit_limit_price, lots, isLong, orderText):

            if isLong:
                LongExitOrder.ExitLimitPrice = exit_limit_price
                LongExitOrder.ExitBarNum = bar
                LongExitOrder.LotsInOrder = lots
            else:
                ShortExitOrder.ExitLimitPrice = exit_limit_price
                ShortExitOrder.ExitBarNum = bar
                ShortExitOrder.LotsInOrder = lots

    # endregion

    # region Trades Table
    position = []
    lots = []
    entry_signal = []
    entry_bar = []
    entry_date = []
    entry_price = []
    entry_commission = []
    exit_signal = []
    exit_bar = []
    exit_date = []
    exit_price = []
    
    # endregion

    # region Получаем данные о ценах OHLC и дате из провайдера данных: преобразуем в numPy (для перевода на С)
    """
    Ряд данных:             bar_number из столбца индекса 'bar' датафрейма pandas
    Тип ряда данных:        ndarray: (82910,)
    Тип данных в ячейках:   int64 () 
    """
    Open_pd = Bars_df['Open']
    Open_np = Bars_df['Open'].to_numpy()
    """
    Ряд данных:             Open
    Тип ряда данных:        ndarray: (82910,)
    Тип данных в ячейках:   float64 () 
    """

    High_pd = Bars_df['High']
    High_np = Bars_df['High'].to_numpy()
    """
    Ряд данных:             High
    Тип ряда данных:        ndarray: (82910,)
    Тип данных в ячейках:   float64 () 
    """

    Low_pd = Bars_df['Low']
    Low_np = Bars_df['Low'].to_numpy()
    """
    Ряд данных:             Low
    Тип ряда данных:        ndarray: (82910,)
    Тип данных в ячейках:   float64 () 
    """
    Close_pd = Bars_df['Close']
    Close_np = Bars_df['Close'].to_numpy()
    """
    Ряд данных:             Close
    Тип ряда данных:        ndarray: (82910,)
    Тип данных в ячейках:   float64 () 
    """

    Date_np = Bars_df['Date_dt'].to_numpy()
    """
    Ряд данных:             Date_np
    Тип ряда данных:        ndarray: (82910,)
    Тип данных в ячейках:   numpy.datetime64:   Пример: numpy.datetime64('2019-09-08T20:29:59.999000000')
    """
    Date_pd = Bars_df['Date_dt']
    """
    Ряд данных:             Date_pd
    Тип ряда данных:        Series: (82910,)
    Тип данных в ячейках:   Timestamp библиотеки pandas:    Пример: Timestamp('2019-09-08 20:29:59.999000')
    """
    # with warnings.catch_warnings(action="ignore"): #убираем предупреждения о будущих изменениях
    Date_dt = np.array(Date_pd.dt.to_pydatetime()) #datetime: Пример: datetime.datetime(2019, 9, 8, 20, 29, 59, 999000)
    """
    Ряд данных:             Date_dt
    Тип ряда данных:        ndarray: (82910,)
    Тип данных в ячейках:   datetime.datetime:    Пример: datetime.datetime(2019, 9, 8, 20, 29, 59, 999000)
    """
    # endregion

    bars_count = len(Bars_df) # Количество свечей в торговой стратегии


    # region Создаём пустые массивы (контейнеры) для метрик, которые считаются в процессе торговли
    net_profit_fixed_arr = np.zeros(bars_count, dtype=np.float64)  # для хранения реализованного pnl
    net_profit_arr = np.zeros(bars_count, dtype=np.float64)  # для хранения для хранения текущего pnl
    positions_arr = np.array([], dtype=np.float64)  # Пустой массив (контейнеры) для хранения всех сделок

    # endregion

    # region Переменные для всех торговых систем

    first_valid_value: int = 0  # первый номер бара, на котором существуют все индикаторы

    # Для Long
    isLong: bool = False  # Находимся в длинной позиции?
    signalBuy: bool = False  # Сигнал на вход в длинную позицию
    orderEntryLong: float = 0.00  # Цена, где будет расположен вход в длинную позицию
    stopPriceLong: float = 0.00  # Цена где будет расположен StopLoss длинной позиции
    trailing_stop_long: float = 0.00  # Цена, где будет располагаться трейлинг
    kontraktLong_mPR: float = 0.00  # Количество лотов для входа в длинную позицию
    entryPriceLong: float = 0.00  # Текущая цена открытия позиции лонг

    # Для Short
    isShort: bool = False  # Находимся в короткой позиции?
    signalShort: bool = False  # Сигнал на вход в короткую позицию
    orderEntryShort: float = 0.00  # Цена, где будет расположен вход в длинную позицию
    stopPriceShort: float = 0.00  # Цена где будет расположен StopLoss длинной позиции
    trailing_stop_short: float = 0.00  # Цена, где будет располагаться трейлинг
    kontraktShort_mPR: float = 0.00  # Количество лотов для входа в длинную позицию

    # для выхода из позиции
    exitLimitPrice: float = 0.00  # //Цена выставления Лимитированной заявки на выход

    # Для передачи Менеджеру
    lots_in_order: float = 0.00
    ActiveOrderPrice: float = 0.00
    LotsInPosition: float = 0.00



    # endregion

    # region Создаём ордера для Лонга и Шорта (вход и выход) - Создание экземпляров класса ордеров

    LongEntryOrder = LongEntryOrder()
    LongExitOrder = LongExitOrder()
    ShortEntryOrder = ShortEntryOrder()
    ShortExitOrder = ShortExitOrder()

    # endregion

    LastActivePosition = LastActivePosition()  # region Создаем объект LastActivePosition с исходными значениями


    # region Забираем значения из параметров торговой системы

    # Получаем параметры из комбинации - например: [0.3, 10, 200, 20]  # args

    
    InitDeposit: float = _init_deposit  # Величина стартового капитала
    symbol = "BtcUsdt"
    _maxPctRisk = maxPctRisk
    _pctOfReinvest = pctOfReinvest

    # Для конкретной торговой стратегии
    periodHighEnter: int = int(_periodEnterPC)  # Определяем период канала
    periodLowEnter: int = int(_periodEnterPC)
    periodExitPC: int = int(_periodEnterPC)
    maxLeverage: int = int(maxLeverage)
    koeff: float = float(_koeff)  # Коэфф, показывающий на сколько ужимаем (if<1 или увеличиваем if > 1) ширину канала
    periodAtr: int = int(_periodAtr)  # Период ATR
    dividerAtr: float = float(_dividerAtr)  # Делитель ATR

    # Для комиссии
    useAbsComission: bool = False  # Применять абсолютную комиссию? (или относительную)
    absComission: float = 0.00  # Величина абсолютной комиссии (рублей)
    relComission: float = 0.05  # Величина относительной комиссии (%)

    # Для настройки методов управления размером позиции
    maxPercentRisk:float = _maxPctRisk # Максимальный риск в одной сделке
    finResReinvestPct:float = _pctOfReinvest # Сколько % финансового дохода реинвестировать (от 0 до 100)

    # Прочие настройки
    maxCountOfMinLotSteps: int = 10000000000  # max количество минимальных лотов (чтобы ограничивать масштаб торговли)
    tradeMinLotSize: bool = False  # Нужно ли торговать только одним контрактом?


    # endregion

    # region Переменные для работы с PosSizer

    FinResForBar: float = 0.00  # Хранится финансовый результат на текущий бар (для Лаборатории)
    moneyForTradingSystem: float = 0.00  # Переменная, которая будет хранить в себе оценку портфеля

    # endregion


    # region Рассчитываем Индикаторы и находим first_valid_value


    
    # endregion

    # region Информационное табло для отображения на графике даты начала, окончания торговли и FirstValidValue

    first_date = Date_dt[0]
    first_date_string = first_date.strftime("%d-%m-%Y")
        #f"{first_date.day}.{first_date.month}.{first_date.year}"  # текстовая переменная

    last_date = Date_dt[-1]
    last_date_string = last_date.strftime("%d-%m-%Y")
        #f"{last_date.day}.{last_date.month}.{last_date.year}"  # текстовая переменная


    # endregion

    # region Главный торговый цикл

    # пробегаемся по всем барам от first_valid_value до последнего бара
    for bar in nb.prange(int(first_valid_value), bars_count):

        # region Работа с ордерами - проверяем сработали ли ордера на текущей свече по ордерам с предыдущей


        if LastActivePosition.IsNull == True:  # Если активная позиция отсутствует

            if max(LongEntryOrder.EntryBarNum, ShortEntryOrder.EntryBarNum) == bar:  # Вход возможен та текущем баре
                pass  # print(f"Хотим войти на баре №{bar}")  # ничего не делаем

            # region Срабатывает ордер на вход в Long
            if LongEntryOrder.EntryBarNum == bar and Low_np[bar] < LongEntryOrder.EntryLimitPrice:

                # происходит вход в длинную позицию

                if write_to_log:
                    print(
                        f"BuyLimit at bar:\t\t"
                        f" {bar},\t price:\t"
                        f" {LongEntryOrder.EntryLimitPrice},\t"
                        f" Open[bar]\t = {Open_np[bar]}"
                    )  # выводит входы в лонг

                # Изменяем Поля позиции
                LastActivePosition.IsNull = False  # Позиция теперь существует
                LastActivePosition.IsLong = True  # Длинная позиция
                LastActivePosition.PosSizeLotShares = LongEntryOrder.LotsInOrder  # Сколько лотов в позиции
                LastActivePosition.AverageEntryPrice = LongEntryOrder.EntryLimitPrice  # Какая средняя цена входа в позицию
                LastActivePosition.EntryBarNum = bar  # Номер входа в открытую позицию

                # Добавляем информацию в таблицу о позициях
                position.append('Long')
                entry_signal.append('LongEnter')
                entry_bar.append(bar)
                lots.append(LongEntryOrder.LotsInOrder)
                entry_price.append(LongEntryOrder.EntryLimitPrice)
                entry_date.append(Date_dt[bar])
            # endregion

            # region Срабатывает ордер на вход позиции в Short
            if ShortEntryOrder.EntryBarNum == bar and High_np[bar] > ShortEntryOrder.EntryLimitPrice:

                # происходит вход в короткую позицию
                if write_to_log:
                    print(
                        f"SellLimit at bar:\t\t"
                        f" {bar},\t price:\t "
                        f"{ShortEntryOrder.EntryLimitPrice},\t"
                        f" Open[bar]\t = {Open_np[bar]}"
                    )  # выводит входы в Шорт

                # Изменяем своцства позиции
                LastActivePosition.IsNull = False  # Позиция теперь существует
                LastActivePosition.IsLong = False  # Короткая позиция
                LastActivePosition.PosSizeLotShares = ShortEntryOrder.LotsInOrder  # Сколько лотов в позиции
                LastActivePosition.AverageEntryPrice = ShortEntryOrder.EntryLimitPrice  # Какая средняя цена входа в позицию
                LastActivePosition.EntryBarNum = bar  # Номер входа в открытую позицию

                #Добавляем информацию в таблицу о позициях
                position.append('Short')
                entry_signal.append('ShortEnter')
                entry_bar.append(bar)
                lots.append(-1.0 * ShortEntryOrder.LotsInOrder)
                entry_price.append(ShortEntryOrder.EntryLimitPrice)
                entry_date.append(Date_dt[bar])
            # endregion

        elif LastActivePosition.IsNull != True:  # Если активная позиция существует

            # region срабатывает ордер на выход из длинной позиции

            if LongExitOrder.ExitBarNum == bar and High_np[bar] > LongExitOrder.ExitLimitPrice:
                # происходит выход из длинной позиции
                if write_to_log:
                    print(
                        f"ExitLongLimit at bar:\t"
                        f" {bar},\t price:\t"
                        f" {LongExitOrder.ExitLimitPrice},\t"
                        f" Open[bar]\t = {Open_np[bar]}"
                    )  # выводит выход из лонг

                LastActivePosition.IsNull = True  # Указываем, что активной позиции сейчас не существует
                LastActivePosition.ExitBarNum = bar  # Номер выхода из позиции

                # Добавляем информацию в таблицу о позициях
                exit_signal.append('ExitLong')
                exit_bar.append(bar)
                exit_price.append(LongExitOrder.ExitLimitPrice)
                exit_date.append(Date_dt[bar])
            # endregion

            # region Срабатывает ордер на выход из короткой позиции
            if ShortExitOrder.ExitBarNum == bar and Low_np[bar] < ShortExitOrder.ExitLimitPrice:
                # происходит выход из короткой позиции
                if write_to_log:
                    print(
                        f"ExitShortLimit at bar:\t {bar},\t"
                        f" price:\t {ShortExitOrder.ExitLimitPrice},\t"
                        f" Open[bar]\t = {Open_np[bar]}"
                    )  # выводит выход из лонг

                LastActivePosition.IsNull = True  # теперь активной позиции не существует
                LastActivePosition.ExitBarNum = bar  # Номер выхода из позиции

                # Добавляем информацию в таблицу о позициях
                exit_signal.append('ExitShort')
                exit_bar.append(bar)
                exit_price.append(ShortExitOrder.ExitLimitPrice)
                exit_date.append(Date_dt[bar])
            # endregion

        # endregion


        #region Рассчитываем NetProfit и net_profit_fixed
        currentProfit: float = 0.00

        # region Если позиция отсутствует
        if LastActivePosition.IsNull == True:  # Позиции не существует

            # region считаем currentProfit
            if LongExitOrder.ExitBarNum == bar:  # Если вышли из длинной позиции на текущем баре
                currentProfit = tradingTactics.position_profit_calc(True, LastActivePosition.AverageEntryPrice,
                                                                    LongExitOrder.ExitLimitPrice,
                                                                    LastActivePosition.PosSizeLotShares,
                                                                    relComission)
            elif ShortExitOrder.ExitBarNum == bar:  # Если вышли из короткой позиции на текущем баре
                currentProfit = tradingTactics.position_profit_calc(False, LastActivePosition.AverageEntryPrice,
                                                                    ShortExitOrder.ExitLimitPrice,
                                                                    LastActivePosition.PosSizeLotShares,
                                                                    relComission)

            else:  # Позиции не было ни на предыдущем баре ни на текущем баре
                currentProfit = 0

            # endregion

            net_profit_fixed_arr[bar] = net_profit_fixed_arr[bar - 1] + currentProfit
            net_profit_arr[bar] = net_profit_fixed_arr[bar]

            if LastActivePosition.ExitBarNum == bar:  # Если на текущем баре закрылась позиция
                positions_arr = np.append(positions_arr, net_profit_fixed_arr[bar])
                pass  # TODO: сформировать здесь массив из экземлпяра класса Position
        # endregion

        # region Если позиция существует
        elif LastActivePosition.IsNull != True:  # Позиция существует

            net_profit_fixed_arr[bar] = net_profit_fixed_arr[bar - 1]  # Расчёт NetProfitFixed при существующей позиции


            if LastActivePosition.IsLong == True:  # Long
                currentProfit = tradingTactics.position_profit_calc(True, LastActivePosition.AverageEntryPrice,
                                                                    Close_np[bar],
                                                                    LastActivePosition.PosSizeLotShares,
                                                                    relComission)
            elif LastActivePosition.IsLong == False:  # Short

                currentProfit = tradingTactics.position_profit_calc(False, LastActivePosition.AverageEntryPrice,
                                                                    Close_np[bar],
                                                                    LastActivePosition.PosSizeLotShares,
                                                                    relComission)
            else:
                print("Позиция не может быть не длинной ни короткой")
                break

            net_profit_arr[bar] = net_profit_fixed_arr[bar] + currentProfit  # Расчёт NetProfit при существующей позиции

        # endregion

        #endregion

        # region Рассчитываем текущую позицию и цену входа (если она существует)

        if LastActivePosition.IsNull != True:  # Если активная позиция существует
            # Величина текущей позиции в количестве лотов (если отрицательная, значит короткая позиция)
            if LastActivePosition.IsLong:  # Является ли позиция длинной?
                LotsInPosition: float = LastActivePosition.PosSizeLotShares  # Если Long - считаем так

            else:
                LotsInPosition: float = -1 * LastActivePosition.PosSizeLotShares  # Если Short - считаем так

            EntryPrice = LastActivePosition.AverageEntryPrice  # Цена входа в позицию если позиция есть
        else:  # Если активной позиции нет
            LotsInPosition: float = 0.00  # считаем так
            LotsInPosition = float(LotsInPosition)
            EntryPrice = float('nan')  # Цена входа в позицию, если позиции нет

        # endregion

        # region Определяем цены лимитных заявок:

        orderEntryLong = Close_np[bar]  # Цена заявки для входа в длинную позицию
        orderEntryLong = tradingTactics.RoundPrice(orderEntryLong, step_price_symbol)  # Округляем цену до min тика

        orderEntryShort = Close_np[bar]  # Цена заявки для входа в короткую позицию
        orderEntryShort = tradingTactics.RoundPrice(orderEntryShort,step_price_symbol)  # Округляем цену до мim тика

        exitLimitPrice = Close_np[bar]  # Цена заявки на выход из позиции (и для лонга и для шорта)
        exitLimitPrice = tradingTactics.RoundPrice(exitLimitPrice, step_price_symbol)  # Округляем цену до min тика
        # endregion

        # region Условия на вход в позицию

        signalBuy = Close_np[bar] > highLevelEnter_np[bar]  # В лонг
        signalShort = Close_np[bar] < lowLevelEnter_np[bar]  # В шорт

        # endregion

        # region Сопровождение и выход из позиции

        # region Если позиция отсутствует

        if LastActivePosition.IsNull == True:  # Если активной позиции не существует

            # region Нужно входить в длинную позицию?
            if signalBuy:  # Пришёл сигнал в длинную позицию

                stop_price_long = highLevelEnter_np[bar] - (
                        highLevelEnter_np[bar] - lowLevelEnter_np[bar]) * koeff  # устанавливаем стоп-лос
                stop_price_long = tradingTactics.RoundPrice(stop_price_long,
                                                              step_price_symbol)  # округляем цену до минимального тика - шага

                trailing_stop_long = stop_price_long  # Задаём первое значение трейлинга
                trailing_for_long_np[bar] = trailing_stop_long  # Вносим значение для рисования на графике


                # Определяем кол-во контрактов на покупку

                FinResForBar = net_profit_fixed_arr[bar]  # Рассчитываем стоимость дохода на текущий бар в пунктах
                FinResForBar = FinResForBar * finResReinvestPct / 100.0  # Сколько % дохода реинвестировать

                # Определяем сумму для торговой системы
                moneyForTradingSystem = InitDeposit + FinResForBar


                kontraktLong_mPR = tradingTactics.MaxPctRiskBinance(
                    moneyForTradingSystem,
                    maxPercentRisk,
                    orderEntryLong,
                    stop_price_long,
                    min_step_lot_symbol,
                    min_lot_size_usd,
                    maxCountOfMinLotSteps,
                    tradeMinLotSize
                )

                kontraktLong_poE = tradingTactics.pctOfEquityBinance(moneyForTradingSystem, maxLeverage,
                                                                      orderEntryLong,
                                                                      min_step_lot_symbol,
                                                                      min_lot_size_usd,
                                                                      maxCountOfMinLotSteps, tradeMinLotSize)
                kontraktLong_mPR = min(kontraktLong_mPR, kontraktLong_poE)

                # округляем до минимального шага лота
                kontraktLong_mPR = tradingTactics.RoundToMinLotStep(
                    kontraktLong_mPR,
                    min_step_lot_symbol
                )

                # Входим не менее чем на 1 минимальный контракт и не менее чем на минимальную стоимость заявки
                if (kontraktLong_mPR >= min_step_lot_symbol) and (kontraktLong_mPR * orderEntryLong > min_lot_size_usd):
                    orderText = "LongEnter"
                    lots_in_order = kontraktLong_mPR  # Указываем, сколько штук в активной заявке
                    ActiveOrderPrice = orderEntryLong

                    OrdersOperations.BuyAtPrice(
                        LongEntryOrder,
                        bar + 1,
                        kontraktLong_mPR,
                        orderEntryLong,
                        orderText
                    )
                else:
                    kontraktLong_mPR = tradingTactics.MinPosSizeBinanceSpot(orderEntryLong, min_step_lot_symbol,
                                                                            min_lot_size_usd)
                    kontraktLong_mPR = tradingTactics.RoundToMinLotStep(kontraktLong_mPR,
                                                                        min_step_lot_symbol)  # округляем до минимального шага цены

                    kontraktLong_mPR = min(kontraktLong_mPR, maxCountOfMinLotSteps * min_step_lot_symbol)
                    orderText = "LongEnter_minKontrakt"
                    lots_in_order = kontraktLong_mPR  # Указываем, сколько штук в активной заявке
                    ActiveOrderPrice = orderEntryLong
                    OrdersOperations.BuyAtPrice(LongEntryOrder, bar + 1, kontraktLong_mPR, orderEntryLong,
                                                orderText)  # входим хотя бы 1 контрактом
            # endregion

            # region Нужно входить в короткую позицию?

            elif signalShort:
                # Пришёл сигнал в короткую позицию

                stop_price_short = lowLevelEnter_np[bar] + (highLevelEnter_np[bar] - lowLevelEnter_np[
                    bar]) * koeff  # Устанавливаем Стоп (для расчёта кол-ва контрактов)
                stop_price_short = tradingTactics.RoundPrice(stop_price_short,
                                                               step_price_symbol)  # округляем цену до минимального тика

                trailing_stop_short = stop_price_short  # Присваиваем значение трейлинга (barEntry - 1)
                trailing_for_short_np[bar] = trailing_stop_short  # для прорисовки

                # определяем прибыль от торговли на текущий момент и показываем, что хотим учесть даже ещё незафиксированную прибыль

                FinResForBar = net_profit_fixed_arr[bar]  # Рассчитываем доход по портфелю на текущий бар в пунктах
                FinResForBar = FinResForBar * finResReinvestPct / 100  # сколько % реинвестируем

                # определяем сумму для торговой системы
                moneyForTradingSystem = InitDeposit + FinResForBar

                kontraktShort_mPR = tradingTactics.MaxPctRiskBinance(moneyForTradingSystem, maxPercentRisk,
                                                                     orderEntryShort,
                                                                     stop_price_short, min_step_lot_symbol, min_lot_size_usd,
                                                                     maxCountOfMinLotSteps, tradeMinLotSize)
                kontraktShort_poE = tradingTactics.pctOfEquityBinance(moneyForTradingSystem, maxLeverage,
                                                                     orderEntryShort,
                                                                     min_step_lot_symbol,
                                                                     min_lot_size_usd,
                                                                     maxCountOfMinLotSteps, tradeMinLotSize)
                kontraktShort_mPR = min(kontraktShort_mPR, kontraktShort_poE)
                kontraktShort_mPR = tradingTactics.RoundToMinLotStep(kontraktShort_mPR, min_step_lot_symbol)

                # Входим не менее чем на 1 минимальный контракт
                if (kontraktShort_mPR >= min_step_lot_symbol) and (kontraktShort_mPR * orderEntryShort > min_lot_size_usd):
                    orderText = "ShortEnter"
                    lots_in_order = -1.0 * kontraktShort_mPR  # Показываем сколько штук должно войти в короткую позицию
                    ActiveOrderPrice = orderEntryShort
                    OrdersOperations.SellAtPrice(ShortEntryOrder, bar + 1, kontraktShort_mPR, orderEntryShort,
                                                 orderText)
                else:  # объём ордера меньше минимального по условиям биржи
                    # Входим на минимально возможный по правилам биржи размер позиции (не менее 10 долларов)
                    kontraktShort_mPR = tradingTactics.MinPosSizeBinanceSpot(orderEntryShort, min_step_lot_symbol,
                                                                             min_lot_size_usd)  # даже если не хватает денег - заходим одним контрактом
                    kontraktShort_mPR = tradingTactics.RoundToMinLotStep(kontraktShort_mPR, min_step_lot_symbol)
                    orderText = "ShortEnter_minKontrakt"
                    lots_in_order = -1.0 * kontraktShort_mPR  # Показываем сколько штук должно войти в короткую позицию
                    ActiveOrderPrice = orderEntryShort
                    OrdersOperations.SellAtPrice(ShortEntryOrder, bar + 1, kontraktShort_mPR, orderEntryShort,
                                                 orderText)

            # endregion

        # endregion

        # region Если позиция существует
        else:  # if LastActivePosition.IsNull == False:  # Если активная позиция существует

            # region Длинная позиция
            if LastActivePosition.IsLong and LastActivePosition.EntryBarNum != bar:

                if bar == LastActivePosition.EntryBarNum:  # для бара входа
                    prevBar = LastActivePosition.EntryBarNum - 1  #
                    trailing_stop_long = stop_price_long_pd[prevBar]  # устанавливаем стоп-лос

                    trailing_stop_long = tradingTactics.RoundPrice(
                        trailing_stop_long, step_price_symbol)  # округляем цену до минимального тика - шага
                else:  # Если находимся не на баре входа в позицию
                    mustMoveTrailing = Close_np[bar] > Open_np[bar]
                    CurrentAtrValue = atrSeries_np[bar]
                    trailing_stop_long = trailing_stops.trailing_atr_percent(Bars_df, LastActivePosition, trailing_stop_long,
                                                                             atrSeries_np, _dividerAtr, bar, False)

                    trailing_stop_long = tradingTactics.RoundPrice(trailing_stop_long,
                                                                 step_price_symbol)  # округляем цену до минимального тика - шага

                trailing_for_long_np[bar] = trailing_stop_long  # для прорисовки
                ExitLong = False
                ExitLong = Close_np[bar] < trailing_stop_long  # Условие на выход из длинной позиции (техника на выход)

                # нужно выходить на следующем баре
                if ExitLong:
                    lots_in_order = -1.0 * LotsInPosition  # Показываем, что мы должны выйти из позиции
                    ActiveOrderPrice = exitLimitPrice

                    # Вызывая этот метод мы устанавливаем для LongExitOrder Цену и бар выхода равный следующему бару
                    LastActivePosition.CloseAtPrice(
                        bar + 1,
                        exitLimitPrice,
                        lots_in_order,
                        True,
                        "Exit Long")

            # endregion

            # region Короткая позиция
            elif LastActivePosition.IsLong == False and LastActivePosition.EntryBarNum != bar:

                if bar == LastActivePosition.EntryBarNum:  # для бара входа
                    prevBar = LastActivePosition.EntryBarNum - 1

                    trailing_stop_short = stop_price_short_pd[prevBar]  # устанавливаем стоп-лос
                    trailing_stop_short = tradingTactics.RoundPrice(
                        trailing_stop_short, step_price_symbol)  # округляем цену до минимального тика - шага
                    trailing_initial_value = trailing_stop_short

                else:  # не для бара входа
                    mustMoveTrailing = Close_np[bar] < Open_np[bar]  # Если свеча чёрная - то двигаем
                    CurrentAtrValue = atrSeries_np[bar]

                    trailing_stop_short = trailing_stops.trailing_atr_percent(Bars_df, LastActivePosition, trailing_stop_short,
                                                                             atrSeries_np, _dividerAtr, bar, False)

                    trailing_stop_short = tradingTactics.RoundPrice(
                        trailing_stop_short, step_price_symbol)  # округляем цену до минимального тика - шага

                trailing_for_short_np[bar] = trailing_stop_short  # для прорисовки

                ExitShort = False
                ExitShort = Close_np[bar] > trailing_stop_short


                if ExitShort: # нужно выходить на следующем баре
                    lots_in_order = -1.0 * LotsInPosition  # Показываем, что мы должны выйти из позиции
                    ActiveOrderPrice = exitLimitPrice

                    # Вызывая этот метод мы устанавливаем для ShortExitOrder Цену и бар выхода равный следующему бару
                    LastActivePosition.CloseAtPrice(
                        bar + 1,
                        exitLimitPrice,
                        lots_in_order,
                        False,
                        "Exit Short")

            # endregion

        # endregion

        # endregion

    else: #при выходе из главного торгового цикла
        pass
        #print("Главный цикл закончился")

    # endregion конец главного торгового цикла

    #region создаём словарь series_dict для передачи данных необходимых рядов данных

    series_dict = {}

    series_dict['Open_np'] = Open_np
    series_dict['High_np'] = High_np
    series_dict['Low_np'] = Low_np
    series_dict['Close_np'] = Close_np
    series_dict['Date_np'] = Date_np
    series_dict['Date_pd'] = Date_pd
    series_dict['Date_dt'] = Date_dt
    series_dict['net_profit_arr'] = net_profit_arr
    series_dict['net_profit_fixed_arr'] = net_profit_fixed_arr
    series_dict['trailing_for_long_np'] = trailing_for_long_np
    series_dict['trailing_for_short_np'] = trailing_for_short_np
    #series_dict['positions_arr'] = positions_arr #единственный Series, у которого длина массива не совпадает с другими

    #endregion

    #region Создаём словарь positions_dict
    positions_dict = {}


    positions_dict['Position'] = position[0:len(exit_bar)]
    positions_dict['Symbol'] = (['SOLUSDT']*len(positions_dict['Position']))[0:len(exit_bar)]
    positions_dict['Lots'] = lots[0:len(exit_bar)]
    positions_dict['Entry Signal'] = entry_signal[0:len(exit_bar)]
    positions_dict['Entry Bar'] = entry_bar[0:len(exit_bar)]
    positions_dict['Entry Price'] = entry_price[0:len(exit_bar)]
    positions_dict['Entry Date'] = entry_date[0:len(exit_bar)]
    positions_dict['Exit Bar'] = exit_bar
    positions_dict['Exit Price'] = exit_price
    positions_dict['Exit Date'] = exit_date
    positions_dict['Exit Signal'] = exit_signal
    #endregion

    #endregion

    return {
        'series_dict': series_dict,
        'positions_dict': positions_dict
    } # передаём результаты работы стратегии в вызывающую функцию

# endregion Конец Стратегии

# region Функция для генерации уникальных комбинаций из параметров стратегии

def get_combinations(_pct_to_remove = 0): #по умолчанию удаляем ноль
    """
    Функция генерации уникальных комбинаций из
    параметров торговой стратегии
    """
    # mPr = np.array({0.3, 0.5, 0.8, 1.0})  # np.array({1, 2, 3})  # Задаём несколько значений, котороые хотим просчитать явно

    # Настраиваем параметры для оптимизации
    number_parameters = 4 #количество параметров (столбцов)
    percent_to_remove = _pct_to_remove  # Сколько % возможных параметров нужно убрать (Процент ячеек для удаления массива)
    names_parameters = ['koeff', 'dividerAtr', 'periodAtr', 'periodEnterPC'] # Имена параметров (названия столбцов)

    #Шаг №2 - region Меняем параметры для оптимизации для каждого таймфрйма
    #region Создаём одномерные массивы (с помощью шага), наполненные возможными значениями параметров
    #текущие настройки для таймфрейма 30m
    set_koeff = np.arange(0.5, 1.5, 0.1)  # TODO: значение по умолчанию и возможно сюда же стринговое название
    set_dividerAtr = np.arange(0.2, 20, 0.2)  # arrange - создание массива с помощью шага
    set_periodAtr = np.arange(500, 1300, 100)
    set_periodEnterPC = np.arange(200, 1300, 10)
    #endregion

   #region Формируем сетку комбинаций параметров из уменьшенных одномерных массивов

    #создаём координатную сетку из нескольких одномерных массивов (здесь будут все возможные сочетания)
    grids = np.meshgrid(set_koeff, set_dividerAtr, set_periodAtr, set_periodEnterPC)

    #Создаём "таблицу", где столбцами будут параметры стратегии, а строками - все их возможные значения
    grids_np_arr = np.array(grids) #преобразуем матрицу в массив numpy
    grids_np_arr_transposed = grids_np_arr.T #транспонируем массив (меняем строки местами со столбцами
    unique_combinations = grids_np_arr_transposed.reshape(-1,number_parameters) #преобразуем форму массива с нужным количеством столбцов

    #unique_combinations = np.array(grids).T.reshape(-1, number_parameters)

    #endregion

    # Возвращаем список уникальных параметров с именами
    # region Сокращаем количество элементов в одномерных массивах параметров на указанное кол-во процентов

    np.random.shuffle(unique_combinations) #перемешиваем список
    combinations_short = array_operations.reduce_array_by_percentage(unique_combinations, percent_to_remove)
    np.random.shuffle(combinations_short) #ещё раз перемешиваем

    # endregion
    return names_parameters, combinations_short

# endregion

def resample_candles(df, timeframe):
    df.index = pd.to_datetime(df.Date)
    return df.resample(timeframe).agg({
        'Open': 'first',
        'Close': 'last',
        'High': 'max',
        'Low': 'min',
        'Volume': 'sum'
    }).dropna()

def vizualization(metrics_series_dict, metrics_values_dict, bars_df):

    strategy_charts.plot_graph(
        metrics_series_dict['date_dt'],
        metrics_series_dict['net_profit_pct_arr'],
        x_label="Time",
        y_label="NetProfit (%)",
        title="NetProfit (%)",
        legend_label="NetProfit, %",
        color="green"
    )

    strategy_charts.plot_series_range(
        series=metrics_series_dict['monthly_net_profit_pct'],
        start_date=metrics_values_dict['start_time_strategy'],
        end_date=metrics_values_dict['end_time_strategy'],
        title=f"Стартовая сумма депозита = {round(metrics_values_dict['equity_start_punkt'], 0)}. месячный NetProfit (%)",
        xlabel=f"Период торговли: с {metrics_values_dict['start_time_strategy'].strftime('%d-%m-%Y')} по {metrics_values_dict['end_time_strategy'].strftime('%d-%m-%Y')}",
        ylabel="NetProfit (%)"
    )

    strategy_charts.plot_graph_with_close(
        metrics_series_dict['date_dt'],
        metrics_series_dict['net_profit_pct_arr'],
        bars_df['Close'],
        x_label="Time",
        y_label="NetProfit (%)",
        title="NetProfit x B&H (%)",
        legend_label_equity="NetProfit, %",
        color_equity="green",
        color_close='blue'
    )


# region main(): Запуск стратегии автономно - Устанавливаем параметры и запускаем стратегию автономно

def main():
    tf = '60min'

    # Провайдер данных - подгружаем данные
    bars_df = pd.read_csv('/Users/mishulil/PycharmProjects/backtestingGIT/data_test/data/MKRUSDT.csv')
    print(bars_df)
    # bars_df = bars_df.rename(columns={'open': 'Open', 'close': 'Close', 'high': 'High', 'low': 'Low', 'time': 'Date'})
    bars_df = bars_df.rename(columns={'open_price': 'Open', 'close_price': 'Close',
                                      'high_price': 'High', 'low_price': 'Low',
                                      'close_time': 'Date', 'volume': 'Volume'})

    bars_df = resample_candles(bars_df, f'{tf}')
    #bars_df = bars_df[bars_df.index > pd.to_datetime('2025-05-23 20:00')]
    # bars_df = bars_df.reset_index()
    # bars_df.index = pd.to_datetime(bars_df['Unnamed: 0'])
    # bars_df = bars_df[bars_df.index > pd.to_datetime('2024-02-01')]
    bars_df['Date_dt'] = pd.to_datetime(bars_df.index)
    # bars_df['Date_dt'] = pd.to_datetime(bars_df['Date'])
    bars_df = bars_df.reset_index(drop=True)


    # Получаем список уникальных комбинаций параметров. Формируется с помощью функции генерации
    params_to_optimize = [1.2, 9, 488, 60, 100]

    # koeff, dividerAtr, PeriodAtr, PeriodEnterPC
    _koeff, _dividerAtr, _periodAtr, _periodEnterPC, maxLeverage = params_to_optimize #забираем значения для этой функции
    _symbol = "MkrUsdt"
    _init_deposit = 100_000
    _max_pct_risk = 6.5
    _pct_of_reinvest = 0 #_max_pct_risk, _pct_of_reinvest
    must_plot = True #нужно ли рисовать графики


    # запускаем стратегию с комбинациями параметров и забираем результаты её выполнения (любые уже посчитанные метрики)

    strategy_name = 'PcAtr_BTC_60m_03' #Вводим название стратегии для отображения на графиках

    # Получаем текущую дату и время
    current_datetime = datetime.now()
    cur_dt_string = current_datetime.strftime("%d-%m-%Y %H:%M:%S")

    # Выводим текущую дату и время в формате ГГГГ-ММ-ДД ЧЧ:ММ:СС
    print(f"{cur_dt_string}: запускаем стратегию {strategy_name}")

    strategy_results = PcAtr_BTC_60m_03(
        bars_df,
        _symbol,
        _init_deposit,
        _max_pct_risk,
        _pct_of_reinvest,
        *params_to_optimize
    )

    # Получаем от стратегии в виде словаря:
    strategy_series_dict = strategy_results['series_dict']  #все series, необходимые для рассчёта метрик
    strategy_positions_dict = strategy_results['positions_dict'] #таблицу позиций (сделок)

    #region Создаём из словаря два датафрейма:

    strategy_series_df = pd.DataFrame(strategy_series_dict)
    strategy_positions_df = pd.DataFrame(strategy_positions_dict)
    print(strategy_series_df.head(10))
    print(strategy_positions_df.tail(30))
    #endregion

    metrics_calc = performance_metrics_new.PerformanceMetrics_new(
        start_capital=100_000.0,
        Date_np=strategy_series_dict['Date_np'],  # Date_np,
        Date_pd=strategy_series_dict['Date_pd'],  # Date_np,
        Date_dt=strategy_series_dict['Date_dt'],  # Date_np,
        net_profit_punkt_arr=strategy_series_dict['net_profit_arr'],  # net_profit_arr,
        net_profit_punkt_fixed_arr=strategy_series_dict['net_profit_fixed_arr'],  # net_profit_fixed_arr,
        trades_count=len(strategy_positions_dict['Position'])  # positions_arr.size #количество сделок
    )
    #endregion



    #region Считаем метрики и сохраняем их в словарь metrics_values_dict


    # Создание пустого словаря
    metrics_values_dict = {}
    metrics_series_dict = {}

    # Добавление элементов в словарь metrics_values_dict

    #Ряды данных


    #Исходные данные:
    metrics_series_dict['date_dt'] = metrics_calc.Date_dt

    #для гистограмм
    metrics_series_dict['hourly_net_profit_punkt'] = metrics_calc.hourly_net_profit_punkt  # получаем по NetProfit
    metrics_series_dict['daily_net_profit_punkt'] = metrics_calc.daily_net_profit_punkt
    metrics_series_dict['monthly_net_profit_punkt'] = metrics_calc.monthly_net_profit_punkt
    metrics_series_dict['monthly_net_profit_pct'] = metrics_calc.monthly_net_profit_pct
    metrics_series_dict['quartal_net_profit_punkt'] = metrics_calc.quartal_net_profit_punkt  # получаем по NetProfit

    #для графиков

    metrics_series_dict['equity_punkt_arr'] = metrics_calc.equity_punkt_arr
    metrics_series_dict['net_profit_pct_arr'] = metrics_calc.net_profit_pct_arr

    #Характеристика торговой стратегии
    metrics_values_dict['COIN'] = 'BtcUsdt'
    metrics_values_dict['timeframe'] = metrics_calc.timeframe_string
    metrics_values_dict['equity_start_punkt'] = metrics_calc.equity_start_punkt

    metrics_values_dict['start_time_strategy'] = metrics_calc.start_time_strategy
    metrics_values_dict['end_time_strategy'] = metrics_calc.end_time_strategy

    metrics_values_dict['start_time'] = metrics_calc.start_time_str
    metrics_values_dict['end_time'] = metrics_calc.end_time_str

    metrics_values_dict['_max_pct_risk'] = _max_pct_risk
    metrics_values_dict['_pct_of_reinvest'] = _pct_of_reinvest


    #Параметры
    metrics_values_dict['_koeff'] = _koeff
    metrics_values_dict['_dividerAtr'] = _dividerAtr
    metrics_values_dict['_periodAtr'] = _periodAtr
    metrics_values_dict['_periodEnterPC'] = _periodEnterPC


    #Основные метрики

    recovery_factor_graal = metrics_calc.recovery_factor_graal #1.75
    sharpe_month_days_graal = metrics_calc.sharpe_month_days_graal  # 0.6467
    sortino_month_days = metrics_calc.sortino_month_days

    avg_profit_pct_to_entry_price_graal = metrics_calc.avg_profit_pct_to_entry_price_graal #1.0
    trades_in_year_graal = metrics_calc.trades_in_year_graal #0.6234
    calmar_koeff_pct_graal = metrics_calc.calmar_coeff_start_capital_graal #0.829
    months_plus_pct = metrics_calc.months_plus_pct
    quartals_plus_pct = metrics_calc.quartals_plus_pct
    days_plus_pct = metrics_calc.days_plus_pct
    months_plus_pct_graal = metrics_calc.months_plus_pct_graal #

    graal_metr_no_reinvest = metrics_calc.graal_metr_no_reinvest  # 89.8358
    graal_metr_with_reinvest = metrics_calc.graal_metr_with_reinvest  # 89.8358



    beard_coeff_daily_graal = metrics_calc.beard_coeff_daily_graal #0.8235



    recovery_and_sharp_graal = metrics_calc.recovery_and_sharp_graal  # 1,0334

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

    '''
    netprof_df = pd.DataFrame(metrics_series_dict['net_profit_pct_arr'])
    netprof_df.index = bars_df['Date_dt']
    netprof_df.to_csv(f'/Users/mishulil/PycharmProjects/backtestingGIT/trend_backtest/netprof_{_symbol}_pcatr_{tf}_290425.csv')
    '''
 #   metrics_values_dict['Graal_04'] = metrics_punkt.calc_Graal04(metrics_calc.trades_per_year, metrics_values_dict['RecoveryFactor'],metrics_values_dict['SharpeMonth'])

    #print(metrics_values_dict)

    #endregion

    #region Выводим результаты стратегии в консоль


    print(f' Данная комбинация параметров является хорошей')
    print(metrics_values_dict)

    metrics_current_strategy_df = pd.DataFrame([metrics_values_dict]) .T # Создание DataFrame из словаря
    print(metrics_current_strategy_df)

    #metrics_transposed_df = metrics_current_strategy_df.T
    #print (metrics_transposed_df)
    vizualization(metrics_series_dict, metrics_values_dict, bars_df)
    #endregion

#endregion

if __name__ == '__main__': #Если запустили локально - то локальная функция main
    main()


