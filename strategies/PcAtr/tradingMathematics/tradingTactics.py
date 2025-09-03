#region Описание Торговых тактик
'''
"Trading tactics" (торговые тактики) — это конкретные методы и подходы,
которые трейдеры используют для реализации своей общей торговой стратегии.
Эти тактики включают в себя различные аспекты торговли, такие как:

1. **Техника входа в позицию** (Entry tactics): Определение точек входа в рынок на основе различных индикаторов, сигналов или условий.
2. **Техника выхода из позиции** (Exit tactics): Определение точек выхода из рынка для фиксации прибыли или минимизации убытков.
3. **Установка стоп-лоссов** (Stop-loss tactics): Определение уровней, на которых автоматически закрывается позиция для ограничения убытков.
4. **Трейлинг-стопы** (Trailing stop tactics): Динамическое перемещение стоп-лосса в зависимости от движения цены для защиты прибыли.

Торговые тактики являются частью общей торговой стратегии и помогают трейдеру эффективно управлять своими сделками,
минимизировать риски и максимизировать прибыль.
'''
#endregion

#region Импорт библиотек
import math
import pandas as pd
import talib
import numpy as np
import matplotlib.pyplot as plt
import time
from math import ceil
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

import datetime

offset = datetime.timezone(datetime.timedelta(hours=3))
from pathlib import Path
#endregion

#region stopPriceLong (на основе расстояния между каналами и коэффициента)
def stopPriceLong_calc(highLevel: pd.Series, lowLevel: pd.Series, koeff: float) -> pd.Series:
    """
    Рассчитываем первоначальный уровень стоп-лосса исходя из расстояния между каналами

    Parameters:
    highLevel (pd.Series): Ряд данных с высокими уровнями.
    lowLevel (pd.Series): Ряд данных с низкими уровнями.
    koeff (float): Коэффициент для вычисления уровня стоп-лосса.

    Returns:
    pd.Series: Ряд данных с рассчитанным уровнем стоп-лосса.
    """
    # Проверяем и преобразуем входные данные, если это необходимо
    if isinstance(highLevel, np.ndarray):
        highLevel = pd.Series(highLevel)
    if isinstance(lowLevel, np.ndarray):
        lowLevel = pd.Series(lowLevel)

    # Убедимся, что входные данные все еще являются pd.Series
    if not isinstance(highLevel, pd.Series) or not isinstance(lowLevel, pd.Series):
        raise ValueError("Both highLevel and lowLevel must be either pd.Series or np.ndarray")

    # Проверяем, что koeff это число и находится в диапазоне от 0 до 10
    if not isinstance(koeff, (int, float)) or not (koeff >= 0):
        raise ValueError("koeff must be a float between 0 and 1")

    # Рассчитываем первоначальный уровень стоп-лосса
    stopPriceLong = highLevel - (highLevel - lowLevel) * koeff

    return stopPriceLong

#endregion

#region stopPriceShort (на основе расстояния между каналами и коэффициента)
def stopPriceShort_calc(highLevel: pd.Series, lowLevel: pd.Series, koeff: float) -> pd.Series:
    """
    Рассчитываем первоначальный уровень стоп-лосса для шорта исходя из расстояния между каналами

    Parameters:
    highLevel (pd.Series): Ряд данных с высокими уровнями.
    lowLevel (pd.Series): Ряд данных с низкими уровнями.
    koeff (float): Коэффициент для вычисления уровня стоп-лосса.

    Returns:
    pd.Series: Ряд данных с рассчитанным уровнем стоп-лосса.
    """
    # Проверяем и преобразуем входные данные, если это необходимо
    if isinstance(highLevel, np.ndarray):
        highLevel = pd.Series(highLevel)
    if isinstance(lowLevel, np.ndarray):
        lowLevel = pd.Series(lowLevel)

    # Убедимся, что входные данные все еще являются pd.Series
    if not isinstance(highLevel, pd.Series) or not isinstance(lowLevel, pd.Series):
        raise ValueError("Both highLevel and lowLevel must be either pd.Series or np.ndarray")

    # Проверяем, что koeff это число и находится в диапазоне от 0 до 1
    if not isinstance(koeff, (int, float)) or not (koeff >= 0):
        raise ValueError("koeff must be a float between 0 and 1")

    # Рассчитываем первоначальный уровень стоп-лосса
    stopPriceShort = lowLevel + (highLevel - lowLevel) * koeff

    return stopPriceShort

#endregion

#region trailingStopLong (На основе ATR и делителя)
def trailingStopLong_calc(open_price: float, close_price: float,
                          previous_trailingStopLong: float, cur_atrValue: float,
                          divider: float) -> float:
    """
    Рассчитываем текущий трейлинг-стоп опираясь на значение трейлинг стопа на предыдущем баре,
    исходя из текущего значения ATR и делителя
    """
    mustMoveTrailingLong = close_price > open_price
    if mustMoveTrailingLong: #Если свеча белая
        trailingStopLongNew = previous_trailingStopLong + (cur_atrValue / divider)
        # Новый трейлинг-стоп не может опустится ниже предыдущего по принципу "ни шагу назад"
        return max(trailingStopLongNew, previous_trailingStopLong)

    return previous_trailingStopLong
#endregion

#region trailingStopShort (на основе ATR и делителя)
def trailingStopShort_calc(open_price: float, close_price: float,
                           previous_trailingStopShort: float,
                           cur_atrValue: float, divider: float)-> float:
    """
    Рассчитываем текущий трейлинг-стоп для шорта с учетом
    предыдущего уровня трейлинг - стопа, учитывая величину ATR и делителя
    """
    mustMoveTrailingShort = close_price < open_price

    if mustMoveTrailingShort:
        trailingStopShortNew = previous_trailingStopShort - (cur_atrValue / divider)

        # Новый трейлинг-стоп не может быть выше предыдущего
        return min(trailingStopShortNew, previous_trailingStopShort)

    return previous_trailingStopShort
#endregion

#region AtrTrailing: по цене входа Atr и делителю определяем текущее значенние трейлинг Стопа (и для лонга и для шорта)
def AtrTrailing(LongMode, MoveTrailing, CurrentTrailing, atrValue, dividerAtr):
    """
    Трейлинг стоп используя ATR

    :param LongMode: Если (True) - сопровождаем Long. Если (False) - сопровождаем шорт
    :param MoveTrailing: Нужно ли изменить текущий трейлинг
    :param CurrentTrailing: Величина текущего трейлинга
    :param atrValue: Текущее значение ATR в пунктах
    :param dividerAtr: На сколько частей делим ATR для определения ширины шага
    :return: Текущий трейлинг
    """
    result = 0  # искомый индикатор

    stepWidth = atrValue / dividerAtr  # определяем ширину шага

    if LongMode:  # если в режиме сопровождения лонга
        if MoveTrailing:  # Если трейлинг нужно подвинуть
            result = CurrentTrailing + stepWidth
        else:  # Если трейлинг двигать не нужно
            result = CurrentTrailing

        result = max(CurrentTrailing, result)  # Ни шагу назад
    else:  # Если в режиме сопровождения шорта
        if MoveTrailing:  # Если трейлинг нужно подвинуть
            result = CurrentTrailing - stepWidth
        else:  # Если трейлинг двигать не нужно
            result = CurrentTrailing

        result = min(CurrentTrailing, result)  # Ни шагу назад

    return result
#endregion

#region RoundPrice - Округляет заданное число до ближайшего значения, кратного stepPrice.
def RoundPrice(price: float, stepPrice: float) -> float:
    """
        Округляет заданное число до ближайшего значения, кратного stepPrice.

        Параметры:
        price (float): Исходное число, которое требуется округлить.
        stepPrice (float): Шаг округления, до кратности которого будет производиться округление.

        Возвращает:
        float: Округленное значение, кратное stepPrice.

        Исключения:
        ValueError: Если stepPrice равен нулю.

    """


    # Проверка на случай, если stepPrice равен нулю, чтобы избежать деления на ноль
    if stepPrice == 0:
        raise ValueError("stepPrice должен быть больше нуля")

    # Округление числа до ближайшего значения, кратного stepPrice
    rounded_price = round(price / stepPrice) * stepPrice
    return rounded_price
#endregion

#region AbsCommission_calc - Функция для расчета абсолютной комиссии за лот.
def AbsCommissionPerLot_calc(lot_size, abs_commission_per_lot):
    """
    Функция для расчета абсолютной комиссии за лот.

    :param lot_size: Величина биржевой сделки в лотах.
    :param abs_commission_per_lot: Абсолютная комиссия за один лот.
    :return: Величина комиссии, которую нужно заплатить за сделку.
    
    Пример использования функции:
    lot_size = 10  # Пример: 10 лотов
    abs_commission_per_lot = 5.0  # Пример: 5 рублей за один лот
    
    >>>commission_to_pay = AbsCommissionPerLot_calc(lot_size, abs_commission_per_lot)
    
    """
    commission = lot_size * abs_commission_per_lot
    return commission
#endregion

#region RelCommission_calc - Функция для расчета относительной комиссии за биржевую сделку.
def RelCommission_calc(lot_size, entry_price, commission_pct):
    """
    Функция для расчета относительной комиссии за биржевую сделку.

    :param lot_size: Величина биржевой сделки в лотах.
    :param entry_price: Цена входа в сделку.
    :param commission_pct: Комиссия в процентах от объема сделки.
    :return: Величина комиссии, которую нужно заплатить.
    """
    # Вычисляем объем сделки в денежном выражении
    trade_volume = lot_size * entry_price

    # Вычисляем величину комиссии
    commission = trade_volume * (commission_pct / 100.0)

    return commission
#endregion

#region DecimalPlacesCount() - Возвращает количество знаков после запятой (максимальное кол-во 15) у указанного числа
def DecimalPlacesCount(value):
    """
    Возвращает количество знаков после запятой (максимальное кол-во 15) у указанного числа

    :param value: Число типа float
    :returns: Количество знаков после запятой (не больше 15) у указанного числа
    """
    value_double_round_15 = round(value, 15)
    value_int = int(value)
    value_zero = value_double_round_15 - value_int  # оставляем остаток после нуля

    for i in range(15):
        if (value_zero * math.pow(10, 1 + i)) % 10 == 0:  # Если остаток от деления равен нулю
            return i  # именно столько знаков после запятой в введённом числе

    # Если внутри цикла не нашелся результат - говорим, что у числа 15 знаков после запятой
    return 15
#endregion

# region minPosSizeBinanceSpot - рассчитать минимально возможный размер позиции для Бинанс Спот (с учётом мин. по крипте и мин. по стоимости в $)

def MinPosSizeBinanceSpot(target_entry_price, min_lot_step_crypta, min_lot_price_usd):
    """
    Метод определения минимально возможного размера позиции для Бинанс Спот (с учётом мин. по крипте и мин. по стоимости в $)

    :param target_entry_price: Ожидаемая цена входа в позицию
    :param min_lot_step_crypta: величина минимального лота в единицах первой пары (0,001 для BTC)
    :param min_lot_price_usd: величина минимального лота в USD
    :return: Кол-во лотов - минимально возможное для Бинанс Спот с учётом LotStep и minLotPriceUsd
    """

    value_round_for_lots = DecimalPlacesCount(min_lot_step_crypta)  # знаков после запятой в лоте

    # Если нет ограничения по минимальной стоимости лота, то минимальным лотом является минимальный шаг по лоту
    if min_lot_price_usd <= 0:
        return min_lot_step_crypta

    min_pos_size = min_lot_step_crypta  # изначально равен минимальному шагу изменения лота (Symbol.LotStep)
    min_step_price = min_lot_step_crypta * target_entry_price  # стоимость минимального шага изменения лота (LotStep) в $
    count_of_min_lot_steps = math.floor(
        min_lot_price_usd / min_step_price) + 1  # количество шагов (с запасом в один шаг)
    min_pos_size = count_of_min_lot_steps * min_lot_step_crypta  # Размер минимального лота
    min_pos_size = round(min_pos_size, value_round_for_lots)  # делаем округление до нужного размера
    min_pos_size = max(min_pos_size, min_lot_step_crypta)  # должен быть больше минимального в крипте

    return min_pos_size  # Результат, который выдаёт этот метод

#endregion

#region MaxPctRiskBinance - максимальный размер риска в сделке (на бинансе)
def MaxPctRiskBinance(
        SummForSystem,  # 50 - Сумма, выделяемая системе
        maxPctRisk,  # 1.5 % риска на одну сделку
        TargetEntryPrice,  # 9100 - Ожидаемая цена входа в позицию
        StartStopLoss,  # 9050 - Первоначальный Стоп-Лосс
        minLotSizeCrypta,  # 0.001 - величина минимального лота в единицах первой пары (0,001 для BTC)
        minLotSizeUsd,  # 0 величина минимального лота в USD
        maxCountOfMinLot,  # 100 Максимальное кол-во минимальных лотов
        TradeMinContract  # False торговать минимально возможным контрактом?
    ):
    """
    Вычисляет максимальный размер позиции (лот) для торговли на Binance с заданным процентом риска.

    :param SummForSystem: float - Сумма, выделяемая системе
    :param maxPctRisk: float - Процент риска на одну сделку
    :param TargetEntryPrice: float - Ожидаемая цена входа в позицию
    :param StartStopLoss: float - Первоначальный Стоп-Лосс
    :param minLotSizeCrypta: float - Величина минимального лота в единицах первой пары (например, 0.001 для BTC)
    :param minLotSizeUsd: float - Величина минимального лота в USD
    :param maxCountOfMinLot: int - Максимальное количество минимальных лотов
    :param TradeMinContract: bool - Торговать минимально возможным контрактом?
    :return: float - Максимальный размер позиции (количество лотов)
    """
    # Получаем количество знаков после запятой для округления лота
    DecimalPlacesForRounding = DecimalPlacesCount(minLotSizeCrypta)  # 3 - знаков после запятой в лоте

    # Объявляем переменные
    ResultLotSize = 0  # Искомое количество лотов, которое будем ставить в заявку
    RiskForOneMinLot = 0  # Риск потери на один минимальный лот в долларах
    RiskForOneTrade = 0  # Можно потерять в одной сделке не больше чем ... долларов

    # Если указано торговать одним контрактом, возвращаем минимально возможный размер позиции
    if TradeMinContract:
        return MinPosSizeBinanceSpot(TargetEntryPrice, minLotSizeCrypta, minLotSizeUsd)

    # Определяем, сколько долларов мы готовы потерять в одной сделке
    RiskForOneTrade = SummForSystem * maxPctRisk / 100.0  # 50 * 1.5 / 100 = 0.75

    # Определяем риск на минимально возможный лот в долларах
    if TargetEntryPrice > StartStopLoss:  # Если позиция длинная
        giveUSDTBuying = minLotSizeCrypta * TargetEntryPrice  # 0.001 * 9100 = 9.1 - Отдадим долларов при покупке минимального лота по цене входа
        takeUSDTSelling = minLotSizeCrypta * StartStopLoss  # 0.001 * 9050 = 9.05 - Получим долларов при продаже минимального лота по стопу
        RiskForOneMinLot = takeUSDTSelling - giveUSDTBuying  # 9.05 - 9.1 = -0.05 = Потеряем долларов при срабатывании стопа (отрицательное число)
    else:  # Если позиция короткая
        takeUSDTSelling = minLotSizeCrypta * TargetEntryPrice  # Получим долларов при продаже минимального лота по цене входа
        giveUSDTBuying = minLotSizeCrypta * StartStopLoss  # Отдадим долларов при покупке минимального лота по стопу
        RiskForOneMinLot = takeUSDTSelling - giveUSDTBuying  # Потеряем долларов при срабатывании стопа (отрицательное число)

    # Если риск на один минимальный лот положительный, возвращаем 0 (некорректный стоп)
    if RiskForOneMinLot > 0:
        return 0

    # Определяем количество контрактов с дробной частью
    ResultLotSize = RiskForOneTrade / (-1.0 * RiskForOneMinLot) * minLotSizeCrypta  # 0.75 / (0.05 * 0.001)
    # Округляем до нужного размера (до 3-го знака)
    ResultLotSize = round(ResultLotSize, DecimalPlacesForRounding)
    # Ограничиваем количество лотов для торговли
    ResultLotSize = min(ResultLotSize, maxCountOfMinLot * minLotSizeCrypta)

    # Если стоимость размера лота меньше, чем минимальная стоимость в долларах, возвращаем 0
    if ResultLotSize * TargetEntryPrice < minLotSizeUsd:
        ResultLotSize = 0

    return ResultLotSize  # Результат, который выдаёт этот метод

#endregion

# region pctOfEquityBinance - % от эквити (на бинансе)
def pctOfEquityBinance(
        summForSystem,  # 500 - Сумма, выделяемая системе
        pctOfEquity,  # 65 % от суммы
        targetEntryPrice,  # 9100 - Ожидаемая цена входа в позицию
        minLotSizeCrypta,  # 0.001 - величина минимального лота в единицах первой пары (0,001 для BTC)
        minLotSizeUsd=0,  # 0 величина минимального лота в USD
        maxCountOfMinLot=float('inf'),  # 100 Максимальное кол-во минимальных лотов
        tradeMinContract=False  # False торговать минимально возможным контрактом?
    ):
    """
    Вычисляет количество лотов для торговли на Binance, исходя из процента от эквити.

    :param summForSystem: float - Сумма, выделяемая системе
    :param pctOfEquity: float - Процент от суммы
    :param targetEntryPrice: float - Ожидаемая цена входа в позицию
    :param minLotSizeCrypta: float - Величина минимального лота в единицах первой пары (например, 0.001 для BTC)
    :param minLotSizeUsd: float - Величина минимального лота в USD
    :param maxCountOfMinLot: int - Максимальное количество минимальных лотов
    :param tradeMinContract: bool - Торговать минимально возможным контрактом?
    :return: float - Количество лотов
    """
    # Получаем количество знаков после запятой для округления лота
    decimalPlacesForRounding = DecimalPlacesCount(minLotSizeCrypta)  # 3 - знаков после запятой в лоте

    # Если указано торговать одним контрактом, возвращаем минимально возможный размер позиции
    if tradeMinContract:
        return MinPosSizeBinanceSpot(targetEntryPrice, minLotSizeCrypta, minLotSizeUsd)

    # Вычисляем количество лотов
    resultLotSize = ((summForSystem * pctOfEquity) / 100.0) / targetEntryPrice  # ( (500 * 65) / 100) / 9100 = 325 / 9100
    resultLotSize = round(resultLotSize, decimalPlacesForRounding)  # Округляем до нужного размера (до 3-го знака)
    resultLotSize = min(resultLotSize, maxCountOfMinLot * minLotSizeCrypta)  # Ограничиваем количество лотов для торговли

    # Если стоимость размера лота меньше, чем минимальная стоимость в долларах, возвращаем 0
    if resultLotSize * targetEntryPrice < minLotSizeUsd:
        resultLotSize = 0

    return resultLotSize  # Результат, который выдаёт этот метод

# endregion


#region RoundToMinLotStep: Округляет размер лота до минимального шага изменения лота
def RoundToMinLotStep(lotSizeCrypta, minLotStepCrypta):
    """
    Округляет размер лота до минимального шага изменения лота (без учёта знаков после запятой)

    :param lotSizeCrypta: Исходная величина лота
    :param minLotStepCrypta: Минимальный шаг изменения размера лота
    :return: Искомый (округлённый до минимального шага изменения) лот
    """
    countOfMinSteps = lotSizeCrypta / minLotStepCrypta  # узнаём количество минимальных шагов для лота
    countOfMinSteps = math.trunc(countOfMinSteps)  # округляем количество шагов до целого числа
    result = countOfMinSteps * minLotStepCrypta  # Определяем размер лота

    return result
#endregion

#region PositionProfitCalc Функция для расчета текущего финансового результата от Long и Short позиций на бирже с учетом комиссии.
def position_profit_calc(isLongPos: bool, entryPrice: float, currentPrice: float, lots: float, feesPct: float):
    """
    Функция для расчета текущего финансового результата от позиции на бирже с учетом комиссии.

    :param isLong: Флаг указывающий на тип позиции: True для длинной, False для короткой (bool)
    :param entryPrice: Цена входа в позицию (float)
    :param currentPrice: Текущая цена (float)
    :param lots: Количество лотов в позиции (float)
    :param feesPct: Комиссия в % от объёма сделки (float)
    :return: Текущий финансовый результат (float)

    Пример использования функции
    entryPrice = 100.0  # Цена входа
    currentPrice = 110.0  # Текущая цена
    lots = 10  # Количество лотов
    feesPct = 0.5  # Комиссия в %
    isLong = True  # Длинная позиция

    >>> profit = position_profit_calc(_isLongPos, _entryPrice, _currentPrice, _lots, _feesPct, _isLongPos)

    """
    _isLongPos = bool(isLongPos)
    _entryPrice = float(entryPrice)
    _currentPrice = float(currentPrice)
    _lots = float(lots)
    _feesPct = float(feesPct)

    if _isLongPos: # Длинная позиция
        profit_per_lot = _currentPrice - _entryPrice  # Расчет финансового результата от одного лота
    else: # Короткая позиция
        profit_per_lot = _entryPrice - _currentPrice  # Расчет финансового результата от одного лота

    total_profit = profit_per_lot * abs(_lots)  # Общий финансовый результат от всех лотов
    totalEnter_volume = _entryPrice * abs(_lots)  # Общий объем сделки на покупку
    totalExit_volume = _currentPrice * abs(_lots)  # Общий объем сделки на продажу
    total_volume = totalEnter_volume + totalExit_volume
    total_fees = total_volume * (_feesPct / 100)  # Расчет общей комиссии
    net_profit = total_profit - total_fees  # Итоговый финансовый результат с учетом комиссии

    return net_profit

#endregion
