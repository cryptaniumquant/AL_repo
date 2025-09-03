#region Импорт библиотек
import pandas as pd
import talib
import numpy as np
import matplotlib.pyplot as plt
import time
from typing import Tuple

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

import datetime

offset = datetime.timezone(datetime.timedelta(hours=3))
from pathlib import Path
from scipy import stats

#endregion

#region Grid indicators


def sma_calc(prices: pd.Series, period: int) -> pd.Series:
    """
    Рассчитывает простую скользящую среднюю (SMA).

    Параметры:
    prices (pd.Series): Ряд данных с ценами.
    period (int): Период для расчета SMA.

    Возвращает:
    pd.Series: Ряд данных с рассчитанной SMA.
    """
    # Преобразуем входные данные в объект pd.Series, если это еще не сделано
    prices_series = pd.Series(prices)

    # Рассчитываем скользящую среднюю с указанным периодом
    sma_series = prices_series.rolling(window=period).mean()

    return sma_series

def log_pro_calc(prices: pd.Series, period: int):
    """
    Рассчитывает Logarithmic Price Ratio Oscillator (log_pro).

    :param prices: Ряд цен (pandas Series). Индексом ряда данных должна быть временная шкала.
    :param period: Период для скользящей средней.
    :return: pandas Series с рассчитанным log_pro.
    """

    # Проверка корректности входных данных
    if not isinstance(prices, pd.Series):
        raise ValueError("prices должен быть pandas Series")

    if period <= 0:
        raise ValueError("period должен быть положительным числом")

    # Рассчитываем скользящую среднюю за период _bars
    moving_average = sma_calc(prices, period)

    # Находим отношение цены закрытия (close) к скользящей средней
    price_to_ma_ratio = prices / moving_average

    # Находим натуральный логарифм отношений с предыдущего шага
    log_pro = np.log(price_to_ma_ratio)

    return log_pro * 100.0

def calculate_percentile(period: int, percent: float):
    """Расчет квантиля заданного уровня с заданным периодом."""
    return np.percentile(period, percent)


def perc_confidence(values: pd.Series, period: int, percentile: float, confidence_level: float) -> pd.Series:
    """
    Эта функция вычисляет значения осциллятора для заданного перцентиля с определенным уровнем уверенности
    для каждой точки ряда, начиная с периода.

    Параметры:
    - values: Ряд данных (pandas Series)
    - period: Размер окна для расчета
    - percentile: Какой процент данных мы хотим "отрезать" снизу (от 0 до 100)
    - confidence_level: Насколько мы уверены в результате (от 50 до 100%)

    Возвращает:
    - pandas Series: новый ряд данных с вычисленными значениями осциллятора
    """

    # Проверяем, правильные ли данные нам дали
    if not (0 <= percentile <= 100):
        raise ValueError("Перцентиль должен быть между 0 и 100.")
    if not (50 <= confidence_level <= 100):
        raise ValueError("Уровень уверенности должен быть между 50 и 100.")
    if period <= 0:
        raise ValueError("Период должен быть положительным числом.")
    if len(values) < period:
        raise ValueError("У нас недостаточно данных для указанного периода.")

    # Создаем пустой список для хранения результатов
    results = []

    # Заполняем первые (period - 1) значений NaN, так как для них недостаточно данных
    results.extend([np.nan] * (period - 1))

    # Вычисляем z-score для заданного уровня уверенности
    z_score = stats.norm.ppf((1 + confidence_level / 100) / 2)

    # Проходим по всем точкам, начиная с period-й
    for i in range(period - 1, len(values)):
        # Берем окно данных
        window = values[i - period + 1: i + 1]

        # Находим значение перцентиля
        percentile_value = np.percentile(window, percentile)

        # Считаем стандартную ошибку среднего
        standard_error = stats.sem(window)

        # Вычисляем нижнюю и верхнюю границы нашего "уверенного диапазона"
        lower_bound = percentile_value - z_score * standard_error
        upper_bound = percentile_value + z_score * standard_error

        # Добавляем среднее значение между нижней и верхней границей в результаты
        results.append((lower_bound + upper_bound) / 2)

    # Возвращаем результаты в виде pandas Series
    return pd.Series(results, index=values.index)


def calculate_prices(oscillator_series, sma_series):
    """
    Вычисляет цены на основе рядов данных осцилляторов и скользящей средней (SMA).

    Параметры:
    oscillator_series (pd.Series): Ряд данных осцилляторов.
    sma_series (pd.Series): Ряд данных скользящей средней (SMA).

    Возвращает:
    pd.Series: Ряд данных цен, рассчитанных на основе осцилляторов и SMA.

    Исключения:
    ValueError: Если длины входных рядов не совпадают.
    """

    # Проверяем, что длины обоих рядов данных совпадают
    if len(oscillator_series) != len(sma_series):
        raise ValueError("Oscillator and SMA series must have the same length.")

    # Вычисляем цены по формуле: Price = SMA * exp(Oscillator / 100
    prices = sma_series * np.exp(oscillator_series / 100)

    # Возвращаем ряд данных цен
    return prices

def shadow(data: pd.Series, percentile: pd.Series, period_sma: int):
    """Расчет отображения квантилей асцелятора на цены закрытия."""
    return sma_calc(data, period_sma) / np.exp(percentile)


# endregion

#region fluger_sar - Индикатор "Флюгер Stop and Revers"
def fluger_sar(low, high, close, bars_for_extreme, steps, distance_increase_pct):
    """
    Индикатор "Флюгер Stop and Revers"

    :param low: Массив или Series минимальных цен
    :param high: Массив или Series максимальных цен
    :param close: Массив или Series цен закрытия
    :param bars_for_extreme: Количество баров для установки первоначального стопа
    :param steps: Количество шагов, за которое нужно дойти до текущей цены
    :param distance_increase_pct: На сколько процентов нужно увеличить (+) или сократить (-) дистанцию до первоначального стопа
    :return: Индикатор (Series) Флюгер
    """

    # Преобразуем входные данные в numpy массивы, если они не являются таковыми
    low = np.asarray(low)
    high = np.asarray(high)
    close = np.asarray(close)

    # Создаем пустой массив для индикатора
    fluger = np.zeros(len(low))

    # Первоначально говорим, что строим трейлинг для длинной позиции
    is_long_stop = True

    # Рассчитываем индикатор минимум за bars_for_extreme периодов
    lowest_low = pd.Series(low).rolling(window=bars_for_extreme, min_periods=1).min().values

    # Рассчитываем индикатор максимум за bars_for_extreme периодов
    highest_high = pd.Series(high).rolling(window=bars_for_extreme, min_periods=1).max().values

    # Начальные параметры
    distance_for_first_stop = 0.5 / 100.0
    koeff_distance_increase = 1.0 + distance_increase_pct / 100.0

    current_trailing_stop = 0

    for bar in range(len(low)):
        if bar == 0:  # На текущем баре
            if is_long_stop:  # Если ведём по длинной позиции
                distance_to_stop_punkt = low[bar] - lowest_low[bar] if bar > bars_for_extreme else (
                            close[bar] * distance_for_first_stop)
                current_trailing_stop = low[bar] - distance_to_stop_punkt * koeff_distance_increase
            else:  # Если ведём по короткой позиции
                distance_to_stop_punkt = highest_high[bar] - high[bar] if bar > bars_for_extreme else (
                            close[bar] * distance_for_first_stop)
                current_trailing_stop = high[bar] + distance_to_stop_punkt * koeff_distance_increase
        else:  # На всех последующих барах
            if is_long_stop:  # Если поддерживаем трейлинг для лонга
                if close[bar] > current_trailing_stop:  # Если текущий трейлинг не нарушился
                    step_width = (low[bar] - current_trailing_stop) / steps
                    current_trailing_stop = max(current_trailing_stop, current_trailing_stop + step_width)
                else:  # Если произошло пробитие вниз
                    is_long_stop = False  # Сообщаем, что для следующего бара начнём использовать трейлинг для шорта
                    distance_to_stop_punkt = highest_high[bar] - high[bar] if bar > bars_for_extreme else (
                                close[bar] * distance_for_first_stop)
                    current_trailing_stop = high[bar] + distance_to_stop_punkt * koeff_distance_increase
            else:  # Если поддерживаем трейлинг для шорта
                if close[bar] < current_trailing_stop:  # Если текущий трейлинг не нарушился
                    step_width = (current_trailing_stop - high[bar]) / steps
                    current_trailing_stop = min(current_trailing_stop, current_trailing_stop - step_width)
                else:  # Если произошло пробитие вверх
                    is_long_stop = True  # Сообщаем, что для следующего бара начнём использовать трейлинг для лонга
                    distance_to_stop_punkt = low[bar] - lowest_low[bar] if bar > bars_for_extreme else (
                                close[bar] * distance_for_first_stop)
                    current_trailing_stop = low[bar] - distance_to_stop_punkt * koeff_distance_increase

        fluger[bar] = current_trailing_stop

    return pd.Series(fluger)

#endregion

#region calculate_Channels Рассчитываем каналы сразу и верхний и нижний
def calculate_Channels(df_high: pd.Series, df_low: pd.Series, period: int) -> Tuple[pd.Series, pd.Series]:
    """
    Функция расчета ценовых каналов по максимуму и минимуму свечей
    """
    highLevelEnter = df_high.rolling(window=period).max()
    lowLevelEnter = df_low.rolling(window=period).min()
    return highLevelEnter, lowLevelEnter
#endregion

#region Индикатор Highest (вернхний ценовой канал)
def Highest(df_prices, period: int) -> pd.Series:
    """
    Функция расчета верхнего ценового канала.

    Параметры:
    df_prices (pd.Series или np.ndarray): Временной ряд с ценами.
    period (int): Период для расчета максимума.

    Возвращает:
    pd.Series: Временной ряд с вычисленным верхним ценовым каналом.
    """

    if isinstance(df_prices, np.ndarray): # Проверяем, является ли df_prices массивом numpy

        df_prices = pd.Series(df_prices) # Если да, преобразуем его в pandas Series
    # Если df_prices не является ни pandas Series, ни numpy массивом
    elif not isinstance(df_prices, pd.Series):
        # Выбрасываем исключение с указанием допустимых типов данных
        raise TypeError("df_prices должен быть либо pd.Series, либо np.ndarray")

    if not isinstance(period, int) or period <= 0:
        raise ValueError("period must be a positive integer")

    # Рассчитываем максимум за заданный период для каждого момента времени
    highLevel = df_prices.rolling(window=period).max()

    # Возвращаем результат в виде временного ряда (pandas Series)
    return highLevel
#endregion

#region Индикатор Lowest (нижний ценовой канал)
def Lowest(df_prices, period: int) -> pd.Series:
    """
    Функция расчета нижнего ценового канала.

    Параметры:
    df_prices (pd.Series или np.ndarray): Временной ряд с ценами.
    period (int): Период для расчета минимума.

    Возвращает:
    pd.Series: Временной ряд с вычисленным нижним ценовым каналом.
    """
    # Проверяем, является ли df_prices массивом numpy
    if isinstance(df_prices, np.ndarray):
        # Если да, преобразуем его в pandas Series
        df_prices = pd.Series(df_prices)
    # Если df_prices не является ни pandas Series, ни numpy массивом
    elif not isinstance(df_prices, pd.Series):
        # Выбрасываем исключение с указанием допустимых типов данных
        raise TypeError("df_prices должен быть либо pd.Series, либо np.ndarray")

    if not isinstance(period, int) or period <= 0:
        raise ValueError("period must be a positive integer")

    # Рассчитываем минимум за заданный период для каждого момента времени
    lowLevel = df_prices.rolling(window=period).min()

    # Возвращаем результат в виде временного ряда (pandas Series)
    return lowLevel

#endregion

#region ATR_calc (talib)
def ATR_calc(df_close: pd.Series, df_high: pd.Series, df_low: pd.Series, periodAtr: int) -> pd.Series:
    """
    Функция расчета ATR
    """
    atrSeries = talib.ATR(high=df_high, low=df_low, close=df_close, timeperiod=periodAtr)

    return atrSeries
#endregion

#region ATR_tslab_calc: ATR - рассчитывается по алгоритму, который принят в ТСЛаб (ATR_New)
def ATR_tslab_calc(df_close, df_high, df_low, periodAtr: int) -> pd.Series:
    """
    Функция расчета ATR с проверкой типов данных.

    Аргументы:
    df_close -- ряд данных закрытия (pd.Series или np.ndarray)
    df_high -- ряд данных максимума (pd.Series или np.ndarray)
    df_low -- ряд данных минимума (pd.Series или np.ndarray)
    periodAtr -- период для расчета ATR (int)

    Возвращает:
    pd.Series -- рассчитанный ATR.
    """

    # Проверка и преобразование типов данных
    if isinstance(df_close, np.ndarray):
        df_close = pd.Series(df_close)
    if isinstance(df_high, np.ndarray):
        df_high = pd.Series(df_high)
    if isinstance(df_low, np.ndarray):
        df_low = pd.Series(df_low)

    # Проверка, что все входные данные теперь pd.Series
    if not all(isinstance(x, pd.Series) for x in [df_close, df_high, df_low]):
        raise ValueError("Все входные данные должны быть pd.Series или np.ndarray")

    # Создаем DataFrame для расчетов
    df_atr = pd.DataFrame()

    # Рассчитываем True Range (TR)
    df_atr['H-L'] = df_high - df_low  # Разница между максимумом и минимумом
    df_atr['H-PC'] = abs(df_high - df_close.shift(1))  # Разница между максимумом и предыдущим закрытием
    df_atr['L-PC'] = abs(df_low - df_close.shift(1))  # Разница между минимумом и предыдущим закрытием
    df_atr['TR'] = df_atr[['H-L', 'H-PC', 'L-PC']].max(axis=1)  # Находим максимальное значение для TR

    # Рассчитываем ATR на основе TR
    df_atr['ATR'] = df_atr['TR'].rolling(window=periodAtr).mean()  # Среднее значение TR за заданный период
    return df_atr['ATR']  # Возвращаем рассчитанный ATR

#endregion


#new
def calculate_SMA(df_close: pd.Series, periodAtr: int) -> pd.Series:
    """
    Функция расчета SMA
    """
    SMAseries = talib.SMA(df_close, timeperiod=periodAtr)
    return SMAseries

#new
def calculate_EMA(df_close: pd.Series, periodAtr: int) -> pd.Series:
    """
    Функция расчета EMA
    """
    EMAseries = talib.EMA(df_close, timeperiod=periodAtr)
    return EMAseries

#new
def calculate_BBANDS(df_close: pd.Series, periodAtr: int) -> Tuple[pd.Series, pd.Series]:
    """
    Функция расчета Bollinger Bands
    """
    BBANDSseries = talib.BBANDS(df_close, timeperiod=periodAtr)
    BBANDSseriesHigh = BBANDSseries[0]
    SMAseries = BBANDSseries[1]
    BBANDSseriesLow = BBANDSseries[2]
    return BBANDSseriesHigh, SMAseries, BBANDSseriesLow


def read_data(filename: str) -> pd.DataFrame:

    """
    Функция для подгрузки данных
    """

    col_names = ['Open Time',
                 'Open',
                 'High',
                 'Low',
                 'Close',
                 'Volume',
                 'Close Time',
                 'Quote asset volume',
                 'Num of trades',
                 'Taker buy base',
                 'Taker buy quote',
                 'Ignore']
    dataframe = pd.read_csv(filename, names=col_names)
    dataframe = dataframe[['Close Time', 'Open', 'High', 'Low', 'Close']]
    dataframe = dataframe.set_index(pd.to_datetime(dataframe['Close Time'], unit='ms'), drop=True)
    dataframe = dataframe.drop(['Close Time'], axis=1)
    return dataframe
