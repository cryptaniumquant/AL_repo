import talib
import numpy as np


def SMA(x, period):
    # SMA
    return talib.SMA(real = x, timeperiod = period)

def ATR(close, high, low, period=14):
    """
    Вычисляет Average True Range (ATR) используя NumPy
    
    Параметры:
        close: np.array - массив цен закрытия
        high: np.array - массив максимальных цен
        low: np.array - массив минимальных цен
        period: int - период для расчета ATR (по умолчанию 14)
        
    Возвращает:
        np.array - значения ATR
    """
    # Преобразуем входные данные в numpy массивы
    close = np.asarray(close, dtype=np.float64)
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    
    # Вычисляем True Range (TR)
    prev_close = np.roll(close, 1)
    prev_close[0] = np.nan  # Первое значение не определено
    
    tr1 = high - low
    tr2 = np.abs(high - prev_close)
    tr3 = np.abs(low - prev_close)
    
    tr = np.maximum.reduce([tr1, tr2, tr3])
    
    # Вычисляем ATR как SMA от TR
    atr = np.full_like(tr, np.nan)
    
    # Простое скользящее среднее (SMA)
    for i in range(period-1, len(tr)):
        if np.isnan(tr[i]):
            continue
        atr[i] = np.mean(tr[i-period+1:i+1])
    
    return atr

def EMA(x, period):
    """
    Расчет EMA с помощью NumPy (рекурсивная формула)
    
    Args:
        x: массив цен (numpy array)
        period: период EMA
    
    Returns:
        numpy array с значениями EMA
    """
    if len(x) < period:
        return np.full(len(x), np.nan)
    
    # Коэффициент сглаживания
    alpha = 2 / (period + 1)
    
    # Инициализация массива результатов
    ema = np.full(len(x), np.nan)
    
    # Первое значение EMA - простое среднее
    ema[period - 1] = np.mean(x[:period])
    
    # Рекурсивный расчет остальных значений
    for i in range(period, len(x)):
        ema[i] = alpha * x[i] + (1 - alpha) * ema[i - 1]
    
    return ema

import numpy as np
import pandas as pd

def calculate_vwap(high, low, close, volume, window=None):
    """
    Рассчитывает Volume Weighted Average Price (VWAP)
    
    Parameters:
    high : array-like
        Массив цен high за каждый период
    low : array-like
        Массив цен low за каждый период  
    close : array-like
        Массив цен close за каждый период
    volume : array-like
        Массив объемов за каждый период
    window : int, optional
        Размер окна для скользящего VWAP. Если None, рассчитывается кумулятивный VWAP
    
    Returns:
    numpy.ndarray
        Массив значений VWAP
    """
    # Преобразуем входные данные в numpy arrays
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)
    volume = np.asarray(volume, dtype=np.float64)
    
    # Рассчитываем типичную цену (Typical Price)
    typical_price = (high + low + close) / 3.0
    
    # Рассчитываем объем-взвешенную типичную цену
    price_volume = typical_price * volume
    
    if window is None:
        # Кумулятивный VWAP (стандартный)
        cumulative_pv = np.cumsum(price_volume)
        cumulative_volume = np.cumsum(volume)
        vwap = cumulative_pv / cumulative_volume
    else:
        # Скользящий VWAP с окном
        vwap = np.zeros_like(typical_price)
        cumulative_pv = np.cumsum(price_volume)
        cumulative_volume = np.cumsum(volume)
        
        for i in range(len(typical_price)):
            if i < window:
                # Для первых window элементов используем кумулятивную сумму
                vwap[i] = cumulative_pv[i] / cumulative_volume[i]
            else:
                # Для остальных - скользящее окно
                window_pv = cumulative_pv[i] - cumulative_pv[i - window]
                window_volume = cumulative_volume[i] - cumulative_volume[i - window]
                vwap[i] = window_pv / window_volume
    
    return vwap

# Альтернативная версия с использованием pandas для более эффективного расчета
def VWAP_indicator(high, low, close, volume, window=None):
    """
    Рассчитывает VWAP с использованием pandas для более эффективного расчета
    
    Parameters:
    high : array-like
        Массив цен high за каждый период
    low : array-like
        Массив цен low за каждый период  
    close : array-like
        Массив цен close за каждый период
    volume : array-like
        Массив объемов за каждый период
    window : int, optional
        Размер окна для скользящего VWAP. Если None, рассчитывается кумулятивный VWAP
    
    Returns:
    numpy.ndarray
        Массив значений VWAP
    """
    # Создаем DataFrame для удобства расчетов
    df = pd.DataFrame({
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    })
    
    # Рассчитываем типичную цену
    df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3.0
    
    # Рассчитываем произведение цены на объем
    df['price_volume'] = df['typical_price'] * df['volume']
    
    if window is None:
        # Кумулятивный VWAP
        df['cumulative_pv'] = df['price_volume'].cumsum()
        df['cumulative_volume'] = df['volume'].cumsum()
        df['vwap'] = df['cumulative_pv'] / df['cumulative_volume']
    else:
        # Скользящий VWAP
        df['cumulative_pv'] = df['price_volume'].rolling(window=window, min_periods=1).sum()
        df['cumulative_volume'] = df['volume'].rolling(window=window, min_periods=1).sum()
        df['vwap'] = df['cumulative_pv'] / df['cumulative_volume']
    
    return df['vwap'].values

