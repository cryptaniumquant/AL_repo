#region Импорт библиотек
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from functools import reduce
np.set_printoptions(threshold=1000)
#endregion


#region #NetProfitPct Рассчитывает NetProfitPct с помощью NumPy.
def calc_NetProfitPct(arr_pnl: np.array):
    """
    Рассчитывает NetProfitPct с помощью NumPy.

    Args:
      arr_pnl: Массив NumPy с PnL в виде float.

    Returns:
      Значение NetProfitPct.
    """
    return arr_pnl[-1]

def calc_NetProfitPunkt(arr_pnl_punkt: np.array):
    """
    Рассчитывает NetProfitPunkt с помощью NumPy.

    Args:
      arr_pnl: Массив NumPy с PnL в виде float.

    Returns:
      Значение NetProfitPunkt.
    """
    return(abs(arr_pnl_punkt[arr_pnl_punkt.size-1] - arr_pnl_punkt[0]))

#endregion

#region MaxDrawDawnPctAll
def calc_MaxDrawDawnPctAll(arr_pnl: np.array):
    """
    Вычисляем MaxDrawDawnPunktAll

    Args:
      arr_pnl: Массив NumPy с PnL в виде float.

    Returns:
      MaxDrawDawnPunktAll
    """
    max_drawdown = np.max(np.maximum.accumulate(arr_pnl) - arr_pnl)
    return max_drawdown
#endregion

#region MaxDrawDawnPctAllFromEquity
def calc_MaxDrawDawnPctAllFromEquity(arr_pnl: np.array):
    """
    Вычисляем MaxDrawDawnPctAll

    Args:
      arr_pnl: Массив NumPy с PnL в виде float.

    Returns:
      MaxDrawDawnPctAll
    """
    return np.nanmin((arr_pnl - np.maximum.accumulate(arr_pnl)) / (1+(np.maximum.accumulate(arr_pnl)/100)))
#endregion

#region calc_DrawDawnCurve
def calc_DrawDawnCurve(arr_pnl: np.array):
    """
    Вычисляем DrawDawnCurve

    Args:
      arr_pnl: Массив NumPy с PnL в виде float.

    Returns:
      DrawDawnCurve
    """
    return np.maximum.accumulate(arr_pnl) - arr_pnl

#endregion

#region BeardBarsMax
def calc_BeardBarsMax(arr_pnl: np.array):
    """
    Вычисляем BeardBarsMax

    Args:
      arr_pnl: Массив NumPy с PnL в виде float.

    Returns:
      BeardBarsMax
    """

    # Находим просадки (drawdowns)
    high_watermark = np.maximum.accumulate(arr_pnl)
    drawdowns = arr_pnl - high_watermark

    # Находим продолжительность самой длительной просадки
    ones_indices = np.where(drawdowns == 0)[0]
    # Используем numpy.diff для нахождения разницы между индексами
    diff_indices = np.diff(ones_indices)
    # Находим максимальную длину последовательности единиц
    max_length = max(diff_indices, default=len(arr_pnl)) + 1

    return max_length

#endregion

#region BeardKoeff
def calc_BeardKoeff(arr_pnl: np.array):
    """
    Вычисляем BeardKoeff

    Args:
      arr_pnl: Массив NumPy с PnL в виде float.

    Returns:
      BeardKoeff
    """

    DrowDown = arr_pnl - np.maximum.accumulate(arr_pnl)
    return len(np.unique(np.maximum.accumulate(arr_pnl))) / (calc_BeardBarsMax(DrowDown))  # *1000
#endregion

#region calc_AnnualPctRate
def calc_AnnualPctRate(arr_pnl: np.array, time_step: str):
    """
    Рассчитывает APR с помощью NumPy.

    Args:
      arr_pnl: Массив NumPy с PnL в виде float.
      time_step: Строка, содержащая информацию о временном шаге ('xm' или 'xh', где x - целое число).

    Returns:
      Значение APR.
    """

    if len(time_step) == 3:
        if time_step[2] == 'm':
            t = 24 * 60 / int(time_step[:2])
        elif time_step[2] == 'h':
            t = 24 / int(time_step[:2])

    if len(time_step) == 2:
        if time_step[1] == 'm':
            t = 24 * 60 / int(time_step[:1])
        elif time_step[1] == 'h':
            t = 24 / int(time_step[:1])
    else:
        raise ValueError("Invalid time_step value. Please use 'xm' or 'xh', where x - integer number.")

    days = arr_pnl.size / t
    years = days / 365
    fraction = (arr_pnl[-1])/ years

    return fraction

#endregion

#region CalmarKoeffPct

def calc_CalmarKoeffPct(arr_pnl: np.array, time_step: str):
    """
    Рассчитывает отношение годовой доходности к максимальной просадке для массива доходностей с помощью NumPy.

    Args:
      arr_pnl: Массив NumPy с PnL в виде float.
      time_step: Строка, содержащая информацию о временном шаге ('xm' или 'xh', где x - целое число).

    Returns:
    Calmar coef
    """

    return (calc_AnnualPctRate(arr_pnl, time_step) / (calc_MaxDrawDawnPctAll(arr_pnl)))

#endregion

#region MonthAvgProfit рассчитывает MonthAvgProfit и % of profit months
def calc_MonthAvgProfit(arr_pnl: np.array, time_step: str):
    """
    Рассчитывает MonthAvgProfit с помощью NumPy.

    Args:
      arr_pnl: Массив NumPy с PnL в виде float.
      time_step: Строка, содержащая информацию о временном шаге ('xm' или 'xh', где x - целое число).

    Returns:
    MonthAvgProfit и % of profit months
    """
    if len(time_step) == 3:
        if time_step[2] == 'm':
            t = int(30 * 24 * 60 / int(time_step[:2]))
        elif time_step[2] == 'h':
            t = int(30 * 24 / int(time_step[:2]))

    if len(time_step) == 2:
        if time_step[1] == 'm':
            t = int(30 * 24 * 60 / int(time_step[:1]))
        elif time_step[1] == 'h':
            t = int(30 * 24 / int(time_step[:1]))
    else:
        raise ValueError("Invalid time_step value. Please use 'xm' or 'xh', where x - integer number.")

    # Вычисляем количество полных отрезков t в массиве
    num_full_periods = int(arr_pnl.size // t)

    # Отбрасываем конечный остаток
    trimmed_arr = arr_pnl[:num_full_periods * t]

    month_starts = np.arange(0, trimmed_arr.size, t)
    month_ends = np.arange(t - 1, trimmed_arr.size, t)
    month_prof = trimmed_arr[month_ends] - trimmed_arr[month_starts]
    count = np.sum(month_prof > 0)

    profit_of_months = count / num_full_periods,
    month_avg = np.mean(month_prof)

    return profit_of_months, month_avg

#endregion

#region Рассчитывает MonthStd и MonthMean
def calc_MonthStdMean(arr_pnl: np.array, time_step: str):
    """
    Рассчитывает MonthStd и MonthMean с помощью NumPy.

    Args:
      arr_pnl: Массив NumPy с PnL в виде float.
      time_step: Строка, содержащая информацию о временном шаге ('xm' или 'xh', где x - целое число).

    Returns:
    MonthStd и MonthMean
    """
    # Проверяем корректность времени шага
    if len(time_step) == 3:
        if time_step[2] == 'm':
            t = int(30 * 24 * 60 / int(time_step[:2]))
        elif time_step[2] == 'h':
            t = int(30 * 24 / int(time_step[:2]))
        else:
            raise ValueError("Invalid time_step value. Please use 'xm' or 'xh', where x - integer number.")
    elif len(time_step) == 2:
        if time_step[1] == 'm':
            t = int(30 * 24 * 60 / int(time_step[0]))
        elif time_step[1] == 'h':
            t = int(30 * 24 / int(time_step[0]))
        else:
            raise ValueError("Invalid time_step value. Please use 'xm' or 'xh', where x - integer number.")
    else:
        raise ValueError("Invalid time_step value. Please use 'xm' or 'xh', where x - integer number.")

    num_full_periods = int(arr_pnl.size // t)
    trimmed_arr = arr_pnl[:num_full_periods * t]
    months = trimmed_arr.reshape(-1, t)

    month_changes = [(month[-1] - month[0]) for month in months]
    month_mean = np.mean(month_changes)
    month_std = np.std(month_changes)

    return {'mean' : month_mean,
            'std' : month_std}

#endregion

#region SharpeMonth - Рассчитывает Sharpe Ratio
def calc_SharpeMonth(arr_pnl: np.array, time_step: str, CashReturnRate: float = 0):
    """
    Рассчитывает Sharpe Ratio с помощью NumPy.

    Args:
      arr_pnl: Массив NumPy с PnL в виде float.
      time_step: Строка, содержащая информацию о временном шаге ('xm' или 'xh', где x - целое число).
      CashReturnRate: безрисковая процентная ставка

    Returns:
      Значение Sharpe Ratio.
    """

    sharp = (calc_MonthStdMean(arr_pnl, time_step)['mean'] * np.sqrt(12) - CashReturnRate) / calc_MonthStdMean(arr_pnl, time_step)['std']

    return sharp

#endregion

#region TradesInYear - Рассчитывает TradesInYear с помощью NumPy.
def calc_TradesInYear(arr_pnl: np.array, arr_trades: np.array, time_step: str):
    """
    Рассчитывает TradesInYear с помощью NumPy.

    Args:
      arr_pnl: Массив NumPy с PnL в виде float.
      time_step: Строка, содержащая информацию о временном шаге ('xm' или 'xh', где x - целое число).

    Returns:
      Значение TradesInYear.
    """
    if len(time_step) == 3:
        if time_step[2] == 'm':
            t = 24 * 60 / int(time_step[:2])
        elif time_step[2] == 'h':
            t = 24 / int(time_step[:2])

    if len(time_step) == 2:
        if time_step[1] == 'm':
            t = 24 * 60 / int(time_step[:1])
        elif time_step[1] == 'h':
            t = 24 / int(time_step[:1])
    else:
        raise ValueError("Invalid time_step value. Please use 'xm' or 'xh', where x - integer number.")

    num_full_days = arr_pnl.size / t

    return int(len(arr_trades) / (num_full_days / 365))

#endregion

#region RecoveryFactor - Рассчитывает RecoveryFactor
def calc_RecoveryFactor(arr_pnl: np.array):
    """
    Рассчитывает RecoveryFactor с помощью NumPy.

    Args:
      arr_pnl: Массив NumPy с PnL в виде float.

    Returns:
      Значение RecoveryFactor.
    """
    return (calc_NetProfitPct(arr_pnl) / (-calc_MaxDrawDawnPctAll(arr_pnl)))

#endregion

#region ProfitPctToEntryPriceAvg
def ProfitPctToEntryPriceAvg(arr_trades: np.array):
    return (np.mean(arr_trades))
#endregion

#region Graal05
def calc_Graal05(arr_pnl: np.array, time_step: str, CashReturnRate: float, arr_trades: np.array):

    AvgProfitPctToEntryPriceIdeal = 0.15
    AvgProfitPctToEntryPriceHighLimit = 0.15
    AvgProfitPctToEntryPriceMin = 0.04
    AvgProfitPctToEntryPriceGraal = ProfitPctToEntryPriceAvg(arr_trades)/AvgProfitPctToEntryPriceIdeal
    if (ProfitPctToEntryPriceAvg(arr_trades) < AvgProfitPctToEntryPriceMin):
        AvgProfitPctToEntryPriceGraal = 0
    if ( AvgProfitPctToEntryPriceGraal > AvgProfitPctToEntryPriceHighLimit / AvgProfitPctToEntryPriceIdeal):
        AvgProfitPctToEntryPriceGraal = (AvgProfitPctToEntryPriceHighLimit / AvgProfitPctToEntryPriceIdeal)

    RecoveryIdeal = 8
    RecoveryHighLimit = 14.0
    RecoveryMin = 2.0
    RecoveryFactorGraal = (calc_RecoveryFactor(arr_pnl) / RecoveryIdeal)

    if (calc_RecoveryFactor(arr_pnl) < RecoveryMin):
        RecoveryFactorGraal = 0
    if (RecoveryFactorGraal > (RecoveryHighLimit / RecoveryIdeal)):
        RecoveryFactorGraal = (RecoveryHighLimit / RecoveryIdeal)

    SharpeMonthIdeal = 2.5
    SharpeHighLimit = 3.5
    SharpeMonthMin = 0.5
    SharpeMonthGraal = (calc_SharpeMonth(arr_pnl, time_step, CashReturnRate) / SharpeMonthIdeal)
    if (calc_SharpeMonth(arr_pnl, time_step, CashReturnRate) < SharpeMonthMin):
        SharpeMonthGraal = 0
    if (SharpeMonthGraal > (SharpeHighLimit / SharpeMonthIdeal)):
        SharpeMonthGraal = (SharpeHighLimit / SharpeMonthIdeal)

    TradesInYearIdeal = 120
    TradesInYearHighLimit = 150
    TradesInYearMin = 20
    TradesInYearGraal = (calc_TradesInYear(arr_pnl, arr_trades, time_step) / TradesInYearIdeal)

    AvgProfitPctToEntryPriceIdeal = 0.15
    AvgProfitPctToEntryPriceHighLimit = 0.15
    AvgProfitPctToEntryPriceMin = 0.04
    AvgProfitPctToEntryPriceGraal = (ProfitPctToEntryPriceAvg(arr_trades) / AvgProfitPctToEntryPriceIdeal)
    if (ProfitPctToEntryPriceAvg(arr_trades) < AvgProfitPctToEntryPriceMin):
        AvgProfitPctToEntryPriceGraal = 0
    if (AvgProfitPctToEntryPriceGraal > AvgProfitPctToEntryPriceHighLimit / AvgProfitPctToEntryPriceIdeal):
        AvgProfitPctToEntryPriceGraal = (AvgProfitPctToEntryPriceHighLimit / AvgProfitPctToEntryPriceIdeal)

    GraalMetr = RecoveryFactorGraal * SharpeMonthGraal * AvgProfitPctToEntryPriceGraal * TradesInYearGraal
    GraalMetr_05 = GraalMetr ** (1 / 4)

    return (GraalMetr_05 * 100)
#endregion