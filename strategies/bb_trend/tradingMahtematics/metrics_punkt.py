#region Импортируем библиотеки
import pandas as pd
import numpy as np #(Numeric Python) библиотека - аналог MathLab
import matplotlib.pyplot as plt
from functools import reduce
#endregion

 #region Функция для подсчета количества дней, в течение которых происходила торговля.
def trading_days(start_date: np.datetime64, end_date: np.datetime64) -> int:
    """
    Функция для подсчета количества дней между датой начала и датой окончания торговли.

    :param start_date: np.datetime64 - дата начала торговли
    :param end_date: np.datetime64 - дата окончания торговли
    :return: int - количество дней между датами
    """

    delta = np.datetime64(end_date) - np.datetime64(start_date)
    days = delta.astype('timedelta64[D]').astype(int)

    return days
#endregion

#region NetProfitPunktAllPositions - Итоговая прибsль / убыток в пунктах (считается как сумма дохода (убытка) по всем сделкам
#TODO: Сделать рассчёт показателя
#def NetProfitPunktAllPositions ():

#endregion

#region AnnualPctRate - Рассчитывает APR

def annual_pct_rate_calc(present_value, present_date, future_value, future_date):
    """
    процентная ставка роста за год: APR (annual percentage rate) или CAGR (Compound Annual Growth Rate)

    :param present_value: Начальная сумма эквити
    :param present_date: Дата начала эквити (numpy.datetime64)
    :param future_value: Конечная сумма эквити
    :param future_date: Дата окончания эквити (numpy.datetime64)
    :returns: процентная ставка роста за год
    """
    # Проверка на нулевые или отрицательные значения present_value и future_value
    if present_value <= 0:
        raise ValueError("Начальная сумма эквити должна быть положительным числом.")

    # Находим отрезок времени между двумя датами
    strategy_time = np.datetime64(future_date) - np.datetime64(present_date) #тип данных результата: timedelta64

    # Определяем количество дней во временном промежутке.
    days = strategy_time.astype('timedelta64[D]')

    # Переводим количество дней в количество лет
    days_int = days.astype(int) # Преобразуем количество дней в целочисленный тип

    # Вычисляем количество периодов, деля количество дней на 365 (число дней в году)
    number_of_periods = days_int / 365.0

    # Проверка на нулевую или отрицательную продолжительность периода
    if number_of_periods <= 0:
        raise ValueError("Дата окончания должна быть позже даты начала.")

    # Вычисляем среднегодовую ставку доходности (APR)
    try:
        result = (future_value / present_value) ** (1.0 / number_of_periods) - 1.0 if (future_value / present_value) > 0 else 0
    except ZeroDivisionError:
        raise ValueError("Деление на ноль произошло при вычислении средней годовой ставки доходности.")

    # Переводим результат в проценты
    result = result * 100.0

    return result

#endregion

#region MaxDrawDawnPctAll - Max просадка от текущего максимума эквити в % по серии сделок
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

#region AnnualPctRate - Рассчитывает APR с помощью NumPy
def calc_AnnualPctRate(arr_pnl: np.array, time_step: str):
    """
    Рассчитывает APR с помощью NumPy.

    Args:
      arr_pnl: Массив NumPy с PnL в виде float.
      time_step: Строка, содержащая информацию о временном шаге ('xm' или 'xh', где x - целое число).

    Returns:
      Значение APR.
    """

    if len(time_step)==3:
      if time_step[2]=='m':
        t = 24*60/int(time_step[:2])
      elif time_step[2]=='h':
        t = 24/int(time_step[:2])

    if len(time_step)==2:
      if time_step[1]=='m':
        t = 24*60/int(time_step[:1])
      elif time_step[1]=='h':
        t = 24/int(time_step[:1])
    else:
        raise ValueError("Invalid time_step value. Please use 'xm' or 'xh', where x - integer number.")

    days = arr_pnl.size / t
    years = days / 365
    fraction = (arr_pnl[arr_pnl.size - 1] / arr_pnl[0])**(1/years) - 1

    return fraction*100
#endregion





#region MaxDrawDawnPctAllFromEquity - Max просадка от текущего максимума эквити в % по серии сделок
def calc_MaxDrawDawnPctAllFromEquity(arr_pnl: np.array):
    """
    Вычисляем MaxDrawDawnPctAll

    Args:
      arr_pnl: Массив NumPy с PnL в виде float.

    Returns:
      MaxDrawDawnPctAll
    """
    return np.min((arr_pnl - np.maximum.accumulate(arr_pnl))/np.maximum.accumulate(arr_pnl)*100)
#endregion

#region draw_dawn_curve_punkt_calc - Вычисляем DrawDawnCurve
def draw_dawn_curve_punkt_calc(net_profit_punkt_arr: np.array) -> np.array:
    """
    Вычисляет график просадок (drawdown curve).

    Args:
        net_profit_array: Массив NumPy с чистой прибылью в виде float.

    Returns:
        drawdown_curve: Массив NumPy, представляющий график просадок.
    """

    # Накопленный максимум чистой прибыли на каждом шагу
    cumulative_max_profit = np.maximum.accumulate(net_profit_punkt_arr)

    # Разница между накопленным максимумом и текущей чистой прибылью
    drawdown_curve_punkt = cumulative_max_profit - net_profit_punkt_arr

    return drawdown_curve_punkt
#endregion

#region drawdown_curve_pct_calc - Вычисляем DrawDawnCurve
def draw_dawn_curve_pct_calc(init_deposit:float, net_profit_punkt_arr: np.array) -> np.array:
    """
    Вычисляет график просадок (drawdown curve %).

    Args:
        net_profit_array: Массив NumPy с чистой прибылью в пунктах в виде float.

    Returns:
        drawdown_curve_pct: Просадки в % от максимума эквити Массив NumPy, представляющий график просадок.
    """

    # Накопленный максимум чистой прибыли на каждом шагу
    cumulative_max_profit_punkt = np.maximum.accumulate(net_profit_punkt_arr)

    # Разница между накопленным максимумом и текущей чистой прибылью
    drawdown_curve_punkt = cumulative_max_profit_punkt - net_profit_punkt_arr

    # Вычисляем drawdown в процентах только для ненулевых элементов
    drawdown_curve_pct = drawdown_curve_punkt / (cumulative_max_profit_punkt + init_deposit) * 100.0

    return drawdown_curve_pct
#endregion

#region BeardBarsMaxn
def calc_BeardBarsMax(arr_pnl: np.array):
    """
    Вычисляем BeardBarsMax

    Args:
      arr_pnl: Массив NumPy с PnL в виде float.

    Returns:
      BeardBarsMax
    """

    zero_indices = list(filter(lambda x: arr_pnl[x] == 0, range(len(arr_pnl))))

    if len(zero_indices) == 0:
        return len(arr_pnl)
    if len(zero_indices) == 1:
        return max(len(arr_pnl) - zero_indices[0] - 1, zero_indices[0])

    max_gap = reduce(lambda acc, x: max(acc, x[1] - x[0] - 1), zip(zero_indices, zero_indices[1:]), 0)
    max_gap = max(max_gap, len(arr_pnl) - zero_indices[-1] - 1, zero_indices[0])

    return max_gap
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
    return len(np.unique(np.maximum.accumulate(arr_pnl)))/(calc_BeardBarsMax(DrowDown))*1000
#endregion

#region CalmarKoeffPct - отношение годовой доходности к максимальной просадке для массива доходностей с помощью NumPy.
def calc_CalmarKoeffPct (arr_pnl: np.array, time_step: str):
    """
    Рассчитывает отношение годовой доходности к максимальной просадке для массива доходностей с помощью NumPy.

    Args:
      arr_pnl: Массив NumPy с PnL в виде float.
      time_step: Строка, содержащая информацию о временном шаге ('xm' или 'xh', где x - целое число).

    Returns:
    Calmar coef
    """

    return(calc_AnnualPctRate(arr_pnl, time_step)/(calc_MaxDrawDawnPctAll(arr_pnl)))
#endregion

#region MonthAvgProfit
def calc_MonthAvgProfit(arr_pnl: np.array, time_step: str):
    """
    Рассчитывает MonthAvgProfit с помощью NumPy.

    Args:
      arr_pnl: Массив NumPy с PnL в виде float.
      time_step: Строка, содержащая информацию о временном шаге ('xm' или 'xh', где x - целое число).

    Returns:
    MonthAvgProfit и % of profit months
    """
    if len(time_step)==3:
      if time_step[2]=='m':
        t = 30 * 24 * 60/int(time_step[:2])
      elif time_step[2]=='h':
        t = 30 * 24/int(time_step[:2])

    if len(time_step)==2:
      if time_step[1]=='m':
        t = 30 * 24 * 60/int(time_step[:1])
      elif time_step[1]=='h':
        t = 30 * 24/int(time_step[:1])
    else:
        raise ValueError("Invalid time_step value. Please use 'xm' or 'xh', where x - integer number.")

    # Вычисляем количество полных отрезков t в массиве
    num_full_periods = arr_pnl.size // t
    # Отбрасываем конечный остаток
    trimmed_arr = arr_pnl[:num_full_periods * t]

    month_starts = np.arange(0, trimmed_arr.size, t)
    month_ends = np.arange(t - 1, trimmed_arr.size, t)
    month_prof = trimmed_arr[month_starts] - trimmed_arr[month_ends]
    count = np.sum(month_prof > 0)

    return {'% of profit months': count / num_full_periods,
            'month_avg': np.mean(month_prof)}
#endregion

#region Month StdMean - Рассчитывает MonthStd и MonthMean с помощью NumPy
def calc_MonthAvgProfit(arr_pnl: np.array, time_step: str):
    """
    Рассчитывает MonthAvgProfit с помощью NumPy.

    Args:
      arr_pnl: Массив NumPy с PnL в виде float.
      time_step: Строка, содержащая информацию о временном шаге ('xm' или 'xh', где x - целое число).

    Returns:
    MonthAvgProfit и % of profit months
    """
    # Проверяем корректность времени шага
    if len(time_step) == 3:
        if time_step[2] == 'm':
            t = (30 * 24 * 60) / int(time_step[:2])
        elif time_step[2] == 'h':
            t = (30 * 24) / int(time_step[:2])
        else:
            raise ValueError("Invalid time_step value. Please use 'xm' or 'xh', where x - integer number.")
    elif len(time_step) == 2:
        if time_step[1] == 'm':
            t = (30 * 24 * 60) / int(time_step[0])
        elif time_step[1] == 'h':
            t = (30 * 24) / int(time_step[0])
        else:
            raise ValueError("Invalid time_step value. Please use 'xm' or 'xh', where x - integer number.")
    else:
        raise ValueError("Invalid time_step value. Please use 'xm' or 'xh', where x - integer number.")

    # Вычисляем количество полных отрезков t в массиве
    num_full_periods = arr_pnl.size // t
    # Отбрасываем конечный остаток
    trimmed_arr = arr_pnl[:int(num_full_periods * t)]

    month_profit = []

    month_starts = np.arange(0, trimmed_arr.size, t)
    for i in month_starts:
        month_profit.append(trimmed_arr[int(i+1)] - trimmed_arr[int(i)])

    month_profit = np.array(month_profit)
    count = np.sum(month_profit > 0)

    return {'% of profit months': (count / num_full_periods)*100,
            'month_avg': np.mean(month_profit)}
#endregion

def calc_QuaterAvgProfit(arr_pnl: np.array, time_step: str):
    """
    Рассчитывает MonthAvgProfit с помощью NumPy.

    Args:
      arr_pnl: Массив NumPy с PnL в виде float.
      time_step: Строка, содержащая информацию о временном шаге ('xm' или 'xh', где x - целое число).

    Returns:
    QuaterAvgProfit и % of profit months
    """
    # Проверяем корректность времени шага
    if len(time_step) == 3:
        if time_step[2] == 'm':
            t = (30 * 24 * 60) * 4 / int(time_step[:2])
        elif time_step[2] == 'h':
            t = (30 * 24) * 4/ int(time_step[:2])
        else:
            raise ValueError("Invalid time_step value. Please use 'xm' or 'xh', where x - integer number.")
    elif len(time_step) == 2:
        if time_step[1] == 'm':
            t = (30 * 24 * 60) * 4 / int(time_step[0])
        elif time_step[1] == 'h':
            t = (30 * 24) * 4/ int(time_step[0])
        else:
            raise ValueError("Invalid time_step value. Please use 'xm' or 'xh', where x - integer number.")
    else:
        raise ValueError("Invalid time_step value. Please use 'xm' or 'xh', where x - integer number.")

    # Вычисляем количество полных отрезков t в массиве
    num_full_periods = arr_pnl.size // t
    # Отбрасываем конечный остаток
    trimmed_arr = arr_pnl[:int(num_full_periods * t)]

    month_profit = []

    month_starts = np.arange(0, trimmed_arr.size, t)
    for i in month_starts:
        month_profit.append(trimmed_arr[int(i+1)] - trimmed_arr[int(i)])

    month_profit = np.array(month_profit)
    count = np.sum(month_profit > 0)

    return {'% of profit quater': (count / num_full_periods)*100,
            'quater_avg': np.mean(month_profit)}

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
            t = (30 * 24 * 60) / int(time_step[:2])
        elif time_step[2] == 'h':
            t = (30 * 24) / int(time_step[:2])
        else:
            raise ValueError("Invalid time_step value. Please use 'xm' or 'xh', where x - integer number.")
    elif len(time_step) == 2:
        if time_step[1] == 'm':
            t = (30 * 24 * 60) / int(time_step[0])
        elif time_step[1] == 'h':
            t = (30 * 24) / int(time_step[0])
        else:
            raise ValueError("Invalid time_step value. Please use 'xm' or 'xh', where x - integer number.")
    else:
        raise ValueError("Invalid time_step value. Please use 'xm' or 'xh', where x - integer number.")

    t = int(t)
    num_full_periods = arr_pnl.size // t
    trimmed_arr = arr_pnl[:num_full_periods * t]
    months = trimmed_arr.reshape(-1, t)
    month_changes = [(month[-1] - month[0]) * 100 / month[0] for month in months]
    month_mean = np.mean(month_changes)
    month_std = np.std(month_changes)

    return {'month_mean': month_mean, 'month_std': month_std}
#endregion

#region SharpeMonth - Рассчитывает Sharpe Ratio с помощью NumPy.
def calc_SharpeMonth(net_profit_punkt_np: np.array, timeframe: str, CashReturnRate: float):
    """
    Рассчитывает Sharpe Ratio с помощью NumPy.

    Args:
      net_profit_punkt_np: Массив NumPy с PnL в виде float.
      timeframe: Строка, содержащая информацию о временном шаге ('xm' или 'xh', где x - целое число).
      CashReturnRate: безрисковая процентная ставка

    Returns:
      Значение Sharpe Ratio.
    """

    sharp = (calc_MonthStdMean(net_profit_punkt_np, timeframe)['month_mean'] * np.sqrt(12) - CashReturnRate) / calc_MonthStdMean(net_profit_punkt_np, timeframe)['month_std']

    return sharp
#endregion

#region TradesInYear
def calc_TradesInYear(arr_pnl: np.array, arr_trades: np.array,  time_step: str):
    """
    Рассчитывает TradesInYear с помощью NumPy.

    Args:
      arr_pnl: Массив NumPy с PnL в виде float.
      time_step: Строка, содержащая информацию о временном шаге ('xm' или 'xh', где x - целое число).

    Returns:
      Значение TradesInYear.
    """
    if len(time_step)==3:
      if time_step[2]=='m':
        t = 24*60/int(time_step[:2])
      elif time_step[2]=='h':
        t = 24/int(time_step[:2])

    if len(time_step)==2:
      if time_step[1]=='m':
        t = 24*60/int(time_step[:1])
      elif time_step[1]=='h':
        t = 24/int(time_step[:1])
    else:
        raise ValueError("Invalid time_step value. Please use 'xm' or 'xh', where x - integer number.")

    num_full_days = arr_pnl.size / t


    return (len(arr_trades)/(num_full_days/365))
#endregion

#region NetProfitPct
def calc_NetProfitPct(arr_pnl: np.array):
    """
    Рассчитывает NetProfitPct с помощью NumPy.

    Args:
      arr_pnl: Массив NumPy с PnL в виде float.
      time_step: Строка, содержащая информацию о временном шаге ('xm' или 'xh', где x - целое число).

    Returns:
      Значение NetProfitPct.
    """
    return((abs(arr_pnl[arr_pnl.size-1] - arr_pnl[0])/arr_pnl[0])*100)
#endregion

#region RecoveryFactor
def calc_RecoveryFactor(arr_pnl: np.array):
    """
    Рассчитывает RecoveryFactor с помощью NumPy.

    Args:
      arr_pnl: Массив NumPy с PnL в виде float.

    Returns:
      Значение RecoveryFactor.
    """
    return (calc_NetProfitPct(arr_pnl)/(-calc_MaxDrawDawnPctAll(arr_pnl)))

#endregion

#region ProfitPctToEntryPriceAvg
def ProfitPctToEntryPriceAvg(arr_trades: np.array):
    return (np.mean(arr_trades))
#endregion

#region Graal05
def calc_Graal04( TradesInYear: float, RecoveryFactor: float, SharpeMonth: float):

    RecoveryIdeal = 8
    RecoveryHighLimit = 14.0
    RecoveryMin = 2.0
    RecoveryFactorGraal = (RecoveryFactor / RecoveryIdeal)

    if (RecoveryFactor < RecoveryMin):
        RecoveryFactorGraal = 0
    if (RecoveryFactorGraal > (RecoveryHighLimit / RecoveryIdeal)):
        RecoveryFactorGraal = (RecoveryHighLimit / RecoveryIdeal)

    SharpeMonthIdeal = 2.5
    SharpeHighLimit = 3.5
    SharpeMonthMin = 0.5
    SharpeMonthGraal = (SharpeMonth / SharpeMonthIdeal)
    if (SharpeMonth < SharpeMonthMin):
        SharpeMonthGraal = 0
    if (SharpeMonthGraal > (SharpeHighLimit / SharpeMonthIdeal)):
        SharpeMonthGraal = (SharpeHighLimit / SharpeMonthIdeal)

    TradesInYearIdeal = 120
    TradesInYearHighLimit = 150
    TradesInYearMin = 20
    TradesInYearGraal = TradesInYear / TradesInYearIdeal

    GraalMetr = RecoveryFactorGraal * SharpeMonthGraal * TradesInYearGraal
    GraalMetr_04 = GraalMetr ** (1 / 3)

    return(GraalMetr_04*100)
#endregion