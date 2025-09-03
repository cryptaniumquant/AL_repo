import datetime
import numpy as np
from datetime import datetime
from datetime import datetime, timedelta

#region pct_to_punkt
def pct_to_punkt(start_capital: float, arr_pnl: np.array) -> np.array:
    """
    Преобразует процентное изменение капитала в абсолютное значение.

    Функция принимает начальный капитал и массив процентных изменений (PnL) и
    возвращает массив абсолютных значений капитала после применения каждого процентного изменения.

    Аргументы:
    start_capital (float): Начальный капитал.
    arr_pnl (np.array): Массив процентных изменений капитала.

    Возвращает:
    np.array: Массив абсолютных значений капитала после применения процентных изменений.

    Примеры:
    >>> start_capital = 10000
    >>> arr_pnl = np.array([1, -0.5, 2])
    >>> pct_to_punkt(start_capital, arr_pnl)
    array([10100.,  9950., 10200.])
    """
    return start_capital * (arr_pnl / 100) + start_capital

#endregion

#region punkt_to_pct
def punkt_to_pct(start_capital: float, arr_absolute: np.array) -> np.array:
    """
    Преобразует абсолютное изменение капитала в процентное значение.

    Функция принимает начальный капитал и массив абсолютных изменений капитала (в пунктах) и
    возвращает массив процентных изменений капитала.

    Аргументы:
    start_capital (float): Начальный капитал.
    arr_absolute (np.array): Массив абсолютных (в пунктах) изменений капитала.

    Возвращает:
    np.array: Массив процентных изменений капитала.

    Примеры:
    >>> start_capital = 10000
    >>> arr_absolute = np.array([10100, 9950, 10200])
    >>> punkt_to_pct(start_capital, arr_absolute)
    array([ 1. , -0.5,  2. ])
    """
    return (arr_absolute / start_capital) * 100

#endregion

#region timestamp_to_datetime: Преобразование Даты и времени из timestamp в datetime
def timestamp_to_datetime(timestamp):
    """
    Преобразует метку времени (timestamp) в объект datetime.

    Метка времени (timestamp) представлена в миллисекундах с начала эпохи (01.01.1970).
    Функция преобразует метку времени в секунды и создает объект datetime.

    Аргументы:
    timestamp (int, float): Метка времени в миллисекундах.

    Возвращает:
    datetime.datetime: Объект datetime, представляющий заданную метку времени.

    Примеры:
    >>> timestamp_to_datetime(1633072800000)
    datetime.datetime(2021, 10, 1, 0, 0)
    >>> timestamp_to_datetime(0)
    datetime.datetime(1970, 1, 1, 0, 0)
    """
    return datetime.datetime.fromtimestamp(timestamp / 1000.0)
#endregion

#Вот пример функции, которая принимает numpy массив с timestamp на вход и возвращает numpy массив с датами в формате datetime:

#region convert_timestamps_to_datetimes: Преобразует numpy массив с timestamp в numpy массив с datetime.
def convert_timestamps_to_datetimes(stmpDate_np):
    """
    Преобразует numpy массив с timestamp в numpy массив с datetime.

    Args:
    stmpDate_np (numpy.ndarray): Исходный массив с timestamp.

    Returns:
    numpy.ndarray: Массив с датами в формате datetime.

    Пример использования функции:
    stmpDate_np = np.array([1633072800, 1633159200, 1633245600])  # Пример значений timestamp
    >>> dtDate_np = convert_timestamps_to_datetimes(stmpDate_np)
    """
    # Функция для преобразования timestamp в datetime
    def timestamp_to_datetime(ts):
        return datetime.fromtimestamp(ts)

    # Применяем функцию ко всему массиву с помощью vectorize
    vectorized_timestamp_to_datetime = np.vectorize(timestamp_to_datetime)

    # Преобразуем массив timestamp в массив datetime
    dtDate_np = vectorized_timestamp_to_datetime(stmpDate_np)

    return dtDate_np

#endregion


def fill_hourly_equity(net_profit_arr: np.ndarray, date_np: np.datetime64) -> np.ndarray:
    """
    Заполняет массив почасовыми значениями капитала на основе предоставленных данных.

    Эта функция принимает массив значений капитала и соответствующие временные метки,
    а затем возвращает массив значений капитала для каждого часа в диапазоне временных меток.
    Если для конкретного часа нет значения капитала, используется последнее известное значение.

    Параметры:
    net_profit_arr (ndarray): Массив значений капитала.
    date_np (ndarray): Массив временных меток (объекты numpy.datetime64).

    Возвращает:
    ndarray: Массив значений капитала для каждого часа в диапазоне временных меток.
    """

    # Конвертируем временные метки из numpy.datetime64 в datetime
    cur_date = np.array([datetime.utcfromtimestamp(ts.astype('O') / 1e9) for ts in date_np])

    # Определяем начальное и конечное время
    start_time = cur_date[0]
    end_time = cur_date[-1]

    # Создаем массив всех часов в диапазоне от start_time до end_time
    total_hours = int((end_time - start_time).total_seconds() // 3600) + 1
    hourly_dates = [start_time + timedelta(hours=i) for i in range(total_hours)]

    # Создаем словарь для быстрого поиска значений по временным меткам
    equity_dict = {date: value for date, value in zip(cur_date, net_profit_arr)}

    # Заполняем результат, используя значения из equity_dict
    hourly_equity = []
    last_value = None

    for hour in hourly_dates:
        if hour in equity_dict:
            last_value = equity_dict[hour]
        hourly_equity.append(last_value)

    return np.array(hourly_equity)

def fill_daily_equity(net_profit_arr: np.ndarray, date_np: np.ndarray) -> np.ndarray:
    """
    Заполняет массив дневными значениями капитала на основе предоставленных данных.

    Эта функция принимает массив значений капитала и соответствующие временные метки,
    а затем возвращает массив значений капитала для каждого дня в диапазоне временных меток.
    Если для конкретного дня нет значения капитала, используется последнее известное значение.

    Параметры:
    net_profit_arr (ndarray): Массив значений капитала.
    date_np (ndarray): Массив временных меток (объекты numpy.datetime64).

    Возвращает:
    ndarray: Массив значений капитала для каждого дня в диапазоне временных меток.
    """

    # Конвертируем временные метки из numpy.datetime64 в datetime
    cur_date = date_np.astype('datetime64[s]').astype(datetime)

    # Определяем начальное и конечное время
    start_time = cur_date[0].replace(hour=0, minute=0, second=0, microsecond=0)
    end_time = cur_date[-1].replace(hour=0, minute=0, second=0, microsecond=0)

    # Создаем массив всех дней в диапазоне от start_time до end_time
    total_days = (end_time - start_time).days + 1
    daily_dates = [start_time + timedelta(days=i) for i in range(total_days)]

    # Создаем словарь для быстрого поиска значений по временным меткам
    equity_dict = {date.replace(hour=0, minute=0, second=0, microsecond=0): value for date, value in zip(cur_date, net_profit_arr)}

    # Заполняем результат, используя значения из equity_dict
    daily_equity = []
    last_value = None

    for day in daily_dates:
        if day in equity_dict:
            last_value = equity_dict[day]
        daily_equity.append(last_value)

    return np.array(daily_equity)

    #region функция для случайного удаления указанного количества процентов ячеек массива

def remove_random_elements(arr, percent):
    """
    Убирает случайным образом указанный процент ячеек из массива.

    :param arr: входной одномерный массив
    :param percent: процент ячеек для удаления
    :return: новый массив с удалёнными ячейками
    """
    if not 0 <= percent <= 100:
        percent = 0 #Оставляем массив неизменным

    total_elements = len(arr)
    num_elements_to_remove = int(total_elements * (percent / 100.0))

    if num_elements_to_remove == 0:
        return arr  # Если удалять нечего, возвращаем оригинальный массив

    indices_to_remove = np.random.choice(total_elements, num_elements_to_remove, replace=False)
    new_arr = np.delete(arr, indices_to_remove)

    return new_arr
 #endregion

def reduce_array_by_percentage(unique_combinations, percentage):
    """
    Уменьшает количество строк в массиве на указанный процент случайным образом.

    Параметры:
    unique_combinations (numpy.ndarray): Исходный массив, содержащий строки для уменьшения.
    percentage (float): Процент строк, которые нужно удалить из исходного массива.

    Возвращает:
    numpy.ndarray: Новый массив, в котором количество строк уменьшено на указанный процент случайным образом.
    """

    # Определяем количество строк в исходном массиве
    num_rows = unique_combinations.shape[0]

    # Рассчитываем количество строк, которые нужно оставить после удаления
    num_rows_to_keep = int(num_rows * (1 - percentage / 100))

    # Выбираем случайные индексы строк, которые будут оставлены в новом массиве
    # np.random.choice выбирает уникальные индексы без замены (replace=False)
    indices_to_keep = np.random.choice(num_rows, num_rows_to_keep, replace=False)

    # Создаем новый массив, используя выбранные индексы строк
    combinations_short = unique_combinations[indices_to_keep]

    # Возвращаем новый массив
    return combinations_short
