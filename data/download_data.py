import ccxt
import pandas as pd
from datetime import datetime
import time
import pytz
from pathlib import Path
import sys


def fetch_binance_candles(symbol='BTC/USDT', timeframe='1h', start_date='2025-01-01', limit=100000, timezone='UTC'):
    # Инициализация биржи Binance через ccxt
    exchange = ccxt.binance({
        'enableRateLimit': True,  # Включаем ограничение скорости запросов
    })

    # Преобразуем start_date в timestamp (в миллисекундах)
    start_timestamp = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp() * 1000)

    # Список для хранения всех свечек
    all_candles = []

    while True:
        try:
            # Запрашиваем свечи
            candles = exchange.fetch_ohlcv(symbol, timeframe, since=start_timestamp, limit=limit)

            if not candles:
                break

            all_candles.extend(candles)

            # Обновляем start_timestamp для следующего запроса
            start_timestamp = candles[-1][0] + 1

            # Задержка для соблюдения лимитов API
            time.sleep(exchange.rateLimit / 1000)

        except Exception as e:
            print(f"Ошибка при получении данных: {e}")
            break

    # Создаем DataFrame
    df = pd.DataFrame(all_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

    # Преобразуем timestamp в читаемый формат даты с учетом временной зоны
    tz = pytz.timezone(timezone)
    df['date'] = pd.to_datetime(df['timestamp'], unit='ms').dt.tz_localize('UTC').dt.tz_convert(tz)

    # Убираем информацию о временной зоне, форматируем дату в строку
    df['date'] = df['date'].dt.strftime('%Y-%m-%d %H:%M:%S')

    # Удаляем столбец timestamp, если он не нужен
    df = df[['date', 'open', 'high', 'low', 'close', 'volume']]

    # Переименовываем столбцы
    df = df.rename(columns={
        'open': 'open_price',
        'high': 'high_price',
        'low': 'low_price',
        'close': 'close_price',
        'date': 'close_time'
    })

    return df


# Пример использования
if __name__ == "__main__":
    assert len(sys.argv) == 3, "Please run this file like:\npython python.db_data_get.py btc/usdt 15m"
    
    # Настройки
    symbol = sys.argv[1].upper()  #'APT/USDT'  # Торговая пара
    timeframe = sys.argv[2]  #'15m'  # Интервал свечей (1m, 5m, 15m, 1h, 4h, 1d и т.д.)
    start_date = '2023-01-01'  # Дата начала
    timezone = 'Europe/Moscow'  # Временная зона UTC+7

    # Печатаем настройки
    print(f"> SYMBOL: {symbol}")
    print(f"> TIMEFRAME: {timeframe}")

    # Получаем данные
    candles_df = fetch_binance_candles(symbol, timeframe, start_date, timezone=timezone)

    # Сохраняем в CSV файл
    PATH_TO_SAVE = Path(__file__).parent.joinpath(f"CSV/{symbol.replace('/', '')}.csv")
    candles_df.to_csv(PATH_TO_SAVE, index=False)

    # Выводим первые несколько строк
    print("Saved into:", PATH_TO_SAVE)
    print("candles_df.shape:", candles_df.shape)
            