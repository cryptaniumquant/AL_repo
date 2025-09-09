from pathlib import Path
import pandas as pd
from data_utils import prepare_data
from download_data import fetch_binance_candles
import os
import datetime
import re


PATH_TO_CSV_FOLDER = Path(__file__).parent.joinpath("CSV")

class DataProvider:
    def __init__(self) -> None:
        pass

    def get(self, data_params):
        
        # Получаем инфу о запрашиваемой монете
        symbol = data_params["symbol"]
        timeframe = data_params["timeframe"]
        timezone = data_params["timezone"]
        start_date = data_params["start_date"]
        
        need_to_fetch = False # Флаг, надо ли скачать свечки

        # Путь до свечек
        path_to_candles = str(PATH_TO_CSV_FOLDER.joinpath(symbol.replace('/', '') + ".csv"))


        if os.path.isfile(path_to_candles): # Свечки уже скачаны локально
            print("Found data localy")
            bars_df = pd.read_csv(path_to_candles)

            # Находим timeframe из файла (в виде datetime.timedelta)
            t0 = datetime.datetime.strptime(bars_df["close_time"][0], "%Y-%m-%d %H:%M:%S")
            t1 = datetime.datetime.strptime(bars_df["close_time"][1], "%Y-%m-%d %H:%M:%S")
            local_data_timeframe: datetime.timedelta = t1 - t0

            # Находим timeframe который запрашивает пользователь (в виде datetime.timedelta)
            timeframe_as_timedelta: datetime.timedelta = self.timeframe_to_timedelta(timeframe=timeframe)

            print(f"    > Local data timeframe: {local_data_timeframe}")
            print(f"    > Your timeframe: {timeframe_as_timedelta}")
            
            if (local_data_timeframe < timeframe_as_timedelta):
                print(f"I can transform local timeframe to your timeframe {local_data_timeframe} ---> {timeframe_as_timedelta}")
            else:
                print(f"I can't transform local timeframe to your timeframe {local_data_timeframe} -/-> {timeframe_as_timedelta}")
                need_to_fetch = True
        else:
            need_to_fetch = True

        if need_to_fetch: # Загружаем данные
            print("Fetching...")

            # Данные берем с Binance, поэтому переводим в binance timeframe
            binance_timeframe = self.timeframe_to_binance_timeframe(timeframe=timeframe)
            
            # Получаем данные
            bars_df = fetch_binance_candles(symbol, timeframe=binance_timeframe, start_date=start_date, timezone=timezone)
            
            # Данные пустые ---> ЧТО-ТО ПОШЛО НЕ ТАК
            assert bars_df.shape[0] != 0, 'Data is empty'

            # Сохраняем в CSV файл
            PATH_TO_SAVE = str(PATH_TO_CSV_FOLDER) + f"/{symbol.replace('/', '')}.csv"
            bars_df.to_csv(PATH_TO_SAVE, index=False)

        # Производим смену timeframe, немного меняем названия столбцов
        bars_df = prepare_data(bars_df, timeframe=timeframe)

        return bars_df
        
    def timeframe_to_timedelta(self, timeframe: str) -> datetime.timedelta:
        if re.match(r'^\d+m$', timeframe):
            return datetime.timedelta(minutes=int(timeframe.replace("m", "")))
        elif re.match(r'^\d+min$', timeframe):
            return datetime.timedelta(minutes=int(timeframe.replace("min", "")))
        elif re.match(r'^\d+h$', timeframe):
            return datetime.timedelta(hours=int(timeframe.replace("h", "")))

        raise NotImplementedError("Can't solve your timeframe")

    def timeframe_to_binance_timeframe(self, timeframe: str):
        if re.match(r'^\d+min$', timeframe):
            return timeframe.replace("min", "m")
        
if __name__ == "__main__":
    dp = DataProvider()

    bars_df = dp.get({
        "symbol": "UNIUSDT",
        "timeframe": "5min",
        "timezone": "Europe/Moscow",
        "start_date": "2023-01-01"   
    })

    print(bars_df.head(5))
    