from pathlib import Path
import pandas as pd
from data.data_utils import prepare_data
from data.download_data import fetch_binance_candles
import os


PATH_TO_CSV_FOLDER = Path(__file__).parent.joinpath("CSV")

class DataProvider:
    def __init__(self) -> None:
        pass

    def get(self, data_params):
        
        symbol = data_params["symbol"]
        timeframe = data_params["timeframe"]
        timezone = data_params["timezone"]
        start_date = data_params["start_date"]

        path_to_candles = str(PATH_TO_CSV_FOLDER.joinpath(symbol.replace('/', '') + ".csv"))
        if os.path.isfile(path_to_candles):
            bars_df = pd.read_csv(path_to_candles)
        else:
            print("Fetching...")
            # Получаем данные
            bars_df = fetch_binance_candles(symbol, timeframe=timeframe, start_date=start_date, timezone=timezone)
            assert bars_df.shape[0] != 0, 'Data is empty'
            # Сохраняем в CSV файл
            PATH_TO_SAVE = str(PATH_TO_CSV_FOLDER) + f"/{symbol.replace('/', '')}.csv"
            bars_df.to_csv(PATH_TO_SAVE, index=False)

        bars_df = prepare_data(bars_df, timeframe=timeframe)

        return bars_df
        