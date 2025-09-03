from performance_metrics import performance_metrics_new, strategy_charts
import pandas as pd
from pathlib import Path
import numpy as np
from data.DataProvider import DataProvider
from strategies.FoBo.FoBo import FoBo
import os
import json
import importlib
from mysql.MysqlConnector import MysqlConnector


def run_strategy(experiment, bars_df=None, interval=None):
    
    # Run Strategy
    strategy_class_s = experiment["STRATEGY"]["strategy_class"]
    start_capital = experiment["STRATEGY"]["start_capital"]
    rel_commission = experiment["STRATEGY"]["rel_commission"]
    params = experiment["STRATEGY"]["params"]
    interval = experiment["STRATEGY"]["interval"] if interval is None else interval

    # Импортируем класс стратегии
    module = importlib.import_module(f'strategies.{strategy_class_s}.{strategy_class_s}')
    strategy_class = getattr(module, strategy_class_s)

    # Создаем объект стратегии
    strategy = strategy_class(start_capital, rel_commission, is_optimization=False)

    # Запускаем
    metrics = strategy.run(
        params=experiment["STRATEGY"]["params"],
        bars_df=bars_df,
        interval=[0, 1],
        metrics=experiment["METRICS"])

    # Количество сделок
    print("Number of postions:", len(strategy.positions))
    # Выводим метрики
    for metric_name in metrics:
        print(f"    > {metric_name}: {metrics[metric_name]:.3f}")

    # Сохраняем эксперимент
    for pname in experiment["STRATEGY"]["params"]:
        experiment["STRATEGY"]["params"][pname] = experiment["STRATEGY"]["params"][pname]["v"]

    experiment["TRADES_COUNT"] = len(strategy.positions)

    with open("experiment.json", "w") as f:
        json.dump(experiment, f)

    # DB
    mysql_connector = MysqlConnector()
    mysql_connector.save(
        strategy_class=experiment["STRATEGY"]["strategy_class"],
        symbol=experiment["DATA"]["symbol"],
        timeframe=experiment["DATA"]["timeframe"],
        start_date=bars_df["Date_dt"].min(),
        end_date=bars_df["Date_dt"].max(),
        net_profit=strategy.net_profit,
        params=experiment["STRATEGY"]["params"],
        metrics=metrics
    )

if __name__ == "__main__":

    EXPERIMENT = {
        "DATA": {
            "symbol": "SOL/USDT",
            "timeframe": "1h",
            "timezone": "Europe/Moscow",
            "start_date": "2023-01-01"
            
        },

        "STRATEGY": {
            "strategy_class": "FoBo",
            "start_capital": 100_000,
            "rel_commission": 0.1,
            "params": {
                "atrPeriod": {"v": 98, "min": 10, "max": 600, "type": "int"},
                "smaPeriod": {"v": 366, "min": 10, "max": 600, "type": "int"},
                "skipValue": {"v": 98, "min": 10, "max": 600, "type": "int"},
                "maxPercentRisk": {"v": 2.72, "min": 1, "max": 5, "type": "float"}
            },
            "interval": [0, 1]
        },

        "METRICS": [
            "calmar_coeff_start_capital",
            "graal_metr_no_reinvest",
            "graal_metr_with_reinvest"
        ]
    }

    
    # Создаю объект для получения свечек
    data_provider = DataProvider()

    # Получаю свечки
    print("\nData:")
    bars_df = data_provider.get(data_params=EXPERIMENT["DATA"])
    
    print(f"    > symbol:     {EXPERIMENT['DATA']['symbol']}")
    print(f"    > timeframe:  {EXPERIMENT['DATA']['timeframe']}")
    print(f"    > data.shape: {bars_df.shape}")

    run_strategy(experiment=EXPERIMENT, bars_df=bars_df)
