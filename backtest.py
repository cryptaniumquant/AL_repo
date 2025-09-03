from deap import base, creator, tools, algorithms
from backtest.utils import cast_params_to_types, params_fit_in_intervals
import path
from data.DataProvider import DataProvider
from run_strategy import run_strategy
import numpy as np
import random
import pandas as pd
import importlib


def run_backtest(experiment):
    data_provider = DataProvider()
    bars_df = data_provider.get(data_params=experiment["DATA"])
    print(bars_df)
    # Импортируем класс стратегии
    start_capital = experiment["STRATEGY"]["start_capital"]
    rel_commission = experiment["STRATEGY"]["rel_commission"]
    strategy_class_s = experiment["STRATEGY"]["strategy_class"]
    module = importlib.import_module(f'strategies.{strategy_class_s}.{strategy_class_s}')
    strategy_class = getattr(module, strategy_class_s)
    strategy = strategy_class(start_capital, rel_commission, is_optimization=True)

    # Создаем задачу
    creator.create("FitnessMax", base.Fitness, weights=(1.0,), module=__name__)
    creator.create("Individual", list, fitness=creator.FitnessMax, module=__name__)

    toolbox = base.Toolbox()
    
    # Определение генерации параметров
    for param_name in experiment["STRATEGY"]["params"]:
        #_, start, end, param_type = experiment["STRATEGY"]["params"][param_name]
        min_v = experiment["STRATEGY"]["params"][param_name]["min"]
        max_v = experiment["STRATEGY"]["params"][param_name]["max"]
        param_type = experiment["STRATEGY"]["params"][param_name]["type"]

        if param_type == "int":
            toolbox.register(param_name, random.randint, min_v, max_v)
        elif param_type == "float":
            toolbox.register(param_name, random.uniform, min_v, max_v)

    # Создание индивидуума
    toolbox.register("individual", tools.initCycle, creator.Individual,
                    [getattr(toolbox, param_name) for param_name in experiment["STRATEGY"]["params"]], n=1)
   
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Функция оценки (фитнес-функция)
    def evaluate(individual):
        # Получаем параметры и даём им имена
        params_ind = dict(zip(experiment["STRATEGY"]["params"], individual))

        for param_name in params_ind:
            experiment["STRATEGY"]["params"][param_name]["v"] = params_ind[param_name]

        # Кастуем параметры к нужным типам
        cast_params_to_types(experiment["STRATEGY"]["params"])
        # Проверка, вписывается ли параметр в допустимые значения
        if not params_fit_in_intervals(experiment["STRATEGY"]["params"]):
            return (-100, )
            

        calmar1 = strategy.run(
            params=experiment["STRATEGY"]["params"],
            bars_df=bars_df,
            interval=[0, 1/3],
            metrics=experiment["METRICS"])["calmar_coeff_start_capital"]
        # print("1-run", len(strategy.positions))
        calmar2 = strategy.run(
            params=experiment["STRATEGY"]["params"],
            bars_df=bars_df,
            interval=[1/3, 2/3],
            metrics=experiment["METRICS"])["calmar_coeff_start_capital"]
        # print("2-run", len(strategy.positions))
        print([(pname, experiment["STRATEGY"]["params"][pname]["v"]) for pname in experiment["STRATEGY"]["params"]], f"{calmar1:.2f} | {calmar2:.2f}")
        result = (calmar1 + calmar2) - abs(calmar1 - calmar2)
        return (result, )

    toolbox.register("evaluate", evaluate)

    # Crossover
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    # toolbox.register("mate", tools.cxUniform, indpb=0.5)

    # Mutation
    #toolbox.register("mutate", tools.mutPolynomialBounded, low=10, up=1000, eta=25.0, indpb=0.3)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.2, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=5)

    pop_size = experiment["POP_SIZE"]
    n_generations = experiment["N_GENERATIONS"]
    # Создание начальной популяции
    population = toolbox.population(n=pop_size)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("max", np.max)

    # Запуск генетического алгоритма
    population, logbook = algorithms.eaSimple(population, toolbox,
                                            cxpb=0.7,  # Вероятность кроссовера
                                            mutpb=0.3,  # Вероятность мутации
                                            ngen=n_generations,
                                            stats=stats,
                                            verbose=True)

    # Получаем самые крутые особи
    df = []
    top_individuals = tools.selBest(population, k=pop_size)
    for idx, individual in enumerate(top_individuals):
        params_ind = dict(zip(experiment["STRATEGY"]["params"], individual))
        for param_name in params_ind:
            experiment["STRATEGY"]["params"][param_name]["v"] = params_ind[param_name]
        cast_params_to_types(experiment["STRATEGY"]["params"])

        # Проверка, вписывается ли параметр в допустимые значения
        if params_fit_in_intervals(experiment["STRATEGY"]["params"]):
            
            # Соединяем гены индивида и имена параметров
            for param_name in params_ind:
                params_ind[param_name] = experiment["STRATEGY"]["params"][param_name]["v"]

            print("RESULT", params_ind)
            # Добавляем к индивиду его score
            params_ind["fitness"] = individual.fitness.values[0]

            # Считаем Кальмара на тестовом участке
            params_ind["calmar1"] = strategy.run(params=experiment["STRATEGY"]["params"], bars_df=bars_df, interval=[0, 1/3], metrics=experiment["METRICS"])["calmar_coeff_start_capital"]
            params_ind["calmar2"] = strategy.run(params=experiment["STRATEGY"]["params"], bars_df=bars_df, interval=[1/3, 2/3], metrics=experiment["METRICS"])["calmar_coeff_start_capital"]
            params_ind["calmar3"] = strategy.run(params=experiment["STRATEGY"]["params"], bars_df=bars_df, interval=[2/3, 1], metrics=experiment["METRICS"])["calmar_coeff_start_capital"]
            
            params_ind["calmar_avg"] = (params_ind["calmar1"] + params_ind["calmar2"] + params_ind["calmar3"])/3
            
            # Дополнительные поля (для информации)
            params_ind["symbol"] = experiment["DATA"]["symbol"]
            params_ind["timeframe"] = experiment["DATA"]["timeframe"]
            params_ind["from"] = bars_df["Date_dt"].min()
            params_ind["to"] = bars_df["Date_dt"].max()
            params_ind["strategy_class"] = experiment["STRATEGY"]["strategy_class"]
            print(f"    > {params_ind['calmar1']:.2f}", f"{params_ind['calmar2']:.2f}", f"{params_ind['calmar3']:.2f}")
            df.append(params_ind)
    
    print("Saving backtest_results.csv ...")
    df = pd.DataFrame(df).sort_values(by="calmar_avg", ascending=False)
    df = df.drop_duplicates()
    df.to_csv("backtest_results.csv", index=False)


if __name__ == "__main__":
    EXPERIMENT = {
        "DATA": {
            "symbol": "SOL/USDT",
            "timeframe": "1h",
            "timezone": "Asia/Bangkok",
            "start_date": "2023-01-01"
        },
        "STRATEGY": {
            "strategy_class": "FoBo",
            "start_capital": 100_000,
            "rel_commission": 0.1,
            "params": {
                "atrPeriod":      {"v": 178, "min": 10, "max": 600, "type": "int"},
                "smaPeriod":      {"v": 404, "min": 10, "max": 600, "type": "int"},
                "skipValue":      {"v": 42, "min": 10, "max": 600, "type": "int"},
                "maxPercentRisk": {"v": 3.04, "min": 1, "max": 5, "type": "float"}
            },
            "interval": [0, 1]
        },

        "METRICS": [
            "calmar_coeff_start_capital",
        ],

        "POP_SIZE": 200,
        "N_GENERATIONS": 3
    }

    EXPERIMENT = {
        "DATA": {
            "symbol": "SOL/USDT",
            "timeframe": "1h",
            "timezone": "Asia/Bangkok",
            "start_date": "2023-01-01"
        },
        "STRATEGY": {
            "strategy_class": "FoBo",
            "start_capital": 100_000,
            "rel_commission": 0.1,
            "params": {
                "atrPeriod":      {"v": 178, "min": 10, "max": 600, "type": "int"},
                "smaPeriod":      {"v": 404, "min": 10, "max": 600, "type": "int"},
                "skipValue":      {"v": 42, "min": 10, "max": 600, "type": "int"},
                "maxPercentRisk": {"v": 3.04, "min": 1, "max": 5, "type": "float"}
            },
            "interval": [0, 1]
        },

        "METRICS": [
            "calmar_coeff_start_capital",
        ],

        "POP_SIZE": 200,
        "N_GENERATIONS": 3
    }

    run_backtest(experiment=EXPERIMENT)
    # PATH_TO_DATA = Path(__file__).parent.joinpath("data", "SUIUSDT.csv")
    # STRATEGY_NAME = "FoBo"

    # bars_df = pd.read_csv(PATH_TO_DATA)
    # bars_df = prepare_data(bars_df, timeframe='60min')
    
    
