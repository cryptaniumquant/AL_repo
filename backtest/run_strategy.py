from strategies.FoBo.FoBo import FoBo
from pathlib import Path
import pandas as pd
from prepare_data import prepare_data
from performance_metrics import performance_metrics_new, strategy_charts
import numpy as np
import random
from deap import base, creator, tools, algorithms
import time


def run_strategy(bars_df, params, interval):
    strategy = FoBo(
        start_capital = 100_000,
        rel_commission = 0.1,
        params = params,
        is_optimization=True
    )

    # s = time.time()
    strategy.run(bars_df, interval)
    # print(f"Преобразование дат выполнилось за {time.time() - s:.4f} сек.")
    
    Date_np = bars_df['Date_dt'].to_numpy()
    Date_pd = bars_df['Date_dt']
    Date_dt = np.array(Date_pd.dt.to_pydatetime())
    
    
    metrics_calc = performance_metrics_new.PerformanceMetrics_new(
        start_capital=100_000,
        Date_np=Date_np,
        Date_pd=Date_pd,
        Date_dt=Date_dt,
        net_profit_punkt_arr=strategy.net_profit,
        net_profit_punkt_fixed_arr=strategy.net_profit_fixed,
        trades_count=len(strategy.positions)
    )
    
    return metrics_calc.calmar_coeff_start_capital

def cast_params_to_types(experiment_params, params):
    for param_name in params:
        if experiment_params[param_name][-1] == "int":
            params[param_name] = int(params[param_name])

def params_fit_in_intervals(experiment_params, params) -> bool:
    count = 0
    for param_name in params:
        if experiment_params[param_name][1] < params[param_name] and params[param_name] < experiment_params[param_name][2]:
            count += 1

    if count == len(params):
        return True
    else:
        return False

if __name__ == "__main__":
    EXPERIMENT = {
        "PARAMS": {
            "atrPeriod": (112, 10, 600, "int"),
            "smaPeriod": (432, 10, 600, "int"),
            #"skipValue": (118, 10, 600, "int"),
            "c1": (1.5, 1.3, 1.7, "float"),
            "c2": (2.5, 2.3, 2.7, "float"),
            "maxPercentRisk": (3.1, 1, 5, "float")
        }
    }

    PATH_TO_DATA = Path(__file__).parent.joinpath("data", "SUIUSDT.csv")
    STRATEGY_NAME = "FoBo"

    bars_df = pd.read_csv(PATH_TO_DATA)
    bars_df = prepare_data(bars_df, timeframe='60min')
    print(f"Data shape: {bars_df.shape}")
    # Определение типов для DEAP
    creator.create("FitnessMax", base.Fitness, weights=(1.0,), module=__name__)
    creator.create("Individual", list, fitness=creator.FitnessMax, module=__name__)

    toolbox = base.Toolbox()

    # Определение генерации параметров
    for param_name in EXPERIMENT["PARAMS"]:
        _, start, end, param_type = EXPERIMENT["PARAMS"][param_name]

        if param_type == "int":
            toolbox.register(param_name, random.randint, start, end)
        elif param_type == "float":
            toolbox.register(param_name, random.uniform, start, end)

    # Создание индивидуума
    toolbox.register("individual", tools.initCycle, creator.Individual,
                    [getattr(toolbox, param_name) for param_name in EXPERIMENT["PARAMS"]], n=1)
   
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Функция оценки (фитнес-функция)
    def evaluate(individual):
        # Получаем параметры и даём им имена
        params = dict(zip(EXPERIMENT["PARAMS"], individual))

        for param_name in EXPERIMENT["PARAMS"]:
            # Переводи параметр в int (если нужно)
            cast_params_to_types(EXPERIMENT["PARAMS"], params)
            # Проверка, вписывается ли параметр в допустимые значения
            if not params_fit_in_intervals(EXPERIMENT["PARAMS"], params):
                return (-100, )
            

        print(params)
        calmar1 = run_strategy(bars_df, params, interval=[0, 1/3])
        calmar2 = run_strategy(bars_df, params, interval=[1/3, 2/3])
        
        
        print(params, f"{calmar1:.2f} | {calmar2:.2f}")
        result = (calmar1 + calmar2) - abs(calmar1 - calmar2)
        return (result, )

    toolbox.register("evaluate", evaluate)

    # Crossover
    #toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mate", tools.cxUniform, indpb=0.5)

    # Mutation
    toolbox.register("mutate", tools.mutPolynomialBounded, low=10, up=1000, eta=25.0, indpb=0.3)
    # toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.2, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=5)

    pop_size = 200
    n_generations = 4
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
        # Соединяем гены индивида и имена параметров
        params = dict(zip(EXPERIMENT["PARAMS"], individual))
        cast_params_to_types(EXPERIMENT["PARAMS"], params)

        # Проверка, вписывается ли параметр в допустимые значения
        if params_fit_in_intervals(EXPERIMENT["PARAMS"], params):
        #if params[param_name] < EXPERIMENT["PARAMS"][param_name][2] and params[param_name] > EXPERIMENT["PARAMS"][param_name][1]:
            print("TEST", params)
            # Добавляем к индивиду его score
            params["fitness"] = individual.fitness.values[0]

            # Считаем Кальмара на тестовом участке
            params["calmar1"] = run_strategy(bars_df=bars_df, params=params, interval=[0, 1/3])
            params["calmar2"] = run_strategy(bars_df=bars_df, params=params, interval=[1/3, 2/3])
            params["calmar3"] = run_strategy(bars_df=bars_df, params=params, interval=[2/3, 1])
            
            params["calmar_avg"] = (params["calmar1"] + params["calmar2"] + params["calmar3"])/3
            df.append(params)
    
    print("Saving backtest_results.csv ...")
    df = pd.DataFrame(df).sort_values(by="calmar_avg", ascending=False)
    df.to_csv("backtest_results.csv", index=False)
