from deap import base, creator, tools, algorithms
from backtest.utils import cast_params_to_types, params_fit_in_intervals
import random
import pandas as pd
import numpy as np


def genetic(CONFIG, run_strategy, bars_df):

    # Создаем задачу
    creator.create("FitnessMax", base.Fitness, weights=(1.0,), module=__name__)
    creator.create("Individual", list, fitness=creator.FitnessMax, module=__name__)

    toolbox = base.Toolbox()

    # Определение генерации параметров
    for param_name in CONFIG["PARAMS"]:
        _, start, end, param_type = CONFIG["PARAMS"][param_name]

        if param_type == "int":
            toolbox.register(param_name, random.randint, start, end)
        elif param_type == "float":
            toolbox.register(param_name, random.uniform, start, end)

    # Создание индивидуума
    toolbox.register("individual", tools.initCycle, creator.Individual,
                    [getattr(toolbox, param_name) for param_name in CONFIG["PARAMS"]], n=1)
   
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Функция оценки (фитнес-функция)
    def evaluate(individual):
        # Получаем параметры и даём им имена
        params = dict(zip(CONFIG["PARAMS"], individual))

        for param_name in CONFIG["PARAMS"]:
            # Переводи параметр в int (если нужно)
            cast_params_to_types(CONFIG["PARAMS"], params)
            # Проверка, вписывается ли параметр в допустимые значения
            if not params_fit_in_intervals(CONFIG["PARAMS"], params):
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
        params = dict(zip(CONFIG["PARAMS"], individual))
        cast_params_to_types(CONFIG["PARAMS"], params)

        # Проверка, вписывается ли параметр в допустимые значения
        if params_fit_in_intervals(CONFIG["PARAMS"], params):
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
