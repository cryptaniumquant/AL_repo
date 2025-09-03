import matplotlib.pyplot as plt
import pandas as pd
from typing import Optional
from datetime import datetime
import matplotlib.dates as mdates
from datetime import datetime
import numpy as np
import matplotlib.gridspec as gridspec

def plot_graph_with_close(
    date_dt: np.ndarray,
    equity_np: np.ndarray,
    close_np: np.ndarray,
    x_label: str = 'Date',
    y_label: str = 'Equity Value',
    title: str = 'Equity and Close Over Time',
    legend_label_equity: str = 'Equity Strategy / B&H',
    color_equity: str = 'blue',
    color_close: str = 'red'
):
    # Проверка, что входные данные имеют одинаковую длину
    assert len(equity_np) == len(date_dt), "Длины входных массивов должны совпадать"

    # Создание графика, fig - объект фигуры, ax - объект осей.
    fig, ax = plt.subplots(figsize=(10, 6))  # Устанавливаем размер фигуры 10x6 дюймов

    # Построение графика: date_dt по оси X, equity_np по оси Y
    ax.plot(
        date_dt,  # Даты на оси X
        equity_np,  # Значения equity на оси Y
        label=legend_label_equity,  # Метка для легенды
        color=color_equity,  # Цвет линии
        marker=''  # Маркеры на точках данных
    )
    ax.plot(
        date_dt,  # Даты на оси X
        (close_np/close_np[0] - 1)*100,  # Значения equity на оси Y
        label='B&H, %',  # Метка для легенды
        color=color_close,  # Цвет линии
        marker=''  # Маркеры на точках данных
    )

    # Оформление оси X (даты)
    ax.xaxis.set_major_formatter(
        mdates.DateFormatter('%Y-%m-%d')  # Формат дат как 'год-месяц-день'
    )
    plt.xticks(rotation=45)  # Поворот меток оси X на 45 градусов для лучшей читаемости

    # Оформление оси Y (значения)
    ax.set_ylabel(y_label)  # Установка метки оси Y
    ax.set_xlabel(x_label)  # Установка метки оси X

    # Добавление сетки
    ax.grid(True)  # Включаем сетку на графике

    # Добавление заголовка и легенды
    ax.set_title(title)  # Установка заголовка графика
    ax.legend()  # Включаем легенду на графике

    # Автоматическое форматирование дат на оси X для лучшей читаемости
    fig.autofmt_xdate()

    # Показ графика
    plt.tight_layout()  # Улучшает компоновку элементов на графике, чтобы они не перекрывались
    plt.show()  # Отображает график
def plot_graph(date_dt: np.ndarray, equity_np: np.ndarray,
                x_label: str = 'Date',
                y_label: str = 'Equity Value',
                title: str = 'Equity Over Time',
                legend_label: str = 'Equity',
                color: str = 'blue'):
    # Проверка, что входные данные имеют одинаковую длину
    assert len(equity_np) == len(date_dt), "Длины входных массивов должны совпадать"

    # Создание графика, fig - объект фигуры, ax - объект осей.
    fig, ax = plt.subplots(figsize=(10, 6))  # Устанавливаем размер фигуры 10x6 дюймов

    # Построение графика: date_dt по оси X, equity_np по оси Y
    ax.plot(
        date_dt,  # Даты на оси X
        equity_np,  # Значения equity на оси Y
        label=legend_label,  # Метка для легенды
        color=color,  # Цвет линии
        marker=''  # Маркеры на точках данных
    )

    # Оформление оси X (даты)
    ax.xaxis.set_major_formatter(
        mdates.DateFormatter('%Y-%m-%d')  # Формат дат как 'год-месяц-день'
    )
    plt.xticks(rotation=45)  # Поворот меток оси X на 45 градусов для лучшей читаемости

    # Оформление оси Y (значения)
    ax.set_ylabel(y_label)  # Установка метки оси Y
    ax.set_xlabel(x_label)  # Установка метки оси X

    # Добавление сетки
    ax.grid(True)  # Включаем сетку на графике

    # Добавление заголовка и легенды
    ax.set_title(title)  # Установка заголовка графика
    ax.legend()  # Включаем легенду на графике

    # Автоматическое форматирование дат на оси X для лучшей читаемости
    fig.autofmt_xdate()

    # Показ графика
    plt.tight_layout()  # Улучшает компоновку элементов на графике, чтобы они не перекрывались
    plt.show()  # Отображает график




def plot_series_range(
        series: pd.Series,
        start_date: datetime,
        end_date: datetime,
        title: str = "NetProfit Over Period",
        xlabel: str = "Date",
        ylabel: str = "NetProfit ($)"
):
    """
    Строит график для pd.Series за определённый период времени.

    :param series: Временной ряд (pd.Series) с индексом типа datetime.
    :param start_date: Начальная дата периода (в формате datetime.datetime).
    :param end_date: Конечная дата периода (в формате datetime.datetime).
    :param title: Заголовок графика (по умолчанию "NetProfit Over Period").
    :param xlabel: Метка оси X (по умолчанию "Date").
    :param ylabel: Метка оси Y (по умолчанию "NetProfit ($)").
    """
    # Фильтрация данных по заданному периоду
    filtered_series = series[start_date:end_date]

    # Вычисление количества периодов
    num_periods = len(filtered_series)

    # Установка ширины столбцов на основе количества периодов
    bar_width = 1000 / num_periods if num_periods > 0 else 1
    if bar_width < 2:
        bar_width = 2
    elif bar_width > 50:
        bar_width = 50



    # Разделение на положительные и отрицательные значения
    pos_returns = filtered_series[filtered_series > 0]
    neg_returns = filtered_series[filtered_series < 0]

    # Построение графика
    plt.figure(figsize=(10, 6))

    # Положительные значения
    plt.bar(pos_returns.index, pos_returns.values, color='green', alpha=0.5, label='Positive Returns', width=bar_width)

    # Отрицательные значения
    plt.bar(neg_returns.index, neg_returns.values, color='red', alpha=0.5, label='Negative Returns', width=bar_width)

    # Надписи на графиках
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_strategy_results_grid(must_plot, metrics_calc, metrics_values_dict, strategy_name, _init_deposit, time_index):
    if must_plot:
        # Создаем полотно графика с использованием GridSpec
        fig = plt.figure(figsize=(10, 6))

        # Определяем сетку 2x2 с соотношением ширины и высоты столбцов/строк
        gs = gridspec.GridSpec(2, 2, width_ratios=[3, 1], height_ratios=[3.5, 1])

        # Создаем области для графиков
        ax1 = plt.subplot(gs[0, 0])  # Верхний левый график (первая строка, первый столбец)
        ax2 = plt.subplot(gs[1, 0])  # Нижний левый график (вторая строка, первый столбец)
        ax3 = plt.subplot(gs[:, 1])  # Правая область, занимает обе строки (весь второй столбец)

        # создаем датафрейм с индексом даты для корректной отиросвки
        temp_df = pd.DataFrame({'EqFix':metrics_calc.equity_punkt_fixed_arr, 'Eq':metrics_calc.equity_punkt_arr, 'EqMax': metrics_calc.max_equity_punkt_arr,
                                'DDfromStart':metrics_calc.drawdown_curve_pct_from_start_capital_arr, 'DDfromEq':metrics_calc.drawdown_curve_pct_from_max_eqty_arr })
        temp_df.index = time_index

        # Рисуем первый график (Equity)
        ax1.plot(
            temp_df.EqFix,
            label=f'Equity fixed ($) = NetProfitFixed ($) + Start deposit ($) ({_init_deposit})',
            lw=2,
            color='purple'
        )

        ax1.plot(
            temp_df.Eq,
            label=f'Equity ($) = NetProfit ($) + Start deposit ($) ({_init_deposit})',
            lw=2,
            color='green'
        )

        ax1.plot(
            temp_df.EqMax,
            label=f'Max Equity ($)',
            lw=2,
            color='red'
        )

        # Подписи осей и заголовок для первого графика
        ax1.set_xlabel('Бары')
        ax1.set_ylabel(f'Equity$')

        text_title = (f'Strategy: "{strategy_name}." '
                      f'Период тестирования: c '
                      f'{metrics_calc.start_time_strategy.strftime("%d-%m-%Y")} '
                      f'по '
                      f'{metrics_calc.end_time_strategy.strftime(("%d-%m-%Y"))} '
                      f'таймфрейм: {metrics_calc.timeframe_string}.')

        ax1.set_title(text_title)
        ax1.grid(True)
        ax1.legend()

        # Рисуем 1ый график на 2ом полотне (Drawdown curve)
        ax2.plot(
            temp_df.DDfromStart if metrics_values_dict['_pct_of_reinvest'] == 0
            else temp_df.DDfromEq,
            label='',
            lw=2,
            color='red'
        )

        # Подписи осей и заголовок для второго графика
        ax2.set_xlabel('Trading Period')
        ax2.set_ylabel('DrawDown (%)')
        ax2.set_title('Просадка (DrawDawn), %')
        ax2.grid(True)
        ax2.legend("")

        # Данные для таблицы
        table_data = [
            ['Метрика', 'Значение'],
            ['Symbol', f"{metrics_values_dict['COIN']}"],
            ['timeframe', f"{metrics_values_dict['timeframe']}"],
           # ['Дата тестирования:', f"с {metrics_values_dict['start_time_strategy'].strftime('%d-%m-%Y')} по {metrics_values_dict['end_time_strategy'].strftime('%d-%m-%Y')}"]
        #    ['Start торговли:', {metrics_calc.start_time_str}]
        #    ['End торговли:', {metrics_calc.end_time_str}]

            ['max риск на сделку, %', f"{metrics_values_dict['_max_pct_risk']:.2f}"],
            ['реинвестируем доход, %', f"{metrics_values_dict['_pct_of_reinvest']:.2f}"],

            ['Трейдов в год', f"{metrics_values_dict['trades_per_year']:.0f}"],

            ['Net Profit, %', f"{metrics_values_dict['net_profit_end_pct']:.2f}"],
            ['APR %', f"{metrics_values_dict['apr_pct']:.2f}"],
            ['max Drawdown, %',
                f"{metrics_values_dict['dd_worst_start_capital'] if metrics_values_dict['_pct_of_reinvest'] == 0 else metrics_values_dict['dd_worst_max_eqty']:.2f}"],
            ['Calmar Coeff',
             f"{metrics_values_dict['calmar_coeff_start_capital'] if metrics_values_dict['_pct_of_reinvest'] == 0 else metrics_values_dict['calmar_coeff_max_eqty']:.2f}"],

            ['Recovery Factor',
             f"{metrics_values_dict['recovery_pct_capital'] if metrics_values_dict['_pct_of_reinvest'] == 0 else metrics_values_dict['recovery_pct_eqty']:.2f}"],

            ['Sharpe Koeff', f"{metrics_values_dict['sharpe_month_days']:.2f}"],
         #   ['Sortino Koeff', f"{metrics_values_dict['sortino_month_days']:.2f}"],

            ['Прибыльно месяцев, %', f"{metrics_values_dict['months_plus_pct']:.2f}"],
            ['Прибыльно кварталов, %', f"{metrics_values_dict['quartals_plus_pct']:.2f}"],

            ['Beard Koeff', f"{metrics_values_dict['beard_coeff_daily']:.2f}"],
            ['Бород в год (в среднем)', f"{metrics_values_dict['daily_beards_per_year']:.2f}"],
            ['max борода, дней', f"{metrics_values_dict['daily_beard_max']:.2f}"],

            ['Graal Metr, %',
             f"""{metrics_values_dict['GraalMetr_NoReinvest'] 
             if metrics_values_dict['_pct_of_reinvest'] == 0 
             else metrics_values_dict['GraalMetr_WithReinvest']:.2f}"""]
        ]

        # Настраиваем область для таблицы
        ax3.axis('tight')  # Устанавливаем плотное (tight) расположение осей, чтобы минимизировать отступы
        ax3.axis('off')  # Отключаем отображение осей, чтобы убрать рамки и метки вокруг таблицы

        # Создаем таблицу
        table = ax3.table(
            cellText=table_data,  # Данные для заполнения таблицы
            colLabels=None,  # Заголовки столбцов (None, так как они уже включены в table_data)
            cellLoc='center',  # Размещение текста в ячейках по центру
            loc='center'  # Размещение таблицы по центру области ax3
        )

        # Настраиваем размер шрифта и масштаб таблицы
        table.auto_set_font_size(False)  # Отключаем автоматический выбор размера шрифта
        table.set_fontsize(9)  # Устанавливаем размер шрифта для текста в таблице
        table.scale(1.6, 1.2)  # Масштабируем таблицу по ширине и высоте (коэффициент 1.2)

        # Отображение графика
        plt.tight_layout()
        plt.show()

def plot_strategy_results_pcatr(must_plot, metrics_calc, metrics_values_dict, strategy_name, _init_deposit, time_index):
    if must_plot:
        # Создаем полотно графика с использованием GridSpec
        fig = plt.figure(figsize=(10, 6))

        # Определяем сетку 2x2 с соотношением ширины и высоты столбцов/строк
        gs = gridspec.GridSpec(2, 2, width_ratios=[3, 1], height_ratios=[3.5, 1])

        # Создаем области для графиков
        ax1 = plt.subplot(gs[0, 0])  # Верхний левый график (первая строка, первый столбец)
        ax2 = plt.subplot(gs[1, 0])  # Нижний левый график (вторая строка, первый столбец)
        ax3 = plt.subplot(gs[:, 1])  # Правая область, занимает обе строки (весь второй столбец)

        # создаем датафрейм с индексом даты для корректной отиросвки
        temp_df = pd.DataFrame({'EqFix':metrics_calc.equity_punkt_fixed_arr, 'Eq':metrics_calc.equity_punkt_arr, 'EqMax': metrics_calc.max_equity_punkt_arr,
                                'DDfromStart':metrics_calc.drawdown_curve_pct_from_start_capital_arr, 'DDfromEq':metrics_calc.drawdown_curve_pct_from_max_eqty_arr })
        temp_df.index = time_index

        # Рисуем первый график (Equity)
        ax1.plot(
            temp_df.EqFix,
            label=f'Equity fixed ($) = NetProfitFixed ($) + Start deposit ($) ({_init_deposit})',
            lw=2,
            color='purple'
        )

        ax1.plot(
            temp_df.Eq,
            label=f'Equity ($) = NetProfit ($) + Start deposit ($) ({_init_deposit})',
            lw=2,
            color='green'
        )

        ax1.plot(
            temp_df.EqMax,
            label=f'Max Equity ($)',
            lw=2,
            color='red'
        )

        # Подписи осей и заголовок для первого графика
        ax1.set_xlabel('Бары')
        ax1.set_ylabel(f'Equity$')

        text_title = (f'Strategy: "{strategy_name}." '
                      f'Период тестирования: c '
                      f'{metrics_calc.start_time_strategy.strftime("%d-%m-%Y")} '
                      f'по '
                      f'{metrics_calc.end_time_strategy.strftime(("%d-%m-%Y"))} '
                      f'таймфрейм: {metrics_calc.timeframe_string}.')

        ax1.set_title(text_title)
        ax1.grid(True)
        ax1.legend()

        # Рисуем 1ый график на 2ом полотне (Drawdown curve)
        ax2.plot(
            temp_df.DDfromStart if metrics_values_dict['_pct_of_reinvest'] == 0
            else temp_df.DDfromEq,
            label='',
            lw=2,
            color='red'
        )

        # Подписи осей и заголовок для второго графика
        ax2.set_xlabel('Trading Period')
        ax2.set_ylabel('DrawDown (%)')
        ax2.set_title('Просадка (DrawDawn), %')
        ax2.grid(True)
        ax2.legend("")

        # Данные для таблицы
        table_data = [
            ['Метрика', 'Значение'],
            ['Symbol', f"{metrics_values_dict['COIN']}"],
            ['timeframe', f"{metrics_values_dict['timeframe']}"],
           # ['Дата тестирования:', f"с {metrics_values_dict['start_time_strategy'].strftime('%d-%m-%Y')} по {metrics_values_dict['end_time_strategy'].strftime('%d-%m-%Y')}"]
        #    ['Start торговли:', {metrics_calc.start_time_str}]
        #    ['End торговли:', {metrics_calc.end_time_str}]

            ['_koeff', f"{metrics_values_dict['_koeff']:.2f}"],
            ['_dividerAtr', f"{metrics_values_dict['_dividerAtr']:.0f}"],
            ['_periodAtr', f"{metrics_values_dict['_periodAtr']:.0f}"],
            ['_periodEnterPC', f"{metrics_values_dict['_periodEnterPC']:.0f}"],
            ['max риск на сделку, %', f"{metrics_values_dict['_max_pct_risk']:.2f}"],
            ['реинвестируем доход, %', f"{metrics_values_dict['_pct_of_reinvest']:.2f}"],

            ['Трейдов в год', f"{metrics_values_dict['trades_per_year']:.0f}"],

            ['Net Profit, %', f"{metrics_values_dict['net_profit_end_pct']:.2f}"],
            ['APR %', f"{metrics_values_dict['apr_pct']:.2f}"],
            ['max Drawdown, %',
                f"{metrics_values_dict['dd_worst_start_capital'] if metrics_values_dict['_pct_of_reinvest'] == 0 else metrics_values_dict['dd_worst_max_eqty']:.2f}"],
            ['Calmar Coeff',
             f"{metrics_values_dict['calmar_coeff_start_capital'] if metrics_values_dict['_pct_of_reinvest'] == 0 else metrics_values_dict['calmar_coeff_max_eqty']:.2f}"],

            ['Recovery Factor',
             f"{metrics_values_dict['recovery_pct_capital'] if metrics_values_dict['_pct_of_reinvest'] == 0 else metrics_values_dict['recovery_pct_eqty']:.2f}"],

            ['Sharpe Koeff', f"{metrics_values_dict['sharpe_month_days']:.2f}"],
         #   ['Sortino Koeff', f"{metrics_values_dict['sortino_month_days']:.2f}"],

            ['Прибыльно месяцев, %', f"{metrics_values_dict['months_plus_pct']:.2f}"],
            ['Прибыльно кварталов, %', f"{metrics_values_dict['quartals_plus_pct']:.2f}"],

            ['Beard Koeff', f"{metrics_values_dict['beard_coeff_daily']:.2f}"],
            ['Бород в год (в среднем)', f"{metrics_values_dict['daily_beards_per_year']:.2f}"],
            ['max борода, дней', f"{metrics_values_dict['daily_beard_max']:.2f}"],

            ['Graal Metr, %',
             f"""{metrics_values_dict['GraalMetr_NoReinvest'] 
             if metrics_values_dict['_pct_of_reinvest'] == 0 
             else metrics_values_dict['GraalMetr_WithReinvest']:.2f}"""]
        ]

        # Настраиваем область для таблицы
        ax3.axis('tight')  # Устанавливаем плотное (tight) расположение осей, чтобы минимизировать отступы
        ax3.axis('off')  # Отключаем отображение осей, чтобы убрать рамки и метки вокруг таблицы

        # Создаем таблицу
        table = ax3.table(
            cellText=table_data,  # Данные для заполнения таблицы
            colLabels=None,  # Заголовки столбцов (None, так как они уже включены в table_data)
            cellLoc='center',  # Размещение текста в ячейках по центру
            loc='center'  # Размещение таблицы по центру области ax3
        )

        # Настраиваем размер шрифта и масштаб таблицы
        table.auto_set_font_size(False)  # Отключаем автоматический выбор размера шрифта
        table.set_fontsize(9)  # Устанавливаем размер шрифта для текста в таблице
        table.scale(1.6, 1.2)  # Масштабируем таблицу по ширине и высоте (коэффициент 1.2)

        # Отображение графика
        plt.tight_layout()
        plt.show()

def plot_strategy_results_fluger(must_plot, metrics_calc, metrics_values_dict, strategy_name, _init_deposit, time_index):
    if must_plot:
        # Создаем полотно графика с использованием GridSpec
        fig = plt.figure(figsize=(10, 6))

        # Определяем сетку 2x2 с соотношением ширины и высоты столбцов/строк
        gs = gridspec.GridSpec(2, 2, width_ratios=[3, 1], height_ratios=[3.5, 1])

        # Создаем области для графиков
        ax1 = plt.subplot(gs[0, 0])  # Верхний левый график (первая строка, первый столбец)
        ax2 = plt.subplot(gs[1, 0])  # Нижний левый график (вторая строка, первый столбец)
        ax3 = plt.subplot(gs[:, 1])  # Правая область, занимает обе строки (весь второй столбец)

        # создаем датафрейм с индексом даты для корректной отрисовки
        temp_df = pd.DataFrame(
            {
                'EqFix':metrics_calc.equity_punkt_fixed_arr,
                'Eq':metrics_calc.equity_punkt_arr,
                'EqMax': metrics_calc.max_equity_punkt_arr,
                'DDfromStart':metrics_calc.drawdown_curve_pct_from_start_capital_arr,
                'DDfromEq':metrics_calc.drawdown_curve_pct_from_max_eqty_arr
            }
        )
        temp_df.index = time_index

        # Рисуем первый график (Equity)
        ax1.plot(
            temp_df.EqFix,
            label=f'Equity fixed ($) = NetProfitFixed ($) + Start deposit ($) ({_init_deposit})',
            lw=2,
            color='purple'
        )

        ax1.plot(
            temp_df.Eq,
            label=f'Equity ($) = NetProfit ($) + Start deposit ($) ({_init_deposit})',
            lw=2,
            color='green'
        )

        ax1.plot(
            temp_df.EqMax,
            label=f'Max Equity ($)',
            lw=2,
            color='red'
        )

        # Подписи осей и заголовок для первого графика
        ax1.set_xlabel('Бары')
        ax1.set_ylabel(f'Equity$')

        text_title = (f'Strategy: "{strategy_name}." '
                      f'Период тестирования: c '
                      f'{metrics_calc.start_time_strategy.strftime("%d-%m-%Y")} '
                      f'по '
                      f'{metrics_calc.end_time_strategy.strftime(("%d-%m-%Y"))} '
                      f'таймфрейм: {metrics_calc.timeframe_string}.')

        ax1.set_title(text_title)
        ax1.grid(True)
        ax1.legend()

        # Рисуем 1ый график на 2ом полотне (Drawdown curve)
        ax2.plot(
            temp_df.DDfromStart if metrics_values_dict['_pct_of_reinvest'] == 0
            else temp_df.DDfromEq,
            label='',
            lw=2,
            color='red'
        )

        # Подписи осей и заголовок для второго графика
        ax2.set_xlabel('Trading period')
        ax2.set_ylabel('DrawDown (%)')
        ax2.set_title('Просадка (DrawDawn), %')
        ax2.grid(True)
        ax2.legend("")

        # Данные для таблицы
        table_data = [
            ['Метрика', 'Значение'],
            ['Symbol', f"{metrics_values_dict['COIN']}"],
            ['timeframe', f"{metrics_values_dict['timeframe']}"],
           # ['Дата тестирования:', f"с {metrics_values_dict['start_time_strategy'].strftime('%d-%m-%Y')} по {metrics_values_dict['end_time_strategy'].strftime('%d-%m-%Y')}"]
        #    ['Start торговли:', {metrics_calc.start_time_str}]
        #    ['End торговли:', {metrics_calc.end_time_str}]


            ['max риск на сделку, %', f"{metrics_values_dict['_max_pct_risk']:.2f}"],
            ['реинвестируем доход, %', f"{metrics_values_dict['_pct_of_reinvest']:.2f}"],

            ['Трейдов в год', f"{metrics_values_dict['trades_per_year']:.0f}"],

            ['Net Profit, %', f"{metrics_values_dict['net_profit_end_pct']:.2f}"],
            ['APR %', f"{metrics_values_dict['apr_pct']:.2f}"],
            ['max Drawdown, %',
                f"{metrics_values_dict['dd_worst_start_capital'] if metrics_values_dict['_pct_of_reinvest'] == 0 else metrics_values_dict['dd_worst_max_eqty']:.2f}"],
            ['Calmar Coeff',
             f"{metrics_values_dict['calmar_coeff_start_capital'] if metrics_values_dict['_pct_of_reinvest'] == 0 else metrics_values_dict['calmar_coeff_max_eqty']:.2f}"],

            ['Recovery Factor',
             f"{metrics_values_dict['recovery_pct_capital'] if metrics_values_dict['_pct_of_reinvest'] == 0 else metrics_values_dict['recovery_pct_eqty']:.2f}"],

            ['Sharpe Koeff', f"{metrics_values_dict['sharpe_month_days']:.2f}"],
         #   ['Sortino Koeff', f"{metrics_values_dict['sortino_month_days']:.2f}"],

            ['Прибыльно месяцев, %', f"{metrics_values_dict['months_plus_pct']:.2f}"],
            ['Прибыльно кварталов, %', f"{metrics_values_dict['quartals_plus_pct']:.2f}"],

            ['Beard Koeff', f"{metrics_values_dict['beard_coeff_daily']:.2f}"],
            ['Бород в год (в среднем)', f"{metrics_values_dict['daily_beards_per_year']:.2f}"],
            ['max борода, дней', f"{metrics_values_dict['daily_beard_max']:.2f}"],

            ['Graal Metr, %',
             f"""{metrics_values_dict['GraalMetr_NoReinvest'] 
             if metrics_values_dict['_pct_of_reinvest'] == 0 
             else metrics_values_dict['GraalMetr_WithReinvest']:.2f}"""]
        ]

        # Настраиваем область для таблицы
        ax3.axis('tight')  # Устанавливаем плотное (tight) расположение осей, чтобы минимизировать отступы
        ax3.axis('off')  # Отключаем отображение осей, чтобы убрать рамки и метки вокруг таблицы

        # Создаем таблицу
        table = ax3.table(
            cellText=table_data,  # Данные для заполнения таблицы
            colLabels=None,  # Заголовки столбцов (None, так как они уже включены в table_data)
            cellLoc='center',  # Размещение текста в ячейках по центру
            loc='center'  # Размещение таблицы по центру области ax3
        )

        # Настраиваем размер шрифта и масштаб таблицы
        table.auto_set_font_size(False)  # Отключаем автоматический выбор размера шрифта
        table.set_fontsize(9)  # Устанавливаем размер шрифта для текста в таблице
        table.scale(1.6, 1.2)  # Масштабируем таблицу по ширине и высоте (коэффициент 1.2)

        # Отображение графика
        plt.tight_layout()
        plt.show()

def plot_strategy_results_var(must_plot, metrics_calc, metrics_values_dict, strategy_name, _init_deposit,
                                     time_index):
        if must_plot:
            # Создаем полотно графика с использованием GridSpec
            fig = plt.figure(figsize=(10, 6))

            # Определяем сетку 2x2 с соотношением ширины и высоты столбцов/строк
            gs = gridspec.GridSpec(2, 2, width_ratios=[3, 1], height_ratios=[3.5, 1])

            # Создаем области для графиков
            ax1 = plt.subplot(gs[0, 0])  # Верхний левый график (первая строка, первый столбец)
            ax2 = plt.subplot(gs[1, 0])  # Нижний левый график (вторая строка, первый столбец)
            ax3 = plt.subplot(gs[:, 1])  # Правая область, занимает обе строки (весь второй столбец)

            # создаем датафрейм с индексом даты для корректной отрисовки
            temp_df = pd.DataFrame(
                {
                    'Eq': metrics_calc.equity_punkt_arr,
                    'EqMax': metrics_calc.max_equity_punkt_arr,
                    'DDfromStart': metrics_calc.drawdown_curve_pct_from_start_capital_arr,
                    'DDfromEq': metrics_calc.drawdown_curve_pct_from_max_eqty_arr
                }
            )
            temp_df.index = time_index

            # Рисуем первый график (Equity)

            ax1.plot(
                temp_df.Eq,
                label=f'Equity ($) = NetProfit ($) + Start deposit ($) ({_init_deposit})',
                lw=2,
                color='green'
            )

            ax1.plot(
                temp_df.EqMax,
                label=f'Max Equity ($)',
                lw=2,
                color='red'
            )

            # Подписи осей и заголовок для первого графика
            ax1.set_xlabel('Бары')
            ax1.set_ylabel(f'Equity$')

            text_title = (f'Strategy: "{strategy_name}." '
                          f'Период тестирования: c '
                          f'{metrics_calc.start_time_strategy.strftime("%d-%m-%Y")} '
                          f'по '
                          f'{metrics_calc.end_time_strategy.strftime(("%d-%m-%Y"))} '
                          f'таймфрейм: {metrics_calc.timeframe_string}.')

            ax1.set_title(text_title)
            ax1.grid(True)
            ax1.legend()

            # Рисуем 1ый график на 2ом полотне (Drawdown curve)
            ax2.plot(
                temp_df.DDfromStart if metrics_values_dict['_pct_of_reinvest'] == 0
                else temp_df.DDfromEq,
                label='',
                lw=2,
                color='red'
            )

            # Подписи осей и заголовок для второго графика
            ax2.set_xlabel('Trading period')
            ax2.set_ylabel('DrawDown (%)')
            ax2.set_title('Просадка (DrawDawn), %')
            ax2.grid(True)
            ax2.legend("")

            # Данные для таблицы
            table_data = [
                ['Метрика', 'Значение'],
                ['Symbol', f"{metrics_values_dict['COIN']}"],
                ['timeframe', f"{metrics_values_dict['timeframe']}"],
                # ['Дата тестирования:', f"с {metrics_values_dict['start_time_strategy'].strftime('%d-%m-%Y')} по {metrics_values_dict['end_time_strategy'].strftime('%d-%m-%Y')}"]
                #    ['Start торговли:', {metrics_calc.start_time_str}]
                #    ['End торговли:', {metrics_calc.end_time_str}]

                ['_period_var', f"{metrics_values_dict['_period_var']:.0f}"],
                ['_conf_lvl_enter', f"{metrics_values_dict['_conf_lvl_enter']:.2f}"],
                ['_conf_lvl_exit', f"{metrics_values_dict['_conf_lvl_exit']:.2f}"],
                ['max риск на сделку, %', f"{metrics_values_dict['_max_pct_risk']:.2f}"],
                ['реинвестируем доход, %', f"{metrics_values_dict['_pct_of_reinvest']:.2f}"],

                ['Трейдов в год', f"{metrics_values_dict['trades_per_year']:.0f}"],

                ['Net Profit, %', f"{metrics_values_dict['net_profit_end_pct']:.2f}"],
                ['APR %', f"{metrics_values_dict['apr_pct']:.2f}"],
                ['max Drawdown, %',
                 f"{metrics_values_dict['dd_worst_start_capital'] if metrics_values_dict['_pct_of_reinvest'] == 0 else metrics_values_dict['dd_worst_max_eqty']:.2f}"],
                ['Calmar Coeff',
                 f"{metrics_values_dict['calmar_coeff_start_capital'] if metrics_values_dict['_pct_of_reinvest'] == 0 else metrics_values_dict['calmar_coeff_max_eqty']:.2f}"],

                ['Recovery Factor',
                 f"{metrics_values_dict['recovery_pct_capital'] if metrics_values_dict['_pct_of_reinvest'] == 0 else metrics_values_dict['recovery_pct_eqty']:.2f}"],

                ['Sharpe Koeff', f"{metrics_values_dict['sharpe_month_days']:.2f}"],
                #   ['Sortino Koeff', f"{metrics_values_dict['sortino_month_days']:.2f}"],

                ['Прибыльно месяцев, %', f"{metrics_values_dict['months_plus_pct']:.2f}"],
                ['Прибыльно кварталов, %', f"{metrics_values_dict['quartals_plus_pct']:.2f}"],

                ['Beard Koeff', f"{metrics_values_dict['beard_coeff_daily']:.2f}"],
                ['Бород в год (в среднем)', f"{metrics_values_dict['daily_beards_per_year']:.2f}"],
                ['max борода, дней', f"{metrics_values_dict['daily_beard_max']:.2f}"],

                ['Graal Metr, %',
                 f"""{metrics_values_dict['GraalMetr_NoReinvest']
                 if metrics_values_dict['_pct_of_reinvest'] == 0
                 else metrics_values_dict['GraalMetr_WithReinvest']:.2f}"""]
            ]

            # Настраиваем область для таблицы
            ax3.axis('tight')  # Устанавливаем плотное (tight) расположение осей, чтобы минимизировать отступы
            ax3.axis('off')  # Отключаем отображение осей, чтобы убрать рамки и метки вокруг таблицы

            # Создаем таблицу
            table = ax3.table(
                cellText=table_data,  # Данные для заполнения таблицы
                colLabels=None,  # Заголовки столбцов (None, так как они уже включены в table_data)
                cellLoc='center',  # Размещение текста в ячейках по центру
                loc='center'  # Размещение таблицы по центру области ax3
            )

            # Настраиваем размер шрифта и масштаб таблицы
            table.auto_set_font_size(False)  # Отключаем автоматический выбор размера шрифта
            table.set_fontsize(9)  # Устанавливаем размер шрифта для текста в таблице
            table.scale(1.6, 1.2)  # Масштабируем таблицу по ширине и высоте (коэффициент 1.2)

            # Отображение графика
            plt.tight_layout()
            plt.show()


def plot_strategy_results_grid_avg(must_plot, metrics_calc, metrics_values_dict, strategy_name, _init_deposit,
                              time_index):
    if must_plot:
        # Создаем полотно графика с использованием GridSpec
        fig = plt.figure(figsize=(10, 6))

        # Определяем сетку 2x2 с соотношением ширины и высоты столбцов/строк
        gs = gridspec.GridSpec(2, 2, width_ratios=[3, 1], height_ratios=[3.5, 1])

        # Создаем области для графиков
        ax1 = plt.subplot(gs[0, 0])  # Верхний левый график (первая строка, первый столбец)
        ax2 = plt.subplot(gs[1, 0])  # Нижний левый график (вторая строка, первый столбец)
        ax3 = plt.subplot(gs[:, 1])  # Правая область, занимает обе строки (весь второй столбец)

        # создаем датафрейм с индексом даты для корректной отрисовки
        temp_df = pd.DataFrame(
            {
                'Eq': metrics_calc.equity_punkt_arr,
                'EqMax': metrics_calc.max_equity_punkt_arr,
                'DDfromStart': metrics_calc.drawdown_curve_pct_from_start_capital_arr,
                'DDfromEq': metrics_calc.drawdown_curve_pct_from_max_eqty_arr
            }
        )
        temp_df.index = time_index

        # Рисуем первый график (Equity)

        ax1.plot(
            temp_df.Eq,
            label=f'Equity ($) = NetProfit ($) + Start deposit ($) ({_init_deposit})',
            lw=2,
            color='green'
        )

        ax1.plot(
            temp_df.EqMax,
            label=f'Max Equity ($)',
            lw=2,
            color='red'
        )

        # Подписи осей и заголовок для первого графика
        ax1.set_xlabel('Бары')
        ax1.set_ylabel(f'Equity$')

        text_title = (f'Strategy: "{strategy_name}." '
                      f'Период тестирования: c '
                      f'{metrics_calc.start_time_strategy.strftime("%d-%m-%Y")} '
                      f'по '
                      f'{metrics_calc.end_time_strategy.strftime(("%d-%m-%Y"))} '
                      f'таймфрейм: {metrics_calc.timeframe_string}.')

        ax1.set_title(text_title)
        ax1.grid(True)
        ax1.legend()

        # Рисуем 1ый график на 2ом полотне (Drawdown curve)
        ax2.plot(
            temp_df.DDfromStart if metrics_values_dict['_pct_of_reinvest'] == 0
            else temp_df.DDfromEq,
            label='',
            lw=2,
            color='red'
        )

        # Подписи осей и заголовок для второго графика
        ax2.set_xlabel('Trading period')
        ax2.set_ylabel('DrawDown (%)')
        ax2.set_title('Просадка (DrawDawn), %')
        ax2.grid(True)
        ax2.legend("")

        # Данные для таблицы
        table_data = [
            ['Метрика', 'Значение'],
            ['Symbol', f"{metrics_values_dict['COIN']}"],
            ['timeframe', f"{metrics_values_dict['timeframe']}"],
            # ['Дата тестирования:', f"с {metrics_values_dict['start_time_strategy'].strftime('%d-%m-%Y')} по {metrics_values_dict['end_time_strategy'].strftime('%d-%m-%Y')}"]
            #    ['Start торговли:', {metrics_calc.start_time_str}]
            #    ['End торговли:', {metrics_calc.end_time_str}]

            ['реинвестируем доход, %', f"{metrics_values_dict['_pct_of_reinvest']:.2f}"],

            ['Трейдов в год', f"{metrics_values_dict['trades_per_year']:.0f}"],

            ['Net Profit, %', f"{metrics_values_dict['net_profit_end_pct']:.2f}"],
            ['APR %', f"{metrics_values_dict['apr_pct']:.2f}"],
            ['max Drawdown, %',
             f"{metrics_values_dict['dd_worst_start_capital'] if metrics_values_dict['_pct_of_reinvest'] == 0 else metrics_values_dict['dd_worst_max_eqty']:.2f}"],
            ['Calmar Coeff',
             f"{metrics_values_dict['calmar_coeff_start_capital'] if metrics_values_dict['_pct_of_reinvest'] == 0 else metrics_values_dict['calmar_coeff_max_eqty']:.2f}"],

            ['Recovery Factor',
             f"{metrics_values_dict['recovery_pct_capital'] if metrics_values_dict['_pct_of_reinvest'] == 0 else metrics_values_dict['recovery_pct_eqty']:.2f}"],

            ['Sharpe Koeff', f"{metrics_values_dict['sharpe_month_days']:.2f}"],
            #   ['Sortino Koeff', f"{metrics_values_dict['sortino_month_days']:.2f}"],

            ['Прибыльно месяцев, %', f"{metrics_values_dict['months_plus_pct']:.2f}"],
            ['Прибыльно кварталов, %', f"{metrics_values_dict['quartals_plus_pct']:.2f}"],

            ['Beard Koeff', f"{metrics_values_dict['beard_coeff_daily']:.2f}"],
            ['Бород в год (в среднем)', f"{metrics_values_dict['daily_beards_per_year']:.2f}"],
            ['max борода, дней', f"{metrics_values_dict['daily_beard_max']:.2f}"],

            ['Graal Metr, %',
             f"""{metrics_values_dict['GraalMetr_NoReinvest']
             if metrics_values_dict['_pct_of_reinvest'] == 0
             else metrics_values_dict['GraalMetr_WithReinvest']:.2f}"""]
        ]

        # Настраиваем область для таблицы
        ax3.axis('tight')  # Устанавливаем плотное (tight) расположение осей, чтобы минимизировать отступы
        ax3.axis('off')  # Отключаем отображение осей, чтобы убрать рамки и метки вокруг таблицы

        # Создаем таблицу
        table = ax3.table(
            cellText=table_data,  # Данные для заполнения таблицы
            colLabels=None,  # Заголовки столбцов (None, так как они уже включены в table_data)
            cellLoc='center',  # Размещение текста в ячейках по центру
            loc='center'  # Размещение таблицы по центру области ax3
        )

        # Настраиваем размер шрифта и масштаб таблицы
        table.auto_set_font_size(False)  # Отключаем автоматический выбор размера шрифта
        table.set_fontsize(9)  # Устанавливаем размер шрифта для текста в таблице
        table.scale(1.6, 1.2)  # Масштабируем таблицу по ширине и высоте (коэффициент 1.2)

        # Отображение графика
        plt.tight_layout()
        plt.show()

def plot_strategy_results_LowVolume(must_plot, metrics_calc, metrics_values_dict, strategy_name, _init_deposit,
                              time_index):
    if must_plot:
        # Создаем полотно графика с использованием GridSpec
        fig = plt.figure(figsize=(10, 6))

        # Определяем сетку 2x2 с соотношением ширины и высоты столбцов/строк
        gs = gridspec.GridSpec(2, 2, width_ratios=[3, 1], height_ratios=[3.5, 1])

        # Создаем области для графиков
        ax1 = plt.subplot(gs[0, 0])  # Верхний левый график (первая строка, первый столбец)
        ax2 = plt.subplot(gs[1, 0])  # Нижний левый график (вторая строка, первый столбец)
        ax3 = plt.subplot(gs[:, 1])  # Правая область, занимает обе строки (весь второй столбец)

        # создаем датафрейм с индексом даты для корректной отрисовки
        temp_df = pd.DataFrame(
            {
                'Eq': metrics_calc.equity_punkt_arr,
                'EqMax': metrics_calc.max_equity_punkt_arr,
                'DDfromStart': metrics_calc.drawdown_curve_pct_from_start_capital_arr,
                'DDfromEq': metrics_calc.drawdown_curve_pct_from_max_eqty_arr
            }
        )
        temp_df.index = time_index

        # Рисуем первый график (Equity)

        ax1.plot(
            temp_df.Eq,
            label=f'Equity ($) = NetProfit ($) + Start deposit ($) ({_init_deposit})',
            lw=2,
            color='green'
        )

        ax1.plot(
            temp_df.EqMax,
            label=f'Max Equity ($)',
            lw=2,
            color='red'
        )

        # Подписи осей и заголовок для первого графика
        ax1.set_xlabel('Бары')
        ax1.set_ylabel(f'Equity$')

        text_title = (f'Strategy: "{strategy_name}." '
                      f'Период тестирования: c '
                      f'{metrics_calc.start_time_strategy.strftime("%d-%m-%Y")} '
                      f'по '
                      f'{metrics_calc.end_time_strategy.strftime(("%d-%m-%Y"))} '
                      f'таймфрейм: {metrics_calc.timeframe_string}.')

        ax1.set_title(text_title)
        ax1.grid(True)
        ax1.legend()

        # Рисуем 1ый график на 2ом полотне (Drawdown curve)
        ax2.plot(
            temp_df.DDfromStart if metrics_values_dict['_pct_of_reinvest'] == 0
            else temp_df.DDfromEq,
            label='',
            lw=2,
            color='red'
        )

        # Подписи осей и заголовок для второго графика
        ax2.set_xlabel('Trading period')
        ax2.set_ylabel('DrawDown (%)')
        ax2.set_title('Просадка (DrawDawn), %')
        ax2.grid(True)
        ax2.legend("")

        # Данные для таблицы
        table_data = [
            ['Метрика', 'Значение'],
            ['Symbol', f"{metrics_values_dict['COIN']}"],
            ['timeframe', f"{metrics_values_dict['timeframe']}"],
            # ['Дата тестирования:', f"с {metrics_values_dict['start_time_strategy'].strftime('%d-%m-%Y')} по {metrics_values_dict['end_time_strategy'].strftime('%d-%m-%Y')}"]
            #    ['Start торговли:', {metrics_calc.start_time_str}]
            #    ['End торговли:', {metrics_calc.end_time_str}]

            ['period_low', f"{metrics_values_dict['period_low']:.0f}"],
            ['period_volume', f"{metrics_values_dict['period_volume']:.2f}"],

            ['Трейдов в год', f"{metrics_values_dict['trades_per_year']:.0f}"],

            ['Net Profit, %', f"{metrics_values_dict['net_profit_end_pct']:.2f}"],
            ['APR %', f"{metrics_values_dict['apr_pct']:.2f}"],
            ['max Drawdown, %',
             f"{metrics_values_dict['dd_worst_start_capital'] if metrics_values_dict['_pct_of_reinvest'] == 0 else metrics_values_dict['dd_worst_max_eqty']:.2f}"],
            ['Calmar Coeff',
             f"{metrics_values_dict['calmar_coeff_start_capital'] if metrics_values_dict['_pct_of_reinvest'] == 0 else metrics_values_dict['calmar_coeff_max_eqty']:.2f}"],

            ['Recovery Factor',
             f"{metrics_values_dict['recovery_pct_capital'] if metrics_values_dict['_pct_of_reinvest'] == 0 else metrics_values_dict['recovery_pct_eqty']:.2f}"],

            ['Sharpe Koeff', f"{metrics_values_dict['sharpe_month_days']:.2f}"],
            #   ['Sortino Koeff', f"{metrics_values_dict['sortino_month_days']:.2f}"],

            ['Прибыльно месяцев, %', f"{metrics_values_dict['months_plus_pct']:.2f}"],
            ['Прибыльно кварталов, %', f"{metrics_values_dict['quartals_plus_pct']:.2f}"],

            ['Beard Koeff', f"{metrics_values_dict['beard_coeff_daily']:.2f}"],
            ['Бород в год (в среднем)', f"{metrics_values_dict['daily_beards_per_year']:.2f}"],
            ['max борода, дней', f"{metrics_values_dict['daily_beard_max']:.2f}"],

            ['Graal Metr, %',
             f"""{metrics_values_dict['GraalMetr_NoReinvest']
             if metrics_values_dict['_pct_of_reinvest'] == 0
             else metrics_values_dict['GraalMetr_WithReinvest']:.2f}"""]
        ]

        # Настраиваем область для таблицы
        ax3.axis('tight')  # Устанавливаем плотное (tight) расположение осей, чтобы минимизировать отступы
        ax3.axis('off')  # Отключаем отображение осей, чтобы убрать рамки и метки вокруг таблицы

        # Создаем таблицу
        table = ax3.table(
            cellText=table_data,  # Данные для заполнения таблицы
            colLabels=None,  # Заголовки столбцов (None, так как они уже включены в table_data)
            cellLoc='center',  # Размещение текста в ячейках по центру
            loc='center'  # Размещение таблицы по центру области ax3
        )

        # Настраиваем размер шрифта и масштаб таблицы
        table.auto_set_font_size(False)  # Отключаем автоматический выбор размера шрифта
        table.set_fontsize(9)  # Устанавливаем размер шрифта для текста в таблице
        table.scale(1.6, 1.2)  # Масштабируем таблицу по ширине и высоте (коэффициент 1.2)

        # Отображение графика
        plt.tight_layout()
        plt.show()
        