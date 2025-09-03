import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from tradingMahtematics import metrics_punkt
from tradingMahtematics import array_operations
from functools import reduce

class PerformanceMetrics:
    def __init__(
            self,
            start_capital: float,
            dates_arr_np: np.datetime64,
            net_profit_punkt_arr: np.array,
            net_profit_punkt_fixed_arr: np.array,
            trades_count: int
    ) -> object:
        # Инициализация основных переменных
        #self.all_positions = positions  # Все позиции (сделки)
        self.start_time_strategy = dates_arr_np[0] #Дата начала торговли
        self.end_time_strategy = dates_arr_np[-1] #Дата окончания торговли
        self.start_capital = start_capital  # Стартовый капитал (init_deposit)
        self.dates_arr_np = dates_arr_np  # Побарный список дат
        self.net_profit_punkt_arr = net_profit_punkt_arr  # Побарная кривая изменения капитала (зелёная)
        self.net_profit_punkt_fixed_arr = net_profit_punkt_fixed_arr  # Побарная кривая изменения капитала (фиолетовая)
        self.trades_count = trades_count  # Количество сделок


        #рассчитываем остальные показатели
        self.bars_count = self.net_profit_punkt_arr.size # Количество баров
        self.net_profit_punkt = self.net_profit_punkt_arr[-1] #NetProfit за весь период торговли
        self.ending_capital = self.start_capital + self.net_profit_punkt #Конечный капитал (на последнем баре)
        self.trading_days = metrics_punkt.trading_days(self.dates_arr_np[0], self.dates_arr_np[-1]) #Количество торговых дней
        self.net_profit_pct_arr = (self.net_profit_punkt_arr / self.start_capital * 100.0) / self.trading_days * 365
        self.trades_per_year = int (float(self.trades_count) / float(self.trading_days) * 365.0)

        self.apr_pct = metrics_punkt.annual_pct_rate_calc(
            self.start_capital,
            self.start_time_strategy,
            self.ending_capital,
            self.end_time_strategy

        )

        self.timeframe_minutes = self.timeframe_strategy_minutes() #Таймфрейм стратегии в минутах

    def timeframe_strategy_minutes(self):
        startDate = self.dates_arr_np[-2]
        endDate = self.dates_arr_np[-1]
        time_difference = endDate - startDate
        minutes = time_difference / np.timedelta64(1, 'm') # Преобразуем во временной интервал в минутах
        return int(minutes)
    def timeframe_string(self):
        time_frame_float = self.timeframe_minutes
        timeframe_str = f"{time_frame_float:2d}m"
        return timeframe_str


    def equity_punkt_arr(self):
        return self.net_profit_punkt_arr + self.start_capital

    def equity_puntk_fixed_arr(self):
        return self.net_profit_punkt_fixed_arr + self.start_capital

    def max_net_profit_punkt_arr (self):
        return np.maximum.accumulate(self.net_profit_punkt_arr) # Накопленный максимум чистой прибыли на каждом шагу

    def max_equity_punkt_arr (self):
        return (self.start_capital + self.max_net_profit_punkt_arr())

    def drawdown_punkt_arr (self):
        return self.net_profit_punkt_arr - self.max_net_profit_punkt_arr()

    def drawdown_worst_punkt(self):
        return np.min(self.drawdown_punkt_arr())  # Худшая просадка за всё время торговли

    def drawdown_pct_from_max_eqty_arr (self):
        return self.drawdown_punkt_arr() / self.max_equity_punkt_arr() * 100.0

    def drawdown_worst_from_max_eqty_pct(self):
        return np.min(self.drawdown_pct_from_max_eqty_arr())  # Худшая просадка за всё время торговли

    def drawdown_pct_from_start_capital_arr(self):
        return self.drawdown_punkt_arr() / self.start_capital * 100.0

    def drawdown_worst_from_start_capital_pct(self):
        return np.min(self.drawdown_pct_from_start_capital_arr())  # Худшая просадка за всё время торговли

    def calmar_coeff_max_eqty(self):
        return self.apr_pct / abs(self.drawdown_worst_from_max_eqty_pct())

    def calmar_coeff_start_capital(self):
        return self.apr_pct / abs(self.drawdown_worst_from_start_capital_pct())

    def net_profit_pct_fixed_arr (self):
        return array_operations.punkt_to_pct(
        self.start_capital,
        self.net_profit_punkt_fixed_arr
    )

    def net_profit_pct_arr(self):
        return array_operations.punkt_to_pct(
            self.start_capital,
            self.net_profit_punkt_arr
        )

    def drawdown_pct_curve_arr(self):
        return -1.00 * metrics_punkt.draw_dawn_curve_pct_calc(
            self.start_capital,
            self.net_profit_punkt_arr
        )

    def hourly_equity(self):
        return array_operations.fill_hourly_equity(
            self.net_profit_punkt_arr,
            self.dates_arr_np
        )

    def daily_equity(self):
        return array_operations.fill_daily_equity(
            self.net_profit_punkt_arr,
            self.dates_arr_np
        )
    def profit_per_bar(self):return (self.net_profit_punkt / self.bars_count) if self.bars_count > 0 else 0.00

    def daily_beards_per_year(self):
        # Проверка, что массив не пустой и содержит хотя бы два элемента
        if len(self.daily_equity()) < 2:
            return 0

        # Инициализация переменных
        max_capital = self.daily_equity()[0]
        new_highs_count = 0

        # Проход по массиву начиная со второго элемента (индекс 1)
        for capital in self.daily_equity()[1:]:
            if capital > max_capital:
                new_highs_count += 1
                max_capital = capital

        return int (new_highs_count / self.trading_days * 365)

    def daily_beard_max(self):
        """
        Определяет максимальное расстояние в барах между двумя моментами, когда значение капитала обновляло свой максимум.

        Параметры:
        daily_equity_arr (np.ndarray): Массив, отражающий подневное изменение капитала.

        Возвращает:
        int: Максимальное расстояние в барах между двумя моментами обновления максимума капитала.
        """
        daily_equity = self.daily_equity()
        current_max = -np.inf # Инициализация переменной для хранения текущего максимума капитала
        last_max_index = -1 # Инициализация переменной для хранения индекса последнего обновления максимума
        max_distance = 0 # Инициализация переменной для хранения максимального расстояния между обновлениями максимума

        for bar in range(len(daily_equity)): # Проход по массиву daily_equity_arr

            if daily_equity[bar] > current_max: # Проверка, обновляется ли максимум капитала на текущем баре

                if last_max_index != -1: # Если максимум уже обновлялся ранее (не первый раз), вычисляем расстояние

                    # Вычисляем расстояние между текущим и предыдущим моментами обновления максимума
                    max_distance = max(max_distance, bar - last_max_index)
                current_max = daily_equity[bar] # Обновляем текущий максимум капитала
                last_max_index = bar # Обновляем индекс последнего обновления максимума
        return max_distance # Возвращаем максимальное расстояние между моментами обновления максимума

    def beard_coeff_daily (self):
        if float (self.daily_beard_max()) == 0:
            return float (float (self.daily_beards_per_year()) / 1 * 100.00) #если ширина бороды была нулю
        else:
            return float (float (self.daily_beards_per_year()) / float (self.daily_beard_max()) * 100.00)

    def SharpeMonth (self):
        tf_string = self.timeframe_string()
        CashReturnRate = 0.00
        return metrics_punkt.calc_SharpeMonth(
            self.net_profit_punkt_arr + 10000,
            tf_string,
            CashReturnRate
        )

    def QuaterAvgProfit(self):
        tf_string = self.timeframe_string()
        return metrics_punkt.calc_QuaterAvgProfit(
            self.net_profit_punkt_arr,
            tf_string
        )['% of profit quater']
    # Дополнительные методы для расчета других показателей

    def MonthAvgProfit(self):
        tf_string = self.timeframe_string()
        return metrics_punkt.calc_MonthAvgProfit(
            self.net_profit_punkt_arr,
            tf_string
        )['% of profit months']


''' 
Пример использования класса
if __name__ == "__main__":
    # Допустим, у нас есть некоторые данные для инициализации
    positions = []  # Список позиций
    capital_curve_all = [1000, 1050, 1030, 1080]  # Пример кривой капитала
    cash_all = [200, 150, 170, 120]  # Пример кривой наличности
    bars_count = len(capital_curve_all)
    trades_count = len(positions)
    starting_capital = capital_curve_all[0]
    ending_capital = capital_curve_all[-1]

    # Создаем экземпляр класса
    metrics_calc = PerformanceMetrics(positions, capital_curve_all, cash_all, bars_count, trades_count, starting_capital, ending_capital)

    # Вызываем методы для расчета
    net_profit = metrics_calc.calculate_net_profit()
    profit_per_bar = metrics_calc.calculate_profit_per_bar()

    print(f"Net Profit: {net_profit}")
    print(f"Profit per Bar: {profit_per_bar}")
'''