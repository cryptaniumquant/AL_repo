#region Импортируем библиотеки
import numpy as np
import pandas as pd
from datetime import timedelta
from datetime import datetime
import math


#endregion

class PerformanceMetrics_new:
    '''
    Класс, в котором происходит рассчёт основных метрик торговых стратегий.

    '''

    #Конструктор класса. Инициализирует новый объект класса и устанавливает начальные значения его атрибутов.
    def __init__(
            self,
            start_capital: float,  # Начальный капитал
            Date_np: np.ndarray,  # Массив дат в формате np.array (ячейки содержат numpy.datetime64)
            Date_pd: pd.Series,  # Массив дат в формате pd.Series (ячейки содержат Timestamp)
            Date_dt: np.ndarray,  # Массив дат в формате np.ndarray (ячейки содержат datetime.datetime)
            net_profit_punkt_arr: np.ndarray,  # Массив net_profit в пунктах
            net_profit_punkt_fixed_arr: np.ndarray,  # Массив фиксированной net_profit в пунктах
            trades_count: int  # Количество сделок
    ) -> object:
        """
        Инициализация основных переменных
        """
        #region константы:
        self.DAYS_IN_YEAR = 365
        '''
        Константа:
        Количество дней в году
        '''
        self.NANOSECONDS_IN_SECOND = 1e9
        '''
        Константа:
        Для преобразования наносекунд в секунды
        число 1e9 обозначает (10^9) - это 1 миллиард
        1 наносекунда = миллиард секунд
        '''

        self.SECONDS_IN_MINUTE = 60
        '''
        Константа: 
        Количество секунд в минуте
        '''

        self.SECONDS_IN_DAY = 60 * 60 * 24
        '''
        Константа: 
        Количество секунд в дне
        '''

        self.CashReturnRate = 0.00
        '''
        Константа:
        Безрисковая ставка доходности. По умолчанию = 0
        Применяется для расчёта коэфф. Шарпа
        и прочих метрик
        '''
        #endregion

        #region Ряды данных
        self.Date_np = Date_np
        """
        Ряд данных:             Date_np
        Тип ряда данных:        ndarray: (82910,)
        Тип данных в ячейках:   numpy.datetime64:   Пример: numpy.datetime64('2019-09-08T20:29:59.999000000')
        """

        self.Date_pd = Date_pd
        """
        Ряд данных:             Date_pd
        Тип ряда данных:        Series: (82910,)
        Тип данных в ячейках:   Timestamp библиотеки pandas:    Пример: Timestamp('2019-09-08 20:29:59.999000')
        """

        self.Date_dt = pd.to_datetime(Date_dt)
        """
        Ряд данных:             Date_dt
        Тип ряда данных:        ndarray: (82910,)
        Тип данных в ячейках:   datetime.datetime:    Пример: datetime.datetime(2019, 9, 8, 20, 29, 59, 999000)
        """

        self.net_profit_punkt_arr = net_profit_punkt_arr
        '''
        net_profit (unfixed) - чистый результат (за вычетом комиссии) побарно в пунктах
        в т.ч. незафиксированный результат по открытым позициям
        '''

        self.net_profit_punkt_fixed_arr = net_profit_punkt_fixed_arr
        '''
        net_profit (fixed) - чистый результат (за вычетом комиссии) побарно в пунктах
        рассчитывается только по закрытым позициям
        '''
        #endregion

        self.bars_count = len(self.net_profit_punkt_arr)
        '''
        Количество баров в стратегии
        '''

        self.trades_count = trades_count
        '''
        Количество сделок в стратегии
        '''

        self.equity_start_punkt = start_capital
        """
        Стартовый капитал, пунктов (число)
        """

        self.equity_end_punkt = self.equity_start_punkt + self.net_profit_end_punkt
        """
        Конечный каптиал, пунктов (число)
        """

        self.timeframe_string = f"{self.timeframe_minutes}m"
        """
        Возвращает строковое представление таймфрейма в минутах. Например "60m".
        """

        self.start_time_strategy: datetime = self.Date_dt[0]

        self.end_time_strategy: datetime = self.Date_dt[-1]

        self.start_time_str: str = self.start_time_strategy.strftime("%d-%m-%Y")

        self.end_time_str: str = self.end_time_strategy.strftime("%d-%m-%Y")


    @property
    def trading_days(self) -> int:
        """
        Количество дней между последним и первым баром (int)
        """
        end_time_td = self.end_time_strategy
        start_time_td = self.start_time_strategy
        delta = end_time_td - start_time_td
        result = delta.days
        return result

    @property
    def timeframe_minutes(self) -> int:
        """
        Рассчитывает таймфрейм стратегии в минутах.
        """
        time_difference = self.Date_dt[-1] - self.Date_dt[-2]
        seconds = time_difference.total_seconds()
        minutes = seconds / self.SECONDS_IN_MINUTE
        return int(minutes)


    #region рассчитываем net_profit

    @property
    def net_profit_pct_arr(self) -> np.ndarray:
        """
        Возвращает массив процентной прибыли.
        """
        return (self.net_profit_punkt_arr / self.equity_start_punkt) * 100

    @property
    def net_profit_pct_fixed_arr(self) -> np.ndarray:
        """
        Возвращает массив процентной прибыли (фиксированной).
        """
        return (self.net_profit_punkt_fixed_arr / self.equity_start_punkt) * 100

    def net_profit_pct_per_year_arr (self) -> np.ndarray:
        """
        # Возвращает массив чистой прибыли в процентах, нормализованный на год
        """

        return self.net_profit_pct_arr / self.trading_days * self.DAYS_IN_YEAR


    @property
    def max_net_profit_punkt_arr(self) -> np.ndarray:
        """
        Возвращает накопленный максимум чистой прибыли на каждом баре (шаге).
        """
        return np.maximum.accumulate(self.net_profit_punkt_arr)

    @property
    def net_profit_end_punkt(self) -> float:
        """
        Возвращает net_profit в пунктах на последний торговый бар (число)
        """
        return self.net_profit_punkt_arr[-1].item()

    @property
    def net_profit_end_pct(self) -> float:
        """
        Возвращает net_profit в пунктах на последний торговый бар (число)
        """
        return self.net_profit_end_punkt / self.equity_start_punkt * 100

    #endregion

    #region рассчитываем equity
    @property
    def equity_punkt_arr(self) -> np.ndarray:
        """
        Возвращает побарную кривую изменения капитала.
        """
        return self.net_profit_punkt_arr + self.equity_start_punkt

    @property
    def equity_punkt_fixed_arr(self) -> np.ndarray:
        """
        Возвращает побарную кривую изменения капитала (фиксированная).
        """
        return self.net_profit_punkt_fixed_arr + self.equity_start_punkt

    @property
    def max_equity_punkt_arr(self) -> np.ndarray:
        """
        Возвращает накопленный максимум капитала на каждом баре (шаге).
        """
        return self.equity_start_punkt + self.max_net_profit_punkt_arr

    #endregion

    #region с помощью pandas определям net_profit в пунктах по месяцу и кварталу и среднюю месячную доходность

    @property
    def hourly_net_profit_punkt(self) -> pd.Series:
        """
            Почасовой список net_profit_punkt
            Возвращаемый тип данных: `pandas.Series`
            Индекс: Временные метки (Date_dt)
            Тип данных в каждой ячейке: числовой тип данных (`float`)
        """

        period_label = 'h'
        """
        Переменная `period_label` используется в методе для указания частоты ресэмплинга данных в DataFrame. 
        В данном случае, значение `'ME'` обозначает "конец месяца" ("Month End"). 
        Однако, `period_label` может принимать и другие значения, в зависимости от желаемой частоты ресэмплинга. 
        Вот некоторые другие возможные значения, которые могут быть полезны:

                1. **'A' или 'Y'**: Годовая частота ("Annual" или "Yearly").
                2. **'Q'**: Квартальная частота ("Quarterly").
                3. **'M'**: Ежемесячная частота ("Monthly").
                4. **'W'**: Еженедельная частота ("Weekly").
                5. **'D'**: Ежедневная частота ("Daily").
                6. **'H'**: Почасовая частота ("Hourly").
                7. **'T' или 'min'**: Ежеминутная частота ("Minutely").
                8. **'S'**: Ежесекундная частота ("Secondly").
                9. **'B'**: Ежедневная частота по рабочим дням ("Business day").

        """
        net_profit_df = pd.DataFrame({'NetProfit': self.net_profit_punkt_arr})
        net_profit_df.index = self.Date_dt #назначили индексом
        #теперь индекс для net_profit - не номера строк, а временные метки

        #Из NetProfit на конец месяца (.last) вычитаем NetProfit на начало месяца (.first) и получаем
        hour_net_profit = net_profit_df['NetProfit'].resample(period_label).last() - net_profit_df['NetProfit'].resample(period_label).first()
        return hour_net_profit



    @property
    def monthly_net_profit_punkt(self) -> pd.Series:
        """
            Помесячный список net_profit_punkt
            Возвращаемый тип данных: `pandas.Series`
            Индекс: Временные метки (Date_dt)
            Тип данных в каждой ячейке: числовой тип данных (`float`)
        """

        period_label = 'M'
        """
        Переменная `period_label` используется в методе для указания частоты ресэмплинга данных в DataFrame. 
        В данном случае, значение `'ME'` обозначает "конец месяца" ("Month End"). 
        Однако, `period_label` может принимать и другие значения, в зависимости от желаемой частоты ресэмплинга. 
        Вот некоторые другие возможные значения, которые могут быть полезны:

                1. **'A' или 'Y'**: Годовая частота ("Annual" или "Yearly").
                2. **'Q'**: Квартальная частота ("Quarterly").
                3. **'M'**: Ежемесячная частота ("Monthly").
                4. **'W'**: Еженедельная частота ("Weekly").
                5. **'D'**: Ежедневная частота ("Daily").
                6. **'H'**: Почасовая частота ("Hourly").
                7. **'T' или 'min'**: Ежеминутная частота ("Minutely").
                8. **'S'**: Ежесекундная частота ("Secondly").
                9. **'B'**: Ежедневная частота по рабочим дням ("Business day").

        """
        period_label = 'M'

        equity_punkt_df = pd.DataFrame({'EquityPunkt_df': self.equity_punkt_arr})
        equity_punkt_df.index = self.Date_dt  # назначили индексом
        # теперь индекс для net_profit - не номера строк, а временные метки

        # Получаем последние значения для каждого месяца
        current_month_last_value = equity_punkt_df['EquityPunkt_df'].resample(period_label).last()

        # Сдвигаем серию на один месяц вперед, чтобы получить последние значения предыдущего месяца
        previous_month_last_value = current_month_last_value.shift(1)

        # Вычисляем изменение между последним значением текущего месяца и последним значением предыдущего месяца
        month_net_profit_punkt = (current_month_last_value - previous_month_last_value)

        return month_net_profit_punkt

    @property
    def daily_net_profit_pct(self) -> pd.Series:
        """
        Подневной список net_profit_pct
        Возвращаемый тип данных: pandas.Series
        Индекс: Временные метки (Date_dt)
        Тип данных в каждой ячейке: числовой тип данных (float)
        """
        period_label = 'D'  # Период для дневного ресемплинга

        equity_punkt_df = pd.DataFrame({'EquityPunkt_df': self.equity_punkt_arr})
        equity_punkt_df.index = self.Date_dt  # назначили индексом

        # Получаем последние значения для каждого дня
        current_day_last_value = equity_punkt_df['EquityPunkt_df'].resample(period_label).last()

        # Сдвигаем серию на один день вперед, чтобы получить последние значения предыдущего дня
        previous_day_last_value = current_day_last_value.shift(1)

        # Вычисляем процентное изменение между последним значением текущего дня и последним значением предыдущего дня
        day_net_profit_pct = (current_day_last_value - previous_day_last_value) / previous_day_last_value * 100.0

        return day_net_profit_pct

    @property
    def monthly_net_profit_pct(self) -> pd.Series:
        """
        Помесячный список net_profit_pct
        Возвращаемый тип данных: pandas.Series
        Индекс: Временные метки (Date_dt)
        Тип данных в каждой ячейке: числовой тип данных (float)
        """
        period_label = 'M'

        equity_punkt_df = pd.DataFrame({'EquityPunkt_df': self.equity_punkt_arr})
        equity_punkt_df.index = self.Date_dt  # назначили индексом
        # теперь индекс для net_profit - не номера строк, а временные метки

        # Получаем последние значения для каждого месяца
        current_month_last_value = equity_punkt_df['EquityPunkt_df'].resample(period_label).last()

        # Сдвигаем серию на один месяц вперед, чтобы получить последние значения предыдущего месяца
        previous_month_last_value = current_month_last_value.shift(1)

        # Вычисляем процентное изменение между последним значением текущего месяца и последним значением предыдущего месяца
        month_net_profit_pct = (current_month_last_value - previous_month_last_value) / previous_month_last_value * 100.0

        return month_net_profit_pct


    @property
    def quartal_net_profit_punkt(self) -> pd.Series:
        """
            Поквартальный список net_profit_punkt
            Возвращаемый тип данных: `pandas.Series`
            Индекс: Временные метки (Date_dt)
            Тип данных в каждой ячейке: числовой тип данных (`float`)
        """
        period_label = 'Q'
        """
        Переменная `period_label` используется в методе для указания частоты ресэмплинга данных в DataFrame. 
        В данном случае, значение `'ME'` обозначает "конец месяца" ("Month End"). 
        Однако, `period_label` может принимать и другие значения, в зависимости от желаемой частоты ресэмплинга. 
        Вот некоторые другие возможные значения, которые могут быть полезны:

                1. **'A' или 'Y'**: Годовая частота ("Annual" или "Yearly").
                2. **'Q'**: Квартальная частота ("Quarterly").
                3. **'M'**: Ежемесячная частота ("Monthly").
                4. **'W'**: Еженедельная частота ("Weekly").
                5. **'D'**: Ежедневная частота ("Daily").
                6. **'H'**: Почасовая частота ("Hourly").
                7. **'T' или 'min'**: Ежеминутная частота ("Minutely").
                8. **'S'**: Ежесекундная частота ("Secondly").
                9. **'B'**: Ежедневная частота по рабочим дням ("Business day").

        """
        net_profit_df = pd.DataFrame({'NetProfit': self.net_profit_punkt_arr})
        net_profit_df.index = self.Date_dt #назначили индексом
        #теперь индекс для net_profit - не номера строк, а временные метки

        #Из NetProfit на конец месяца (.last) вычитаем NetProfit на начало месяца (.first) и получаем
        quart_net_profit = net_profit_df['NetProfit'].resample(period_label).last() - net_profit_df['NetProfit'].resample(period_label).first()
        return quart_net_profit

    @property
    def daily_net_profit_punkt(self) -> pd.Series:
        """
            Подневный список net_profit_punkt
            Возвращаемый тип данных: `pandas.Series`
            Индекс: Временные метки (Date_dt)
            Тип данных в каждой ячейке: числовой тип данных (`float`)
        """

        period_label = 'D'
        """
        Переменная `period_label` используется в методе для указания частоты ресэмплинга данных в DataFrame. 
        В данном случае, значение `'ME'` обозначает "конец месяца" ("Month End"). 
        Однако, `period_label` может принимать и другие значения, в зависимости от желаемой частоты ресэмплинга. 
        Вот некоторые другие возможные значения, которые могут быть полезны:

                1. **'A' или 'Y'**: Годовая частота ("Annual" или "Yearly").
                2. **'Q'**: Квартальная частота ("Quarterly").
                3. **'M'**: Ежемесячная частота ("Monthly").
                4. **'W'**: Еженедельная частота ("Weekly").
                5. **'D'**: Ежедневная частота ("Daily").
                6. **'H'**: Почасовая частота ("Hourly").
                7. **'T' или 'min'**: Ежеминутная частота ("Minutely").
                8. **'S'**: Ежесекундная частота ("Secondly").
                9. **'B'**: Ежедневная частота по рабочим дням ("Business day").

        """
        net_profit_df = pd.DataFrame({'NetProfit': self.net_profit_punkt_arr})
        net_profit_df.index = self.Date_dt  # назначили индексом
        # теперь индекс для net_profit - не номера строк, а временные метки

        # Из NetProfit на конец месяца (.last) вычитаем NetProfit на начало месяца (.first) и получаем
        daily_np = net_profit_df['NetProfit'].resample(period_label).last() - net_profit_df['NetProfit'].resample(
            period_label).first()
        return daily_np

    @property
    def months_plus_pct(self):

        # Создаем новый ряд данных с условием: 1 если значение положительное, 0 если значение отрицательное или нулевое
        binary_series = self.monthly_net_profit_punkt.apply(lambda x: 1 if x > 0 else 0)
        total_months = len(binary_series) # Количество всех месяцев
        positive_months = (binary_series > 0).sum() # Количество месяцев с положительным результатом
        positive_percentage = (positive_months / total_months) * 100 # Процент месяцев с положительным результатом
        return positive_percentage

    @property
    def quartals_plus_pct(self):
        # Создаем новый ряд данных с условием: 1 если значение положительное, 0 если значение отрицательное или нулевое
        binary_series = self.quartal_net_profit_punkt.apply(lambda x: 1 if x > 0 else 0)
        total_quartals = len(binary_series) # Количество всех месяцев
        positive_quartals = (binary_series > 0).sum() # Количество месяцев с положительным результатом
        positive_percentage = (positive_quartals / total_quartals) * 100 # Процент месяцев с положительным результатом
        return positive_percentage

    @property
    def days_plus_pct(self):
        # Создаем новый ряд данных с условием: 1 если значение положительное, 0 если значение отрицательное или нулевое
        binary_series = self.daily_net_profit_punkt.apply(lambda x: 1 if x > 0 else 0)
        total_days = len(binary_series) # Количество всех месяцев
        positive_days = (binary_series > 0).sum() # Количество месяцев с положительным результатом
        positive_percentage = (positive_days / total_days) * 100 # Процент месяцев с положительным результатом
        return positive_percentage

    #endregion

    #region рассчитываем нормализованные эквити (часовые, дневные)
    @property
    def hourly_equity(self) -> np.ndarray:
        """
        Возвращает нормализованную часовую equity (массив нормализованного почасового капитала).
        """

        SECONDS_IN_HOUR = 60 * 60 #Количество секунд в часе
        # Конвертируем временные метки из numpy.datetime64 в datetime
        cur_date = [datetime.utcfromtimestamp(ts.tolist() / self.NANOSECONDS_IN_SECOND) for ts in self.Date_np]

        # Преобразуем список обратно в numpy массив
        cur_date = np.array(cur_date)

        # Определяем начальное и конечное время
        start_time = cur_date[0].astype('datetime64[ns]').item() #TODO: проверить .astype('datetime64[ns]').item()
        end_time = cur_date[-1].astype('datetime64[ns]').item() #TODO: проверить .astype('datetime64[ns]').item()

        # Создаем массив всех часов в диапазоне от start_time до end_time
        total_hours = int((end_time - start_time).total_seconds() // SECONDS_IN_HOUR) + 1
        hourly_dates = [start_time + timedelta(hours=i) for i in range(total_hours)]

        # Создаем словарь для быстрого поиска значений по временным меткам
        equity_dict = {date: value for date, value in zip(cur_date, self.net_profit_punkt_arr)}

        # Заполняем результат, используя значения из equity_dict
        hourly_equity = []
        last_value = None

        for hour in hourly_dates:
            if hour in equity_dict:
                last_value = equity_dict[hour]
            hourly_equity.append(last_value)

        return np.array(hourly_equity)

    @property
    def daily_equity(self) -> pd.Series:
        """
        Возвращает нормализованную дневную equity (массив нормализованного дневного капитала).
        """
        period_label = 'D'
        """
        Переменная `period_label` используется в методе для указания частоты ресэмплинга данных в DataFrame. 
        В данном случае, значение `'ME'` обозначает "конец месяца" ("Month End"). 
        Однако, `period_label` может принимать и другие значения, в зависимости от желаемой частоты ресэмплинга. 
        Вот некоторые другие возможные значения, которые могут быть полезны:

                1. **'A' или 'Y'**: Годовая частота ("Annual" или "Yearly").
                2. **'Q'**: Квартальная частота ("Quarterly").
                3. **'M'**: Ежемесячная частота ("Monthly").
                4. **'W'**: Еженедельная частота ("Weekly").
                5. **'D'**: Ежедневная частота ("Daily").
                6. **'H'**: Почасовая частота ("Hourly").
                7. **'T' или 'min'**: Ежеминутная частота ("Minutely").
                8. **'S'**: Ежесекундная частота ("Secondly").
                9. **'B'**: Ежедневная частота по рабочим дням ("Business day").

        """
        equity_df = pd.DataFrame({'Equity': self.equity_punkt_arr})
        equity_df.index = self.Date_dt  # назначили индексом
        # теперь индекс для net_profit - не номера строк, а временные метки

        # берём Equity на конец дня (.last) и получаем
        daily_equity_series = equity_df['Equity'].resample(period_label).last()

        return daily_equity_series

    #endregion

    #region рассчитываем просадки (drawdown)
    @property
    def drawdown_curve_punkt_arr(self) -> np.ndarray:
        """
        Возвращает массив просадок в пунктах (отрицательные числа).
        """
        return self.net_profit_punkt_arr - self.max_net_profit_punkt_arr

    @property
    def drawdown_curve_pct_from_start_capital_arr(self) -> np.ndarray:
        """
        Возвращает массив просадок в процентах от стартового капитала (отрицательные числа).
        """
        return self.drawdown_curve_punkt_arr / self.equity_start_punkt * 100.0


    @property
    def drawdown_curve_pct_from_max_eqty_arr(self) -> np.ndarray:
        """
        Возвращает массив просадок в процентах от максимального капитала (отрицательные числа).
        """
        return self.drawdown_curve_punkt_arr / self.max_equity_punkt_arr * 100.0

    @property
    def drawdown_worst_punkt(self) -> float:
        """
        Возвращает худшую просадку в пунктах за всё время торговли (отрицательное число).
        """
        return np.min(self.drawdown_curve_punkt_arr)


    @property
    def drawdown_worst_from_max_eqty_pct(self) -> float:
        """
        Возвращает худшую просадку в процентах от максимального капитала за всё время торговли (отрицательное число).
        """
        return np.min(self.drawdown_curve_pct_from_max_eqty_arr)


    @property
    def drawdown_worst_from_start_capital_pct(self) -> float:
        """
        Возвращает худшую просадку в процентах от стартового капитала за всё время торговли (отрицательное число).
        """
        return np.min(self.drawdown_curve_pct_from_start_capital_arr)

    #endregion

    #region Основные метрики торговой системы

    @property
    def recovery_factor_punkt(self) -> float:
        """
        Отношение NetProfit к максимальной просадке
        """
        return self.net_profit_end_punkt / abs(self.drawdown_worst_punkt)

    @property
    def recovery_factor_pct_start_capital(self) -> float:
        """
        Отношение NetProfit к максимальной просадке
        """
        return self.net_profit_end_pct / abs(self.drawdown_worst_from_start_capital_pct)

    @property
    def recovery_factor_pct_max_equity(self) -> float:
        """
        Отношение NetProfit к максимальной просадке
        """
        return self.net_profit_end_pct / abs(self.drawdown_worst_from_max_eqty_pct)


    @property
    def apr_pct(self) -> float:
        """
        Процентная ставка роста за год: APR (Annual Percentage Rate) или CAGR (Compound Annual Growth Rate)
        """

        # Проверка на нулевые или отрицательные значения начальной суммы эквити
        if self.equity_start_punkt <= 0:
            raise ValueError("Начальная сумма эквити должна быть положительным числом.")

        # Определяем количество дней во временном промежутке
        days = self.trading_days

        # Переводим количество дней в количество лет
        number_of_periods = days / self.DAYS_IN_YEAR

        # Проверка на нулевую или отрицательную продолжительность периода
        if number_of_periods <= 0:
            raise ValueError("Дата окончания должна быть позже даты начала.")

        # Вычисляем среднегодовую ставку доходности (APR)
        try:
            growth_factor = self.equity_end_punkt / self.equity_start_punkt
            if growth_factor > 0:
                apr = (growth_factor ** (1.0 / number_of_periods)) - 1.0
            else:
                apr = 0
        except ZeroDivisionError:
            raise ValueError("Деление на ноль произошло при вычислении средней годовой ставки доходности.")

        # Переводим результат в проценты
        result = apr * 100.0

        return result

    @property
    def calmar_coeff_start_capital(self) -> float:
        """
        Возвращает коэффициент Калмара, рассчитанный от стартового капитала.
        """
        return self.apr_pct / abs(self.drawdown_worst_from_start_capital_pct)

    @property
    def calmar_coeff_max_eqty(self) -> float:
        """
        Возвращает коэффициент Калмара, рассчитанный от максимального капитала.
        """
        return self.apr_pct / abs(self.drawdown_worst_from_max_eqty_pct)

    @property
    def trades_per_year(self) -> float:
        return self.trades_count / self.trading_days * self.DAYS_IN_YEAR
    '''
    Количество сделок, генерируемое стратегией в пересчёте на один год
    '''

    @property
    def profit_per_bar(self) -> float:
        """
        Возвращает прибыль на бар.
        """
        return (self.net_profit_end_punkt / self.bars_count) if self.bars_count > 0 else 0.00

    @property
    def daily_beards_per_year(self) -> int:
        """
        Количество дневных "бород" за год
        Количество обновлений максимума по дневной эквити в среднем за год
        Чем больше раз дневная эквити обновляет свой максимум, тем лучше торговая система
        """
        daily_equity_np = self.daily_equity.to_numpy()

        if len(daily_equity_np) < 2:
            return 0

        max_capital = daily_equity_np[0]
        new_highs_count = 0

        for capital in daily_equity_np[1:]:
            if capital > max_capital:
                new_highs_count += 1
                max_capital = capital

        result = new_highs_count / self.trading_days * self.DAYS_IN_YEAR
        return int(result)

    @property
    def daily_beard_max(self) -> int:
        """
        Максимальная "борода", выраженная в днях
        Максимальное количество дней между двумя моментами обновления максимума дневной эквити.
        Чем меньше ширина "бороды", тем лучше для торговой системы, т.к. мы меньше дней сидим в просадке
        """

        daily_equity_np = self.daily_equity.to_numpy()

        current_max = -np.inf
        last_max_index = -1
        max_distance = 0

        for bar in range(len(daily_equity_np)):
            if daily_equity_np[bar] > current_max:
                if last_max_index != -1:
                    max_distance = max(max_distance, bar - last_max_index)
                current_max = daily_equity_np[bar]
                last_max_index = bar

        return max_distance

    @property
    def beard_coeff_daily(self) -> float:
        """
        Коэффициент бороды - это отношение:
            среднего количества бород за год (в числителе) к
            максимальной ширине бороды
        Т.е. среднее в год количество обновлений максимума по дневной эквити к максимальному количеству дней в просадке
        """
        if self.daily_beard_max == 0:
            return self.daily_beards_per_year / 1 * 100.00 #если не было ни дня просадки
        else:
            return self.daily_beards_per_year / self.daily_beard_max * 100.00

    # endregion

    @property
    def sharpe_month_days(self) -> float:
        """
        коэффициент Шарпа за месяц.Посчитанный по дневным NetProfit
        Количество торговых дней в году = 365
        Безрисковая ставка доходности = 0
        """


        daily_net_profit_pct = self.daily_net_profit_pct
        DAYS_IN_YEAR = 365
        CashReturnRate = 0

        # Вычисление среднего значения NetProfit по дневной эквити
        avg_net_profit_pct_day = daily_net_profit_pct.mean()

        # Вычисление стандартного отклонения NetProfit по дневной эквити
        std_dev_day = daily_net_profit_pct.std()

        # Вычисление месячного коэфф Шарпа по дневным net_profit
        if std_dev_day != 0:
            result = (avg_net_profit_pct_day * math.sqrt(DAYS_IN_YEAR) - CashReturnRate) / std_dev_day
        else:
            result = None  # или можно задать другое значение или обработку, если DevDay равно 0

        return result

    @property
    def sortino_month_days(self) -> float:
        """
        Коэффициент Сортино за месяц, посчитанный по дневным NetProfit.
        Количество торговых дней в году = 365
        Безрисковая ставка доходности = 0
        """
        daily_net_profit_pct = self.daily_net_profit_pct
        DAYS_IN_YEAR = 365
        CashReturnRate = 0

        # Вычисление среднего значения NetProfit по дневной эквити
        avg_net_profit_pct_day = daily_net_profit_pct.mean()

        # Вычисление стандартного отклонения только отрицательных доходностей
        negative_returns = daily_net_profit_pct[daily_net_profit_pct < 0]
        downside_deviation = np.std(negative_returns)

        # Вычисление месячного коэфф Сортино по дневным net_profit
        if downside_deviation != 0:
            result = (avg_net_profit_pct_day * math.sqrt(DAYS_IN_YEAR) - CashReturnRate) / downside_deviation
        else:
            result = None  # или можно задать другое значение или обработку, если downside_deviation равно 0

        return result

    @property
    def sharpe_month_months(self) -> float:
        """
        коэффициент Шарпа за месяц.Посчитанный по месячным NetProfit
        Количество торговых дней в году = 365
        Безрисковая ставка доходности = 0
        """


        monthly_net_profit_pct = self.monthly_net_profit_pct
        MONTHS_IN_YEAR = 12
        CashReturnRate = 0

        # Вычисление среднего значения NetProfit по дневной эквити
        avg_net_profit_pct_day = monthly_net_profit_pct.mean()

        # Вычисление стандартного отклонения NetProfit по дневной эквити
        std_dev_day = monthly_net_profit_pct.std()

        # Вычисление месячного коэфф Шарпа по дневным net_profit
        if std_dev_day != 0:
            result = (avg_net_profit_pct_day * math.sqrt(MONTHS_IN_YEAR) - CashReturnRate) / std_dev_day
        else:
            result = None  # или можно задать другое значение или обработку, если DevDay равно 0

        return result

    @property
    def sharpe_month_hours(self) -> float:
        """
        коэффициент Шарпа за месяц.Посчитанный по часовым NetProfit
        Количество торговых дней в году = 365
        Безрисковая ставка доходности = 0
        """


        hourly_net_profit_punkt = self.hourly_net_profit_punkt
        HOURS_IN_YEAR = 365 * 24
        CashReturnRate = 0

        # Вычисление среднего значения NetProfit по дневной эквити
        avg_net_profit_day = hourly_net_profit_punkt.mean()

        # Вычисление стандартного отклонения NetProfit по дневной эквити
        std_dev_day = hourly_net_profit_punkt.std()

        # Вычисление месячного коэфф Шарпа по дневным net_profit
        if std_dev_day != 0:
            result = (avg_net_profit_day * math.sqrt(HOURS_IN_YEAR) - CashReturnRate) / std_dev_day
        else:
            result = None  # или можно задать другое значение или обработку, если DevDay равно 0

        return result

    @property
    def QuaterAvgProfit(self) -> float:
        """
        Возвращает среднюю прибыль за квартал.
        """
        tf_string = self.timeframe_string
        arr_pnl = self.net_profit_punkt_arr

        # Определяем значение t в зависимости от временного шага time_step
        if len(tf_string) == 3:  # случай '10m' или '10h'
            if tf_string[2] == 'm':
                t = (30 * 24 * 60) * 4 / int(tf_string[:2])
            elif tf_string[2] == 'h':
                t = (30 * 24) * 4 / int(tf_string[:2])
            else:
                raise ValueError("Invalid time_step value. Please use 'xm' or 'xh', where x - integer number.")
        elif len(tf_string) == 2:  # случай '1m' или '1h'
            if tf_string[1] == 'm':
                t = (30 * 24 * 60) * 4 / int(tf_string[0])
            elif tf_string[1] == 'h':
                t = (30 * 24) * 4 / int(tf_string[0])
            else:
                raise ValueError("Invalid time_step value. Please use 'xm' or 'xh', where x - integer number.")
        else:
            raise ValueError("Invalid time_step value. Please use 'xm' or 'xh', where x - integer number.")

        # Вычисляем количество полных кварталов в массиве
        num_full_periods = arr_pnl.size // t
        # Отбрасываем конечный остаток, чтобы иметь целое число кварталов
        trimmed_arr = arr_pnl[:int(num_full_periods * t)]

        # Инициализируем список для хранения прибыли за каждый квартал
        quater_profit = []

        # Определяем индексы начала каждого квартала
        quater_starts = np.arange(0, trimmed_arr.size, t)
        for i in quater_starts:
            # Вычисляем прибыль за квартал
            quater_profit.append(trimmed_arr[int(i+1)] - trimmed_arr[int(i)])

        # Преобразуем список в numpy массив для удобства расчетов
        quater_profit = np.array(quater_profit)
        # Считаем количество кварталов с положительной прибылью
        count = np.sum(quater_profit > 0)

        # Возвращаем среднюю прибыль за квартал в процентах
        return (count / num_full_periods) * 100

    @property
    def MonthAvgProfit(self) -> float:
        """
        Возвращает среднюю прибыль за месяц.
        """
        tf_string = self.timeframe_string
        arr_pnl = self.net_profit_punkt_arr

        # Определяем значение t в зависимости от временного шага time_step
        if len(tf_string) == 3:  # случай '10m' или '10h'
            if tf_string[2] == 'm':
                t = (30 * 24 * 60) / int(tf_string[:2])  # количество временных шагов в одном месяце
            elif tf_string[2] == 'h':
                t = (30 * 24) / int(tf_string[:2])  # количество временных шагов в одном месяце
            else:
                raise ValueError("Invalid time_step value. Please use 'xm' or 'xh', where x - integer number.")
        elif len(tf_string) == 2:  # случай '1m' или '1h'
            if tf_string[1] == 'm':
                t = (30 * 24 * 60) / int(tf_string[0])  # количество временных шагов в одном месяце
            elif tf_string[1] == 'h':
                t = (30 * 24) / int(tf_string[0])  # количество временных шагов в одном месяце
            else:
                raise ValueError("Invalid time_step value. Please use 'xm' or 'xh', where x - integer number.")
        else:
            raise ValueError("Invalid time_step value. Please use 'xm' or 'xh', where x - integer number.")

        # Вычисляем количество полных месяцев в массиве
        num_full_periods = arr_pnl.size // t
        # Отбрасываем конечный остаток, чтобы иметь целое число месяцев
        trimmed_arr = arr_pnl[:int(num_full_periods * t)]

        # Инициализируем список для хранения прибыли за каждый месяц
        month_profit = []

        # Определяем индексы начала каждого месяца
        month_starts = np.arange(0, trimmed_arr.size, t)
        for i in month_starts:
            # Вычисляем прибыль за каждый месяц
            month_profit.append(trimmed_arr[int(i+1)] - trimmed_arr[int(i)])

        # Преобразуем список в numpy массив для удобства расчетов
        month_profit = np.array(month_profit)
        # Считаем количество месяцев с положительной прибылью
        count = np.sum(month_profit > 0)

        # Возвращаем среднюю прибыль за месяц в процентах
        return (count / num_full_periods) * 100

    #region Считаем граальность отдельных показателей эффективности (metrics_graal)

    @property
    def recovery_factor_graal(self) -> float:
        """
        Рассчитывает показатель граальности на основе Recovery Factor.

        :return: Показатель граальности.
        """

        recovery_factor_pct = self.recovery_factor_punkt #TODO: Сделать Recovery_pct и переназначить

        # Если используем один показатель рекавери для расчета граальности, то он установит её в диапазоне от 0 до 14/8*100 (175%)
        RecoveryIdeal = 8.0  # Если считаем Граальность только на основе рекавери (Факт / Идеал * 100), то при RecoveryIdeal граальность == 100%
        RecoveryHighLimit = 14.0  # Если этот показатель поднимается выше 14 - то для нас все системы выше 14 уже хорошие
        RecoveryMin = 2.0  # Если этот показатель ниже указанного минимального значения, то Граальность == 0 (выбрасываем из результатов)

        # Если рековери меньше идеала показатели системы будут ухудшаться
        recovery_factor_graal = recovery_factor_pct / RecoveryIdeal

        # Если показатель ниже минимального значения, то Граальность == 0
        if recovery_factor_pct < RecoveryMin:
            recovery_factor_graal = 0

        # Не может улучшать показатели системы больше чем в 3 раза
        if recovery_factor_graal > (RecoveryHighLimit / RecoveryIdeal):
            recovery_factor_graal = (RecoveryHighLimit / RecoveryIdeal)

        return recovery_factor_graal

    @property
    def sharpe_month_days_graal(self) -> float:
        """
        Рассчитывает показатель граальности на основе месячного Шарпа.

        :return: Показатель граальности.
        """

        sharpe_month = self.sharpe_month_days

        # Месячный шарп (можно взять недельный)
        SharpeMonthIdeal = 1.7  # Если шарп меньше 2,2 - показатели системы будут ухудшаться
        SharpeHighLimit = 3.5  # Если шарп поднимается выше 3.5 - все системы рассматриваем как очень хорошие
        SharpeMonthMin = 1.0  # Если шарп ниже 1.0 - обнуляем граальность


        if sharpe_month != None:
            sharpe_month_graal = sharpe_month / SharpeMonthIdeal
            # Если показатель ниже минимального значения, то Граальность == 0
            if sharpe_month < SharpeMonthMin:
                sharpe_month_graal = 0

            # Не может улучшать показатели системы больше чем в 1.5 раза
            if sharpe_month_graal > (SharpeHighLimit / SharpeMonthIdeal):
                sharpe_month_graal = (SharpeHighLimit / SharpeMonthIdeal)
        else:
            sharpe_month_graal = 0
        # Рассчитываем начальное значение граальности на основе месячного Шарпа




        return sharpe_month_graal

    @property
    def sortino_month_days_graal(self) -> float:
        """
        Рассчитывает показатель граальности на основе месячного Sortino.

        :return: Показатель граальности.
        """

        sortino_month = self.sortino_month_days


        # Месячный сортино (можно взять недельный)
        SortinoMonthIdeal = 2.6  # Если сортино меньше 2,6 - показатели системы будут ухудшаться
        SortinoHighLimit = 4.0  # Если Sortino поднимается выше 4.0 - все системы рассматриваем как очень хорошие
        SortinoMonthMin = 1.5  # Если Sortino ниже 1.5 - обнуляем граальность

        # Рассчитываем начальное значение граальности на основе месячного Шарпа
        if sortino_month is not None:
            sortino_month_graal = sortino_month / SortinoMonthIdeal
        else:
            # Обработайте случай, когда sortino_month равен None
            # Например, установите sortino_month_graal в значение по умолчанию или выведите сообщение об ошибке
            sortino_month_graal = 0  # или любое другое значение по умолчанию
            print("Ошибка: sortino_month имеет значение None")
            return sortino_month_graal

        # Если показатель ниже минимального значения, то Граальность == 0
        if sortino_month < SortinoMonthMin:
            sortino_month_graal = 0
            return sortino_month_graal

        # Не может улучшать показатели системы больше чем в 1.5 раза

        if sortino_month_graal > (SortinoHighLimit / SortinoMonthIdeal):
            sortino_month_graal = (SortinoHighLimit / SortinoMonthIdeal)

        return sortino_month_graal

    @property
    def avg_profit_pct_to_entry_price_graal(self) -> float:
        """
        Рассчитывает показатель граальности на основе среднего процента сделки к цене входа.

        :return: Показатель граальности.
        """

        # Средний процент сделки в % (Убираем щипачей)
        AvgProfitPctToEntryPriceIdeal = 0.15  # Если меньше 0,15 - показатели системы будут ухудшаться
        AvgProfitPctToEntryPriceHighLimit = 0.15  # Все система с профитом больше чем 0.15% не улучшают граальность (рассматриваем их как однозначные)
        AvgProfitPctToEntryPriceMin = 0.04  # Если средний размер сделки ниже указанного - обнуляемся

        profit_pct_to_entry_price_avg = 0.25 #self.profit_pct_to_entry_price_avg #TODO: сделать этот показатель на основе positions_arr

        # Рассчитываем начальное значение граальности на основе среднего процента сделки к цене входа
        avg_profit_pct_to_entry_price_graal = (profit_pct_to_entry_price_avg / AvgProfitPctToEntryPriceIdeal)

        # Если средний процент сделки ниже минимального значения, то Граальность == 0
        if profit_pct_to_entry_price_avg < AvgProfitPctToEntryPriceMin:
            avg_profit_pct_to_entry_price_graal = 0

        # Не может улучшать показатели системы больше чем в 1 раз
        if avg_profit_pct_to_entry_price_graal > (AvgProfitPctToEntryPriceHighLimit / AvgProfitPctToEntryPriceIdeal):
            avg_profit_pct_to_entry_price_graal = (AvgProfitPctToEntryPriceHighLimit / AvgProfitPctToEntryPriceIdeal)

        return avg_profit_pct_to_entry_price_graal

    @property
    def trades_in_year_graal(self) -> float:
        """
        Рассчитывает показатель граальности на основе количества сделок в год.

        :return: Показатель граальности.
        """

        # Количество сделок в год (отсеиваем ленивые системы, которые очень лениво торгуют - проводят малое кол-во сделок в год)
        TradesInYearIdeal = 80  # Если меньше 100 - показатели системы будут ухудшаться
        TradesInYearHighLimit = 150  # Если сделок больше чем указанное кол-во то уже не обращаем на это внимание
        TradesInYearMin = 20  # Если количество сделок в год < 20 - обнуляем показатель

        trades_in_year = self.trades_per_year

        # Рассчитываем начальное значение граальности на основе количества сделок в год
        trades_in_year_graal = trades_in_year / TradesInYearIdeal

        # Если количество сделок в год ниже минимального значения, то Граальность == 0
        if trades_in_year < TradesInYearMin:
            trades_in_year_graal = 0

        # Не может улучшать показатели системы больше чем в 2 раза
        if trades_in_year_graal > (TradesInYearHighLimit / TradesInYearIdeal):
            trades_in_year_graal = (TradesInYearHighLimit / TradesInYearIdeal)

        return trades_in_year_graal

    @property
    def calmar_coeff_start_capital_graal(self) -> float:
        """
        Рассчитывает показатель граальности на основе коэффициента Кальмара.

        :return: Показатель граальности.
        """

        calmar_koeff_pct = self.calmar_coeff_start_capital

        # Коэфф. Кальмара (Даём приоритет системам у которых больше соотношение прибыльности к риску)
        CalmarKoeffPctIdeal = 2.0  # Если меньше 4 - показатели системы будут ухудшаться
        CalmarKoeffHighLimit = 4.0  # Все системы с показателем выше указанного воспринимаем как Отличные
        CalmarKoeffMin = 1.0  # Если кальмар меньше 1.3 то обнуляем



        # Рассчитываем начальное значение граальности на основе коэффициента Кальмара
        calmar_koeff_pct_graal = calmar_koeff_pct / CalmarKoeffPctIdeal

        # Если показатель Кальмара ниже минимального значения, то Граальность == 0
        if calmar_koeff_pct < CalmarKoeffMin:
            calmar_koeff_pct_graal = 0

        # Не может улучшать показатели системы больше чем в 2 раза
        if calmar_koeff_pct_graal > (CalmarKoeffHighLimit / CalmarKoeffPctIdeal):
            calmar_koeff_pct_graal = (CalmarKoeffHighLimit / CalmarKoeffPctIdeal)

        return calmar_koeff_pct_graal

    @property
    def calmar_coeff_max_eqty_graal(self) -> float:
        """
        Рассчитывает показатель граальности на основе коэффициента Кальмара.

        :return: Показатель граальности.
        """

        calmar_koeff_pct = self.calmar_coeff_max_eqty

        # Коэфф. Кальмара (Даём приоритет системам у которых больше соотношение прибыльности к риску)
        CalmarKoeffPctIdeal = 3.0  # Если меньше 4 - показатели системы будут ухудшаться
        CalmarKoeffHighLimit = 6.0  # Все системы с показателем выше указанного воспринимаем как Отличные
        CalmarKoeffMin = 1.3  # Если кальмар меньше 1.3 то обнуляем

        # Рассчитываем начальное значение граальности на основе коэффициента Кальмара
        calmar_koeff_pct_graal = calmar_koeff_pct / CalmarKoeffPctIdeal

        # Если показатель Кальмара ниже минимального значения, то Граальность == 0
        if calmar_koeff_pct < CalmarKoeffMin:
            calmar_koeff_pct_graal = 0

        # Не может улучшать показатели системы больше чем в 2 раза
        if calmar_koeff_pct_graal > (CalmarKoeffHighLimit / CalmarKoeffPctIdeal):
            calmar_koeff_pct_graal = (CalmarKoeffHighLimit / CalmarKoeffPctIdeal)

        return calmar_koeff_pct_graal

    #TODO: Добавить показатель идеальности выхода ProfitToMpr и % прибыльных дней, месяцев, ...

    @property
    def pft_to_mfe_graal(self) -> float:
        """
        Рассчитывает показатель граальности на основе Profit to MFE в %.

        :return: Показатель граальности.
        """
        # Оцениваем идеальность выходов (Profit to MFE в %)
        PftToMfeIdeal = 30.0  # Если меньше 35 - показатели системы будут ухудшаться
        PftToMfeHighLimit = 60.0  # Все системы с показателем выше указанного воспринимаем как Отличные

        profit_to_mfe_pct = 30 #TODO: Реализовать эту функцию из positions_arr self.profit_to_mfe_pct

        # Рассчитываем начальное значение граальности на основе Profit to MFE
        pft_to_mfe_graal = profit_to_mfe_pct / PftToMfeIdeal

        # Не может улучшать показатели системы больше чем в 3 раза
        if pft_to_mfe_graal > PftToMfeHighLimit / PftToMfeIdeal:
            pft_to_mfe_graal = PftToMfeHighLimit / PftToMfeIdeal

        return pft_to_mfe_graal

    @property
    def quartals_plus_pct_graal(self) -> float:
        """
        Рассчитывает показатель граальности на основе среднего процента прибыльных периодов.

        :param pct_profitable_periods_avg: Средний процент прибыльных периодов.
        :return: Показатель граальности.
        """

        pct_profitable_periods_avg = self.quartals_plus_pct

        # Если меньше 70 - показатели системы будут ухудшаться
        PlusPeriodsAvgPctIdeal = 80.0

        # Все системы с показателем выше указанного воспринимаем как Отличные
        PlusPeriodsAvgPctHighLimit = 100.0

        # Рассчитываем показатель граальности как отношение среднего процента прибыльных периодов к идеальному значению
        PlusPeriodsAvgPctGraal = pct_profitable_periods_avg / PlusPeriodsAvgPctIdeal

        # Не может улучшать показатели системы больше чем в 3 раза
        if PlusPeriodsAvgPctGraal > PlusPeriodsAvgPctHighLimit / PlusPeriodsAvgPctIdeal:
            PlusPeriodsAvgPctGraal = PlusPeriodsAvgPctHighLimit / PlusPeriodsAvgPctIdeal

        return PlusPeriodsAvgPctGraal

    @property
    def months_plus_pct_graal(self) -> float:
        """
        Рассчитывает показатель граальности на основе среднего процента прибыльных периодов.

        :param pct_profitable_periods_avg: Средний процент прибыльных периодов.
        :return: Показатель граальности.
        """

        pct_profitable_periods_avg = self.months_plus_pct

        # Если меньше 70 - показатели системы будут ухудшаться
        PlusPeriodsAvgPctIdeal = 65.0

        # Все системы с показателем выше указанного воспринимаем как Отличные
        PlusPeriodsAvgPctHighLimit = 75.0

        # Рассчитываем показатель граальности как отношение среднего процента прибыльных периодов к идеальному значению
        PlusPeriodsAvgPctGraal = pct_profitable_periods_avg / PlusPeriodsAvgPctIdeal

        # Не может улучшать показатели системы больше чем в 3 раза
        if PlusPeriodsAvgPctGraal > PlusPeriodsAvgPctHighLimit / PlusPeriodsAvgPctIdeal:
            PlusPeriodsAvgPctGraal = PlusPeriodsAvgPctHighLimit / PlusPeriodsAvgPctIdeal

        return PlusPeriodsAvgPctGraal

    @property
    def days_plus_pct_graal(self) -> float:
        """
        Рассчитывает показатель граальности на основе среднего процента прибыльных периодов.

        :param pct_profitable_periods_avg: Средний процент прибыльных периодов.
        :return: Показатель граальности.
        """

        pct_profitable_periods_avg = self.days_plus_pct

        # Если меньше 70 - показатели системы будут ухудшаться
        PlusPeriodsAvgPctIdeal = 50.0

        # Все системы с показателем выше указанного воспринимаем как Отличные
        PlusPeriodsAvgPctHighLimit = 70.0

        # Рассчитываем показатель граальности как отношение среднего процента прибыльных периодов к идеальному значению
        PlusPeriodsAvgPctGraal = pct_profitable_periods_avg / PlusPeriodsAvgPctIdeal

        # Не может улучшать показатели системы больше чем в 3 раза
        if PlusPeriodsAvgPctGraal > PlusPeriodsAvgPctHighLimit / PlusPeriodsAvgPctIdeal:
            PlusPeriodsAvgPctGraal = PlusPeriodsAvgPctHighLimit / PlusPeriodsAvgPctIdeal

        return PlusPeriodsAvgPctGraal

    @property
    def beard_coeff_daily_graal(self) -> float:
        """
        Рассчитывает показатель граальности на основе коэффициента бороды.

        :return: Показатель граальности.
        """

        beard_koeff = self.beard_coeff_daily

        # Коэфф. Бороды (Даём приоритет системам у которых много маленьких бород)
        koeff_beard_ideal = 18.0  # Если меньше 50 - показатели системы будут ухудшаться
        koeff_beard_high_limit = 40.0  # Все системы с показателем выше указанного воспринимаем как Отличные

        # Рассчитываем показатель граальности как отношение коэффициента бороды к идеальному значению
        koeff_beard_graal = beard_koeff / koeff_beard_ideal

        # Не может улучшать показатели системы больше чем в 3 раза
        if koeff_beard_graal > koeff_beard_high_limit / koeff_beard_ideal:
            koeff_beard_graal = koeff_beard_high_limit / koeff_beard_ideal

        return koeff_beard_graal

    @property
    def daily_beards_per_year_graal(self) -> float:
        """
        Рассчитывает показатель граальности на основе коэффициента бороды.

        :return: Показатель граальности.
        """

        beard_koeff = self.daily_beards_per_year

        # Коэфф. Бороды (Даём приоритет системам у которых много маленьких бород)
        koeff_beard_ideal = 22.0  # Если меньше 50 - показатели системы будут ухудшаться
        koeff_beard_high_limit = 50.0  # Все системы с показателем выше указанного воспринимаем как Отличные

        # Рассчитываем показатель граальности как отношение коэффициента бороды к идеальному значению
        koeff_beard_graal = beard_koeff / koeff_beard_ideal

        # Не может улучшать показатели системы больше чем в 3 раза
        if koeff_beard_graal > koeff_beard_high_limit / koeff_beard_ideal:
            koeff_beard_graal = koeff_beard_high_limit / koeff_beard_ideal

        return koeff_beard_graal

    @property
    def daily_beard_max_graal(self) -> float:
        """
        Рассчитывает показатель граальности на основе коэффициента бороды.

        :return: Показатель граальности.
        """

        beard_koeff = self.daily_beard_max

        # Коэфф. Бороды (Даём приоритет системам у которых много маленьких бород)
        koeff_beard_ideal = 150.0  # Если больше 150 - показатели системы будут ухудшаться
        koeff_beard_high_limit = 90.0  # Все системы с показателем ниже указанного воспринимаем как Отличные

        # Рассчитываем показатель граальности как отношение коэффициента бороды к идеальному значению
        koeff_beard_graal =  koeff_beard_ideal / beard_koeff if beard_koeff !=0 else 0

        # Не может улучшать показатели системы больше чем в 3 раза
        if koeff_beard_graal > koeff_beard_ideal / koeff_beard_high_limit:
            koeff_beard_graal = koeff_beard_ideal / koeff_beard_high_limit

        return koeff_beard_graal

    @property
    def recovery_and_sharp_graal(self) -> float:
        """
        Рассчитывает показатель граальности на основе Recovery Factor и Sharpe Ratio.

        :return: Показатель граальности.
        """
        recovery_factor_pct = self.recovery_factor_punkt
        sharpe_month = self.sharpe_month_days

        recovery_ideal = 9
        sharp_month_ideal = 2.5


        # Расчётный показатель на основе коэффициентов Recovery Factor и Sharpe Ratio
        RecoveryAndSharp = ((recovery_factor_pct / recovery_ideal) * (sharpe_month / sharp_month_ideal))

        # Если показатель >1 - хорошая система. Если <1 - не очень.

        return RecoveryAndSharp

    #endregion

    @property
    def graal_metr_no_reinvest (self) -> float:
        """
        Рассчитывает показатель граальности на основе нескольких метрик.

        :return: Показатель граальности.
        """

        calmar_koeff_gr = self.calmar_coeff_start_capital_graal
        beard_coeff_daily_gr = self.beard_coeff_daily_graal
        months_plus_pct_gr = self.months_plus_pct_graal
        quartals_plus_pct_gr = self.quartals_plus_pct_graal
        sharpe_month_days_gr = self.sharpe_month_days_graal
        sortino_month_days_gr = self.sortino_month_days_graal


        # Проверяем каждую переменную и выводим сообщение, если она равна None
        if calmar_koeff_gr is None:
            #print("calmar_koeff_gr is None")
            calmar_koeff_gr = 1.0

        if beard_coeff_daily_gr is None:
            #print("beard_coeff_daily_gr is None")
            beard_coeff_daily_gr = 1.0

        if months_plus_pct_gr is None:
            #print("months_plus_pct_gr is None")
            months_plus_pct_gr = 1.0

        if quartals_plus_pct_gr is None:
            #print("quartals_plus_pct_gr is None")
            quartals_plus_pct_gr = 1.0

        if sharpe_month_days_gr is None:
            #print("sharpe_month_days_gr is None")
            sharpe_month_days_gr = 1.0

        if sortino_month_days_gr is None:
            #print("sortino_month_days_gr is None")
            sortino_month_days_gr = 1.0

        # Теперь выполняем умножение



        proisv = (
                calmar_koeff_gr *
                beard_coeff_daily_gr *
                months_plus_pct_gr *
                quartals_plus_pct_gr *
                sharpe_month_days_gr *
                sortino_month_days_gr
        )

        # Взять корень 6-й степени (потому что 6 показателя)
        graal_metr = math.pow(proisv, (1 / 6))

        # Перевести в проценты
        graal_metr = graal_metr * 100

        return graal_metr

    @property
    def graal_metr_with_reinvest (self) -> float:
        """
        Рассчитывает показатель граальности на основе нескольких метрик.

        :return: Показатель граальности.
        """

        calmar_koeff_gr = self.calmar_coeff_max_eqty_graal
        beard_coeff_daily_gr = self.beard_coeff_daily_graal
        months_plus_pct_gr = self.months_plus_pct_graal
        quartals_plus_pct_gr = self.quartals_plus_pct_graal
        sharpe_month_days_gr = self.sharpe_month_days_graal
        sortino_month_days_gr = self.sharpe_month_days_graal

        # Проверяем каждую переменную и выводим сообщение, если она равна None
        if calmar_koeff_gr is None:
            #print("calmar_koeff_gr is None")
            calmar_koeff_gr = 1.0

        if beard_coeff_daily_gr is None:
            #print("beard_coeff_daily_gr is None")
            beard_coeff_daily_gr = 1.0

        if months_plus_pct_gr is None:
            #print("months_plus_pct_gr is None")
            months_plus_pct_gr = 1.0

        if quartals_plus_pct_gr is None:
            #print("quartals_plus_pct_gr is None")
            quartals_plus_pct_gr = 1.0

        if sharpe_month_days_gr is None:
            #print("sharpe_month_days_gr is None")
            sharpe_month_days_gr = 1.0

        if sortino_month_days_gr is None:
            #print("sortino_month_days_gr is None")
            sortino_month_days_gr = 1.0

        # Теперь выполняем умножение


        proisv = (
                calmar_koeff_gr *
                beard_coeff_daily_gr *
                months_plus_pct_gr *
                quartals_plus_pct_gr *
                sharpe_month_days_gr *
                sortino_month_days_gr
        )

        # Взять корень 6-й степени (потому что 6 показателя)
        graal_metr = math.pow(proisv, (1 / 6))

        # Перевести в проценты
        graal_metr = graal_metr * 100

        return graal_metr


