import pymysql

HOST = "5.104.86.165"
PORT = 9453
USER = "artem"
PASSWORD = "UWf5M5VzDjYDCNX5dL5Zd6Mf"
DATABASE = "artem"


class MysqlConnector:
    def __init__(self) -> None:
        self.connection = pymysql.connect(
            host=HOST,  # только IP, без порта
            port=PORT,            # порт указывается отдельно
            user=USER,
            password=PASSWORD,
            database=DATABASE,
            cursorclass=pymysql.cursors.DictCursor,
            connect_timeout=10    # увеличение таймаута
        )

    def get_strategy_id_by_name(self, strategy_class):
        with self.connection.cursor() as cursor:
            sql = f"SELECT * FROM strategies WHERE name='{strategy_class}'"
            cursor.execute(sql)
            results = cursor.fetchall()
            
            if len(results) == 0: raise ValueError(f"Can't find stratrgy with name: {strategy_class}")
            if len(results) > 1:  raise ValueError("Find more then one strategy")

            return results[0]["id"]

    def save(self, strategy_class, symbol, timeframe, start_date, end_date, net_profit, params, metrics):
        # Получаем ID стратегии по её имени (из БД)
        strategy_id = self.get_strategy_id_by_name(strategy_class=strategy_class)
        
        # Зполняем experiments базу
        with self.connection.cursor() as cursor:
            # Вставка одной записи
            sql = "INSERT INTO experiments (strategy_id, crypto_pair, timeframe, start_date, end_date, calmar_coeff_start_capital, graal_metr_no_reinvest, graal_metr_with_reinvest) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)"
            cursor.execute(sql, (strategy_id, symbol, timeframe, start_date, end_date, metrics["calmar_coeff_start_capital"], metrics["graal_metr_no_reinvest"], metrics["graal_metr_with_reinvest"]))
            experiment_id = cursor.lastrowid
        
        # Заполняем experiments_params базу
        with self.connection.cursor() as cursor:
            params_to_sql = []
            for pname in params:
                params_to_sql.append([experiment_id, pname, str(params[pname])])
            sql = "INSERT INTO experiment_params (experiment_id, param_name, param_value) VALUES (%s, %s, %s)"
            cursor.executemany(sql, params_to_sql)

        # Заполняем net_profit
        with self.connection.cursor() as cursor:
            net_profit_sql = '[' + ", ".join([str(v) for v in net_profit]) + ']'
            sql = "INSERT INTO experiment_results (experiment_id, results) VALUES (%s, %s)"
            cursor.execute(sql, (experiment_id, net_profit_sql))

        self.connection.commit()
        
        print("Данные об эксперименте успешно сохранены в базу")
            