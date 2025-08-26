-- Таблица стратегий (общее описание)
CREATE TABLE strategies (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,    -- "FoBo", "RSIStrategy" и т. д.
    description TEXT
);

-- Таблица параметров стратегий (какие параметры вообще возможны)
CREATE TABLE strategy_possible_params (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    strategy_id INTEGER NOT NULL,
    param_name TEXT NOT NULL,     -- "rsi_period", "ema_length" и т. д.
    param_type TEXT NOT NULL,     -- "int", "float", "str", "bool"
    default_value TEXT,           -- значение по умолчанию
    FOREIGN KEY (strategy_id) REFERENCES strategies(id)
);

-- Таблица экспериментов (запуски тестов)
CREATE TABLE experiments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    strategy_id INTEGER NOT NULL,
    crypto_pair TEXT NOT NULL,    -- "BTC/USDT"
    timeframe TEXT NOT NULL,      -- "1h", "4h"
    start_date DATETIME NOT NULL,
    end_date DATETIME NOT NULL,
    FOREIGN KEY (strategy_id) REFERENCES strategies(id)
);

-- Таблица параметров эксперимента (конкретные значения для данного запуска)
CREATE TABLE experiment_params (
    experiment_id INTEGER NOT NULL,
    param_name TEXT NOT NULL,     -- "rsi_period"
    param_value TEXT NOT NULL,    -- "14" (хранится как строка)
    FOREIGN KEY (experiment_id) REFERENCES experiments(id)
);


-- Таблица для хранения кривой net profit для каждого запуска
CREATE TABLE net_profit (
    experiment_id INTEGER NOT NULL,
    value DECIMAL,
    position INTEGER,
    FOREIGN KEY (experiment_id) REFERENCES experiments(id)
);