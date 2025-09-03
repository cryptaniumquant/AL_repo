def MaxPctRiskBinance(
    SummForSystem,  # 50 - Сумма, выделяемая системе
    maxPctRisk,  # 1.5 % риска на одну сделку
    TargetEntryPrice,  # 9100 - Ожидаемая цена входа в позицию
    StartStopLoss,  # 9050 - Первоначальный Стоп-Лосс
    minLotSizeCrypta,  # 0.001 - величина минимального лота в единицах первой пары (0,001 для BTC)
    minLotSizeUsd=0,  # 0 величина минимального лота в USD
    maxCountOfMinLot=10000000,  # 100 Максимальное кол-во минимальных лотов
    TradeMinContract=False  # False торговать минимально возможным контрактом?
):
    """
    Вычисляет максимальный размер позиции (лот) для торговли на Binance с заданным процентом риска.

    :param SummForSystem: float - Сумма, выделяемая системе
    :param maxPctRisk: float - Процент риска на одну сделку
    :param TargetEntryPrice: float - Ожидаемая цена входа в позицию
    :param StartStopLoss: float - Первоначальный Стоп-Лосс
    :param minLotSizeCrypta: float - Величина минимального лота в единицах первой пары (например, 0.001 для BTC)
    :param minLotSizeUsd: float - Величина минимального лота в USD
    :param maxCountOfMinLot: int - Максимальное количество минимальных лотов
    :param TradeMinContract: bool - Торговать минимально возможным контрактом?
    :return: float - Максимальный размер позиции (количество лотов)
    """
    # Получаем количество знаков после запятой для округления лота
    DecimalPlacesForRounding = 2 #DecimalPlacesCount(minLotSizeCrypta)  # 3 - знаков после запятой в лоте

    # Объявляем переменные
    ResultLotSize = 0  # Искомое количество лотов, которое будем ставить в заявку
    RiskForOneMinLot = 0  # Риск потери на один минимальный лот в долларах
    RiskForOneTrade = 0  # Можно потерять в одной сделке не больше чем ... долларов

    # Если указано торговать одним контрактом, возвращаем минимально возможный размер позиции
    # if TradeMinContract:
    #     return MinPosSizeBinanceSpot(TargetEntryPrice, minLotSizeCrypta, minLotSizeUsd)

    # Определяем, сколько долларов мы готовы потерять в одной сделке
    RiskForOneTrade = SummForSystem * maxPctRisk / 100.0  # 50 * 1.5 / 100 = 0.75

    # Определяем риск на минимально возможный лот в долларах
    if TargetEntryPrice > StartStopLoss:  # Если позиция длинная
        giveUSDTBuying = minLotSizeCrypta * TargetEntryPrice  # 0.001 * 9100 = 9.1 - Отдадим долларов при покупке минимального лота по цене входа
        takeUSDTSelling = minLotSizeCrypta * StartStopLoss  # 0.001 * 9050 = 9.05 - Получим долларов при продаже минимального лота по стопу
        RiskForOneMinLot = takeUSDTSelling - giveUSDTBuying  # 9.05 - 9.1 = -0.05 = Потеряем долларов при срабатывании стопа (отрицательное число)
    else:  # Если позиция короткая
        takeUSDTSelling = minLotSizeCrypta * TargetEntryPrice  # Получим долларов при продаже минимального лота по цене входа
        giveUSDTBuying = minLotSizeCrypta * StartStopLoss  # Отдадим долларов при покупке минимального лота по стопу
        RiskForOneMinLot = takeUSDTSelling - giveUSDTBuying  # Потеряем долларов при срабатывании стопа (отрицательное число)

    # Если риск на один минимальный лот положительный, возвращаем 0 (некорректный стоп)
    if RiskForOneMinLot > 0:
        return 0

    # Определяем количество контрактов с дробной частью
    ResultLotSize = RiskForOneTrade / (-1.0 * RiskForOneMinLot) * minLotSizeCrypta  # 0.75 / (0.05 * 0.001)
    # Округляем до нужного размера (до 3-го знака)
    ResultLotSize = round(ResultLotSize, DecimalPlacesForRounding)
    # Ограничиваем количество лотов для торговли
    ResultLotSize = min(ResultLotSize, maxCountOfMinLot * minLotSizeCrypta)

    # Если стоимость размера лота меньше, чем минимальная стоимость в долларах, возвращаем 0
    if ResultLotSize * TargetEntryPrice < minLotSizeUsd:
        ResultLotSize = 0

    return ResultLotSize  # Результат, который выдаёт этот метод
