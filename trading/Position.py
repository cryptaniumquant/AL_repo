from dataclasses import dataclass
import datetime


@dataclass
class Position:
    IsActive: bool
    IsLong: bool
    EntryBarNum: int = 0
    ExitBarNum: int = 0
    Lots: float = 0.0
    OrderPrice: float = 0.0
    ExitLimitPrice: float = 0.0
    EntryDate: datetime.datetime = None
    ExitDate: datetime.datetime = None

    def CloseAtPrice(self, bar, exit_limit_price, exitDate):
        self.IsActive = False
        self.ExitBarNum = bar
        self.ExitLimitPrice = exit_limit_price
        self.ExitDate = exitDate
    
    def Profit(self):
        comission = (self.OrderPrice * 0.001 + self.ExitLimitPrice*0.001) * self.Lots

        if self.IsLong:
            return (self.ExitLimitPrice - self.OrderPrice) * self.Lots - comission
        else:
            return (self.OrderPrice - self.ExitLimitPrice) * self.Lots - comission

    def CurrentProfit(self, current_price):
        comission = self.OrderPrice * 0.001 * self.Lots

        if self.IsLong:
            return (current_price - self.OrderPrice) * self.Lots - comission
        else:
            return (self.OrderPrice - current_price) * self.Lots - comission
