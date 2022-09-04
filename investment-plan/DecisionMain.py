import pandas as pd
from FundDataCrawler import FundDataCrawler
from FundDecision import FundDecision

fundDataCrawler = FundDataCrawler()
fundDecision = FundDecision()
total = 30000
frequence = 7
wave = 5
days = 250
balance = 10

for code in ['161725']:
    fundData = fundDataCrawler.getFund(code)
    data = pd.read_csv('./investment-plan/train/' + code +
                       '.csv', header=None, dtype=object)
    for cell in data.values.tolist():
        start_time = pd.to_datetime(cell[0])
        end_time = pd.to_datetime(cell[1])

        df = fundDecision.getData(fundData, start_time, end_time)
        start_time = fundDecision.getStartTime(df, start_time)

        realDays = (pd.to_datetime(end_time) - start_time).days
        invest_money = total / (realDays / frequence) * 2  # 每次定投金额
        print("{}开始计算，定投频率：{}日，每次定投：{}，定投开始时间：{}，定投结束时间：{}".format(code, frequence, invest_money, start_time, end_time))

        # res, buy, money = fundDecision.smart_invest(fundData, df, frequence, invest_money, start_time, end_time, days, wave, 1, 3000)
        # print(money)
        # fundDecision.myplot(df, res, buy, "定投频率：{}日，每次定投：{}，收益率：".format(frequence, invest_money))

        # res, buy, money = fundDecision.smart_invest(fundData, df, frequence, invest_money, start_time, end_time, days, wave, 2, 3000)
        # print(money)
        # fundDecision.myplot(df, res, buy, "定投频率：{}日，每次定投：{}，收益率：".format(frequence, invest_money))

        # res, buy, money = fundDecision.smart_invest(fundData, df, frequence, invest_money, start_time, end_time, days, wave, 3, 3000)
        # fundDecision.myplot(df, res, buy, "定投频率：{}日，每次定投：{}，收益率：".format(frequence,invest_money))

        fundDecision.calculate_invest(fundData, '20190821', 7, invest_money, 7813, 0.9738, 0.053, days, wave, 1)

        fundDecision.calculate_invest(fundData, '20190828', 7, invest_money, 12492, 0.9915, 0.055, days, wave, 1)

