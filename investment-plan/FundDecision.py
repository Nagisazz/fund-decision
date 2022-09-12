import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import math

class FundDecision:

    warnings.filterwarnings('ignore')
    plt.rcParams['font.family'] = 'SimHei'
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['figure.figsize'] = (12, 8)

    def getData(self, fundData, start, end):
        '''
        获取指定时间范围内数据
        '''
        data = fundData.drop(fundData[fundData.index < start].index, axis=0, inplace=False)
        data.drop(data[data.index > end].index, axis=0, inplace=True)
        return data

    def getStartTime(self, df, start):
        while df.index[0] != start:
            start += np.timedelta64(1, 'D')
        return start

    def mean_days(self, fundData, df, meanDays, waveDays, endDate):
        '''
        获取n日均线,最近m日振幅
        :param fundData: 所有数据集
        :param df: 定投期内的数据集
        :param meanDays: 均线天数
        :params waveDays: 最近m日振幅
        :return average,wavelength: 均线、近m日振幅
        '''
        average = []
        wavelength = []
        # 均线
        start = df.head(1).index - np.timedelta64(meanDays + 1, "D")
        start = start.astype('str')[0].replace("-", "")  # 转换为字符串
        mean_df = self.getData(fundData, start, endDate)
        for i in df.index:
            start_date = i - np.timedelta64(meanDays + 1, "D")
            ave = mean_df[(mean_df.index > start_date) & (mean_df.index < i)]['收盘价'].mean()
            average.append(ave)
        # 振幅
        start2 = df.head(1).index - np.timedelta64(waveDays + 1, "D")
        start2 = start2.astype('str')[0].replace("-", "")  # 转换为字符串
        wave_df = self.getData(fundData, start2, endDate)
        for i in df.index:
            start_date = i - np.timedelta64(waveDays + 1, "D")
            interval = wave_df[(wave_df.index > start_date) & (wave_df.index < i)]['收盘价']
            length = interval.max()/interval.min() - 1  # 最大涨跌幅
            wavelength.append(length)

        return average, wavelength

    def smart_invest(self, fundData, df, frequence, investMoney, startTime, endTime, meanDays, waveDays, type, initMoney):
        '''
        定投计算
        :param fundData: 所有数据集
        :param df: 定投期内的数据集
        :param frequence: 定投频率
        :param investMoney: 每次定投金额
        :param startTime: 定投起始日期
        :param endTime: 定投结束时间
        :param meanDays: 参考均线天数
        :param waveDays: 参考振幅天数
        :param type: 策略类型
        :param initMoney: 初始金额

        :return over: 资金变化图表数据
        :return invest_log: 定投图表打点数据
        :return invest_rate_log: 定投比率日志
        '''
        invest_log = {}  # 每次定投的日期记录(日期:大盘指数)
        invest_rate_log = {}  # 每次定投的日期记录(日期:比率)
        invest_day = startTime  # 每次投资的时间
        invest_amount = initMoney  # 总投资金额
        buy_amount = initMoney  # 总买入金额
        sale_amount = 0  # 总卖出金额
        profile = 0  # 总投资收益
        balance = 0  # 账户余额
        last_yield = 0  # 上一次定投收益率

        profile_log = []  # 总收益日志
        invest_amount_log = []  # 账户投资金额日志
        balance_log = []  # 余额日志
        Yield = []  # 收益率日志

        res = self.mean_days(fundData, df, meanDays, waveDays, endTime)
        df["均线"] = res[0]  # 获取均线
        df["振幅"] = res[1]  # 获取振幅
        for date, quote_change, index, mean, wave in zip(df.index, df['涨跌幅'], df['收盘价'], df["均线"], df["振幅"]):

            # 判断是否为定投日
            if date == invest_day:
                if (invest_day == startTime):
                    last_price = startTime
                else:
                    formal_day = 1
                    while (True):
                        last_price = date - np.timedelta64(formal_day, "D")  # 前1天的收盘价
                        if (last_price in df.index.tolist()):
                            break
                        else:
                            formal_day += 1
                    mean = df[df.index == last_price]["均线"][0]
                    wave = df[df.index == last_price]["振幅"][0]
                    last_price = df[df.index == last_price]["收盘价"][0]

                    all = False
                    if type == 1:
                        rate = self.stratege1(
                            mean, wave, last_price, last_yield, profile/buy_amount)
                    elif type == 2:
                        rate, all = self.stratege2(
                            mean, wave, last_price, last_yield, profile/buy_amount)
                    elif type == 3:
                        rate = self.stratege3(mean, wave, last_price) - 1.1

                    if all:
                        rate = rate if rate < 1 else 1
                        # 按照余额止盈
                        price = math.floor(balance*rate/10) * 10
                        sale_amount += price
                        invest_amount -= price
                    else:
                        if balance + investMoney*rate < 0:
                            rate = 0
                        price = math.floor(investMoney*rate/10) * 10
                        # 按照定投基准金额定投
                        if rate < 0:
                            sale_amount += -price
                        else:
                            buy_amount += price
                        invest_amount += price
                    invest_log[invest_day] = index  # 记录定投当日的指数
                    invest_rate_log[invest_day] = rate  # 记录定投当日的投资比率

                    last_yield = (profile + quote_change*balance)/buy_amount # 记录定投当日的收益率

                # 判断7天后是否为交易日,如果不是则往后加1天直到找到交易日
                invest_day += np.timedelta64(frequence, 'D')
                flag = 0
                while (True):
                    if (df[df.index == invest_day].index == invest_day):
                        break
                    else:
                        invest_day += np.timedelta64(1, 'D')
                        flag += 1
                        if (flag == 100):
                            break

            if date != startTime:
                profile += quote_change*balance  # 计算当天收益
                profile_log.append(profile)

            invest_amount_log.append(invest_amount)
            balance = invest_amount + profile  # 更新账户总资产
            balance_log.append(balance)
            try:
                Yield.append(profile/buy_amount*100)  # 更新收益率
            except:
                Yield.append(0)
        print("总余额：", balance)
        print("总定投：", invest_amount)
        print("总买入：", buy_amount)
        print("总卖出：", sale_amount)
        print("总收益：", profile)
        print("收益率: ", profile/buy_amount*100, "%")

        over = pd.DataFrame({
            "日期": df.index,
            "收益率": Yield,
            "账户资产": balance_log,
            "投资金额": invest_amount_log
        })
        over = over.set_index("日期")
        return over, invest_log, invest_rate_log

    def calculate_invest(self, fund_data, date, frequence, invest_money, current_money, current_worth, current_yield, mean_days, wave_days, type):
        '''
        定投日单独计算
        :param fund_data: 所有数据集
        :param date: 定投日
        :param frequence: 定投频率
        :param invest_money: 每次定投金额
        :param current_money: 当前资产
        :param current_worth: 当前净值
        :param current_yield: 当前收益率
        :param mean_days: 参考均线天数
        :param wave_days: 参考振幅天数
        :param type: 策略类型

        :return all: 是否按全部资产卖出
        :return rate: 定投操作比率
        :return price: 定投金额
        :return mean: 均线净值
        :return wave: 振幅
        :return last_yield: 上次定投日收益率
        '''
        # 计算前一个交易日
        formal_day = 1
        while (True):
            last_date = pd.to_datetime(date) - np.timedelta64(formal_day, "D")
            if (last_date in fund_data.index.tolist()):
                break
            else:
                formal_day += 1

        # 封装数据集
        start = fund_data.head(1).index
        start = start.astype('str')[0].replace("-", "")  # 转换为字符串
        df = fund_data.drop(fund_data[fund_data.index > start].index, axis=0, inplace=False)
        df = df.reset_index()
        df['日期'][0] = last_date
        df=df.set_index('日期')
        df=df.sort_index()

        # 获取均线和振幅数据
        res = self.mean_days(fund_data, df, mean_days, wave_days, pd.to_datetime(last_date))
        df["均线"] = res[0]
        df["振幅"] = res[1]
        mean = df["均线"][0]
        wave = df["振幅"][0]

        # 计算定投频率前一个周期的收益率
        while (True):
            last_start = pd.to_datetime(date) - np.timedelta64(frequence, "D")
            if (last_start in fund_data.index.tolist()):
                break
            else:
                frequence += 1
        last_df = self.getData(fund_data, last_start, pd.to_datetime(date))
        last_worth = last_df['收盘价'][0]
        last_yield = last_worth/current_worth * (1 + current_yield) - 1

        # 计算定投比率
        all = False
        if type == 1:
            rate = self.stratege1(
                mean, wave, current_worth, last_yield, current_yield)
        elif type == 2:
            rate, all = self.stratege2(
                mean, wave, current_worth, last_yield, current_yield)
        elif type == 3:
            rate = self.stratege3(mean, wave, current_worth) - 1.1

        if all:
            rate = rate if rate < 1 else 1
            # 按照余额止盈
            price = -math.floor(current_money*rate/10) * 10
        else:
            if current_money + invest_money*rate < 0:
                rate = 0
            price = math.floor(invest_money*rate/10) * 10
        
        return all, rate, price, mean, wave, last_yield

    def calculate_price(self, fund_data, current_money, date):
        '''
        单独计算给定日期收益
        :param fund_data: 所有数据集
        :param current_money: 当前资产
        :param date: 日期

        :return current_money: 现有资产
        :return income: 收益
        '''
        df = self.getData(fund_data, date, date)
        income = 0
        if df['涨跌幅'].size != 0:
            income = current_money * df['涨跌幅'][0]
            current_money += income
        return current_money, income
        
    def myplot(self, df1, res, buy, titlename):
        '''
        绘制定投结果图
        '''
        plt.figure()
        df1['收盘价'].plot(label="大盘指数")
        plt.scatter(buy.keys(), buy.values(),
                    color="brown", marker=".", label="定投记录")
        plt.legend(loc='best')
        plt.ylabel("指数")
        plt.twinx()

        res['账户资产'].plot(color="red")
        res['投资金额'].plot(color="orange")
        plt.ylabel("元")
        plt.legend()
        plt.title(titlename+":{:.2f}%".format(res.tail(1)["收益率"][0]))
        plt.show()

    def stratege1(self, mean, wave, lastPrice, lastYield, totalYield):
        '''
        定投策略
        :param mean: 均线
        :param wave: 振幅
        :param lastPrice: 前1日收盘价
        :param lastYield: 上次定投日收益率
        :param totalYield: 总收益率
        '''
        bal = 2
        cal = lastPrice/mean - 1  # 大于0,则高于均线

        # 均线、振幅策略
        if (wave > 0.05):
            if (cal >= 0 and cal < 0.1):
                res = 0.5
            elif (cal >= 0.1 and cal < 0.2):
                res = 0
            elif (cal >= 0.2 and cal < 0.3):
                res = -1
            elif (cal >= 0.3 and cal < 0.5):
                res = -3
            elif (cal >= 0.5 and cal < 1):
                res = -5
            elif (cal >= 1):
                res = -10

            elif (cal >= -0.05 and cal < 0):
                res = 0.6
            elif (cal >= -0.1 and cal < -0.05):
                res = 0.9
            elif (cal >= -0.2 and cal < -0.1):
                res = 1.2
            elif (cal >= -0.3 and cal < -0.2):
                res = 1.5
            elif (cal >= -0.4 and cal < -0.3):
                res = 1.8
            elif (cal < -0.4):
                res = 2.1
        else:
            if (cal >= 0 and cal < 0.1):
                res = 0.8
            elif (cal >= 0.1 and cal < 0.2):
                res = 0.5
            elif (cal >= 0.2 and cal < 0.3):
                res = 0
            elif (cal >= 0.3 and cal < 0.5):
                res = -1
            elif (cal >= 0.5 and cal < 1):
                res = -3
            elif (cal >= 1):
                res = -5

            elif (cal >= -0.05 and cal < 0):
                res = 1.2
            elif (cal >= -0.1 and cal < -0.05):
                res = 1.4
            elif (cal >= -0.2 and cal < -0.1):
                res = 1.6
            elif (cal >= -0.3 and cal < -0.2):
                res = 1.8
            elif (cal >= -0.4 and cal < -0.3):
                res = 2.0
            elif (cal < -0.4):
                res = 2.2

        # 上次定投日与这次收益率相差策略
        # 比上次跌的多多买点，涨的多多卖点，做T
        diffYield = totalYield - lastYield
        if res > 0:
            if (diffYield >= -0.05 and diffYield <= 0.05):
                res = res
            elif (diffYield > 0.05 and diffYield < 0.1):
                res = (-res - 0.2) * 0.5
            elif (diffYield >= 0.1 and diffYield < 0.15):
                res = (-res - 0.2) * 1
            elif (diffYield >= 0.15 and diffYield < 0.2):
                res = (-res - 0.2) * 2
            elif (diffYield >= 0.2):
                res = (-res - 0.2) * 3
            elif (diffYield >= -0.1 and diffYield < -0.05):
                res = res + 0.2
            elif (diffYield >= -0.15 and diffYield < -0.1):
                res = res + 0.5
            elif (diffYield >= -0.2 and diffYield < -0.15):
                res = res + 1.0
            elif (diffYield < -0.2):
                res = res + 1.5

        # 总收益率策略
        if (totalYield >= 0.1 and totalYield < 0.15):
            res = (res - 0.5) * bal
        elif (totalYield >= 0.15 and totalYield < 0.2):
            res = (res - 1) * bal
        elif (totalYield >= 0.2 and totalYield < 0.3):
            res = (res - 2) * bal
        elif (totalYield >= 0.3):
            res = (res - 3) * bal
        elif (totalYield >= -0.2 and totalYield < -0.15):
            res = res + 0.5
        elif (totalYield >= -0.3 and totalYield < -0.2):
            res = res + 1
        elif (totalYield < -0.3):
            res = res + 2

        return res

    def stratege2(self, mean, wave, lastPrice, lastYield, totalYield):
        '''
        定投策略
        :param mean: 均线
        :param wave: 振幅
        :param lastPrice: 前1日收盘价
        :param lastYield: 上次定投日收益率
        :param totalYield: 总收益率
        '''
        all = False
        bal = 0.4
        cal = lastPrice/mean-1  # 大于0,则高于均线

        # 均线、振幅策略
        if (wave > 0.05):
            if (cal >= 0 and cal < 0.1):
                res = 0.5
            elif (cal >= 0.1 and cal < 0.2):
                res = 0
            elif (cal >= 0.2 and cal < 0.3):
                res = -1
            elif (cal >= 0.3 and cal < 0.5):
                res = -3
            elif (cal >= 0.5 and cal < 1):
                res = -5
            elif (cal >= 1):
                res = -10

            elif (cal >= -0.05 and cal < 0):
                res = 0.6
            elif (cal >= -0.1 and cal < -0.05):
                res = 0.9
            elif (cal >= -0.2 and cal < -0.1):
                res = 1.2
            elif (cal >= -0.3 and cal < -0.2):
                res = 1.5
            elif (cal >= -0.4 and cal < -0.3):
                res = 1.8
            elif (cal < -0.4):
                res = 2.1
        else:
            if (cal >= 0 and cal < 0.1):
                res = 0.8
            elif (cal >= 0.1 and cal < 0.2):
                res = 0.5
            elif (cal >= 0.2 and cal < 0.3):
                res = 0
            elif (cal >= 0.3 and cal < 0.5):
                res = -1
            elif (cal >= 0.5 and cal < 1):
                res = -3
            elif (cal >= 1):
                res = -5

            elif (cal >= -0.05 and cal < 0):
                res = 0.8 + bal
            elif (cal >= -0.1 and cal < -0.05):
                res = 1.0 + bal
            elif (cal >= -0.2 and cal < -0.1):
                res = 1.2 + bal
            elif (cal >= -0.3 and cal < -0.2):
                res = 1.4 + bal
            elif (cal >= -0.4 and cal < -0.3):
                res = 1.6 + bal
            elif (cal < -0.4):
                res = 1.8 + bal

        # 上次定投日与这次收益率相差策略
        # 比上次跌的多多买点，涨的多多卖点，做T
        diffYield = totalYield - lastYield
        if res > 0:
            if (diffYield >= -0.05 and diffYield <= 0.05):
                res = res
            elif (diffYield > 0.05 and diffYield < 0.1):
                res = (-res - 0.2) * 0.5
            elif (diffYield >= 0.1 and diffYield < 0.15):
                res = (-res - 0.2) * 1
            elif (diffYield >= 0.15 and diffYield < 0.2):
                res = (-res - 0.2) * 2
            elif (diffYield >= 0.2):
                res = (-res - 0.2) * 3
            elif (diffYield >= -0.1 and diffYield < -0.05):
                res = res + 0.2
            elif (diffYield >= -0.15 and diffYield < -0.1):
                res = res + 0.5
            elif (diffYield >= -0.2 and diffYield < -0.15):
                res = res + 1.0
            elif (diffYield < -0.2):
                res = res + 1.5

        # 总收益率策略
        if (totalYield >= 0.05 and totalYield < 0.5):
            if(totalYield < 0.1 and res >= 0.5):
                res = 0
            else:
                res = res if res > 0 or res < -1 else 0
                res = 2 * totalYield - (res / 30)
                all = True
        elif (totalYield >= -0.2 and totalYield < -0.15):
            res = res + 0.5
        elif (totalYield >= -0.3 and totalYield < -0.2):
            res = res + 1
        elif (totalYield < -0.3):
            res = res + 2

        return res, all

    def stratege3(self, mean, wave, lastPrice):
        '''
        定投策略
        :param mean:均线
        :param wave:振幅
        :param lastPrice:前1日收盘价
        '''
        cal = lastPrice/mean-1  # 大于0,则高于均线
        if (cal >= 0 and cal < 0.15):
            return 0.9
        elif (cal > 0.15 and cal < 0.5):
            return 0.8
        elif (cal >= 0.5 and cal < 1):
            return 0.7
        elif (cal >= 1):
            return 0.6

        elif (wave > 0.05):

            if (cal >= -0.05 and cal < 0):
                return 0.6
            elif (cal >= -0.1 and cal < -0.05):
                return 0.7
            elif (cal >= -0.2 and cal < -0.1):
                return 0.8
            elif (cal >= -0.3 and cal < -0.2):
                return 0.9
            elif (cal >= -0.4 and cal < -0.3):
                return 1.0
            elif (cal < -0.4):
                return 1.1
        else:
            if (cal >= -0.05 and cal < 0):
                return 1.8
            elif (cal >= -0.1 and cal < -0.05):
                return 1.9
            elif (cal >= -0.2 and cal < -0.1):
                return 2.0
            elif (cal >= -0.3 and cal < -0.2):
                return 2.1
            elif (cal >= -0.4 and cal < -0.3):
                return 2.2
            elif (cal < -0.4):
                return 2.3
