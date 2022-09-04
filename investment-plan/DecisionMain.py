import pandas as pd
import argparse
from FundDataCrawler import FundDataCrawler
from FundDecision import FundDecision

fundDataCrawler = FundDataCrawler()
fundDecision = FundDecision()
frequence = 7
wave = 5
days = 250
balance = 10

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--code", help="fund code")
parser.add_argument("-d", "--date", help="date")
parser.add_argument("-f", "--frequence", help="frequence")
parser.add_argument("-i", "--invest", help="invest")
parser.add_argument("-b", "--balance", help="balance")
parser.add_argument("-w", "--worth", help="worth")
parser.add_argument("-y", "--yields", help="yields")
parser.add_argument("-m", "--mean", help="mean")
parser.add_argument("-v", "--wave", help="wave")
parser.add_argument("-t", "--type", help="type")

# 解析参数
args = parser.parse_args()
code = args.code
date = args.date
frequence = int(args.frequence)
invest_money = int(args.invest)
balance = int(args.balance)
current_worth = float(args.worth)
current_yield = float(args.yields)
mean_days = int(args.mean)
wave_days = int(args.wave)
type = int(args.type)

fundData = fundDataCrawler.getFund(code)
all, rate, price, mean, wave, last_yield = fundDecision.calculate_invest(fundData, date, frequence,
 invest_money, balance, current_worth, current_yield, mean_days, wave_days, type)
print(all, rate, price, mean, wave, last_yield)