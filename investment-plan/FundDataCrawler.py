import requests
import time
import execjs
import pandas as pd
import re

folderCode = './investment-plan/data/'

class FundDataCrawler:
    def getUrl(self, fundCode):
        head = 'http://fund.eastmoney.com/pingzhongdata/'
        tail = '.js?v=' + time.strftime("%Y%m%d%H%M%S", time.localtime())
        return head+fundCode+tail

    def getTodayFund(self, fundCode):
        content = requests.get("https://fundgz.1234567.com.cn/js/" + fundCode + ".js?rt=" + str(int(time.time())))
        today_worth = re.findall('"gsz":"(.+?)","gszzl"', content.text)[0]
        return today_worth

    def getFund(self, fundCode):
        content = requests.get(self.getUrl(fundCode))
        jsContent = execjs.compile(content.text)
        fundAll = jsContent.eval('Data_netWorthTrend')
        fundDate = []
        fundWorth = []
        fundTrend= []
        for dayFund in fundAll:
            fundDate.append(time.strftime("%Y-%m-%d", time.localtime(dayFund['x'] / 1000)))
            fundWorth.append(dayFund['y'])
            fundTrend.append(dayFund['equityReturn'])

        dataframe = pd.DataFrame({'日期':fundDate,'收盘价':fundWorth,'涨跌幅':fundTrend})
        fileFund = folderCode + fundCode + ".csv"
        dataframe.to_csv(fileFund,index=False,sep=',')

        data=pd.read_csv(fileFund)
        '''
        数据预处理
        1、将日期转换为时间序列并设置为索引
        2、将数据按时间序列升序排序
        3、删除缺失值
        4、将涨跌幅的单位转换为小数
        '''
        data['日期']=pd.to_datetime(data['日期'])
        data=data.set_index('日期')
        data=data.sort_index()
        data.drop(data[data['涨跌幅']=="None"].index,axis=0,inplace=True)
        data['涨跌幅']=data['涨跌幅'].astype('float')
        data['涨跌幅']/=100
        return data