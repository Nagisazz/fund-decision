import re
import numpy as np
import pandas as pd
import csv
from keras.models import Sequential
from keras.layers import LSTM,Dense,Dropout
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
from math import sqrt
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model


folderCode = './lstm-simple/data/'
modelPath = './lstm-simple/model/modelrmse.hdf5'

plt.rcParams['font.sans-serif'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False

batch_size = 4
epochs = 100

time_step = 6  # 用多少组天数进行预测
input_size = 6  # 每组天数，亦即预测天数
look_back = time_step * input_size
showdays = 120  # 最后画图观察的天数（测试天数）

# 忽略掉最近的forget_days天数据（回退天数，用于预测的复盘）
forget_days = 3

class FundPredictor:
    def __init__(self, plateCode):
        self.plateCode = plateCode

        self.fileFundTrain = folderCode + plateCode + "/train.csv"
        self.fileFundTest = folderCode + plateCode + "/test.csv"

    def create_dataset(self, dataset):
        dataX, dataY = [], []
        # print('len of dataset: {}'.format(len(dataset)))
        for i in range(0, len(dataset) - look_back, input_size):
            x = dataset[i: i + look_back]
            dataX.append(x)
            y = dataset[i + look_back: i + look_back + input_size]
            dataY.append(y)
        return np.array(dataX), np.array(dataY)

    def build_model(self,droupout,type):
        if type == 1:
            # 新建模型
            print('new_model')
            model = Sequential()
            model.add(LSTM(units=128, return_sequences=True, input_shape=(time_step, input_size)))
            if droupout != 0:
                model.add(Dropout(droupout))
            model.add(LSTM(units=32))
            if droupout != 0: 
                model.add(Dropout(droupout))
            model.add(Dense(units=input_size))

            # model.compile(loss='mean_squared_error', optimizer='adam')
            model.compile(loss='mae', optimizer='adam')
            return model
        else:
            # 加载模型
            print('load_model')
            model = load_model(modelPath)
            return model

    def predictor(self,droupout,type,draw,rmseDec):
        x_train = []
        y_train = []
        x_validation = []
        y_validation = []
        testset = []  # 用来保存测试基金的近期净值
        # 设定随机数种子
        seed = 7
        np.random.seed(seed)

        print('start droupout:{},type:{}'.format(droupout,type))

        # 导入数据（训练集）
        with open(self.fileFundTrain) as f:
            row = csv.reader(f, delimiter=',')
            for r in row:
                dataset = []
                r = [x for x in r if x != 'None']
                # 涨跌幅是2天之间比较，数据会减少1个
                days = len(r) - 1
                # 有效天数太少，忽略
                if days <= look_back + input_size:
                    continue
                for i in range(days):
                    f1 = float(r[i])
                    f2 = float(r[i+1])
                    if f1 == 0 or f2 == 0:
                        dataset = []
                        break
                    # 把数据放大100倍，相当于以百分比为单位
                    f2 = (f2 - f1) / f1 * 100
                    # 如果涨跌幅绝对值超过15%，基金数据恐有问题，忽略该组数据
                    if f2 > 15 or f2 < -15:
                        dataset = []
                        break
                    dataset.append(f2)
                n = len(dataset)
                # 进行预测的复盘，忽略掉最近forget_days的训练数据
                n -= forget_days
                if n >= look_back + input_size:
                    # 如果数据不是input_size的整数倍，忽略掉最前面多出来的
                    m = n % input_size
                    X_1, y_1 = self.create_dataset(dataset[m:n])
                    x_train = np.append(x_train, X_1)
                    y_train = np.append(y_train, y_1)

        # 导入数据（测试集）
        with open(self.fileFundTest) as f:
            row = csv.reader(f, delimiter=',')
            # 写成了循环，但实际只有1条测试数据
            for r in row:
                dataset = []
                # 去掉记录为None的数据（当天数据缺失）
                r = [x for x in r if x != 'None']
                # 涨跌幅是2天之间比较，数据会减少1个
                days = len(r) - 1
                # 有效天数太少，忽略，注意：测试集最后会虚构一个input_size
                if days <= look_back:
                    print('only {} days data. exit.'.format(days))
                    continue
                # 只需要最后画图观察天数的数据
                if days > showdays:
                    r = r[days-showdays:]
                    days = len(r) - 1
                for i in range(days):
                    f1 = float(r[i])
                    f2 = float(r[i+1])
                    if f1 == 0 or f2 == 0:
                        print('zero value found. exit.')
                        dataset = []
                        break
                    # 把数据放大100倍，相当于以百分比为单位
                    f2 = (f2 - f1) / f1 * 100
                    # 如果涨跌幅绝对值超过15%，基金数据恐有问题，忽略该组数据
                    if f2 > 15 or f2 < -15:
                        print('{} greater then 15 percent. exit.'.format(f2))
                        dataset = []
                        break
                    testset.append(f1)
                    dataset.append(f2)
                # 保存最近一天基金净值
                f1 = float(r[days])
                testset.append(f1)
                # 测试集虚构一个input_size的数据（若有forget_days的数据，则保留）
                if forget_days < input_size:
                    for i in range(forget_days, input_size):
                        dataset.append(0)
                        testset.append(np.nan)
                else:
                    dataset = dataset[:len(dataset) - forget_days + input_size]
                    testset = testset[:len(testset) - forget_days + input_size]
                if len(dataset) >= look_back + input_size:
                    # 将testset修正为input_size整数倍加1
                    m = (len(testset) - 1) % input_size
                    testset = testset[m:]
                    m = len(dataset) % input_size
                    # 将dataset修正为input_size整数倍
                    x_validation, y_validation = self.create_dataset(dataset[m:])

        # 归一化处理
        # scaler = MinMaxScaler(feature_range=(-1, 1))
        # x_train_scaled = scaler.fit_transform(x_train.reshape(-1,1))
        # x_validation_scaled = scaler.fit_transform(x_validation.reshape(-1,1))

        # 将输入转化成[样本数，时间步长，特征数]
        x_train = x_train.reshape(-1, time_step, input_size)
        x_validation = x_validation.reshape(-1, time_step, input_size)

        # 将输出转化成[样本数，特征数]
        y_train = y_train.reshape(-1, input_size)
        y_validation = y_validation.reshape(-1, input_size)
        
        # 归一化处理
        # y_train_scaled = scaler.fit_transform(y_train)

        print('num of x_train: {}\tnum of y_train: {}'.format(
            len(x_train), len(y_train)))
        print('num of x_validation: {}\tnum of y_validation: {}'.format(
            len(x_validation), len(y_validation)))

        # 训练模型
        model = self.build_model(droupout,type)
        if type == 1:
            history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size,
                    verbose=0, validation_split=0.25, shuffle=False)

            print('train_loss最后3次：{}'.format(history.history['loss'][-3:]))
            print('val_loss最后3次：{}'.format(history.history['val_loss'][-3:]))

            # 绘制损失图
            if draw:
                plt.plot(history.history['loss'], label='train')
                plt.plot(history.history['val_loss'], label='test')
                plt.title(self.plateCode, fontsize='12')
                plt.ylabel('loss', fontsize='10')
                plt.xlabel('epoch', fontsize='10')
                plt.legend()
                plt.show()
        
        # 评估模型
        # train_score = model.evaluate(x_train, y_train, verbose=0)
        # validation_score = model.evaluate(
        #     x_validation, y_validation, verbose=0)

        # 预测
        predict_validation = model.predict(x_validation)

        # 反归一化
        # predict_validation = scaler.inverse_transform(predict_validation)

        # 将之前虚构的最后一组input_size里面的0涨跌改为NAN（不显示虚假的0）
        if forget_days < input_size:
            for i in range(forget_days, input_size):
                y_validation[-1, i] = np.nan

        # print('Train Set Score: {:.3f}'.format(train_score))
        # print('Test Set Score: {:.3f}'.format(validation_score))
        print('未来{}天实际百分比涨幅为：{}'.format(input_size, y_validation[-1]))
        print('未来{}天预测百分比涨幅为：{}'.format(input_size, predict_validation[-1]))

        inv_y = y_validation[:-6,:].reshape(1,-1)
        inv_y_predict = predict_validation[:-6,:].reshape(1,-1)

        # 进行reshape(-1, 1)是为了plt显示
        y_validation = y_validation.reshape(-1, 1)
        predict_validation = predict_validation.reshape(-1, 1)
        testset = np.array(testset).reshape(-1, 1)

        # 图表显示
        if draw:
            fig = plt.figure(figsize=(15, 6))
            plt.plot(y_validation, color='blue', label='基金每日涨幅')
            plt.plot(predict_validation, color='red', label='预测每日涨幅')
            plt.legend(loc='upper left')
            plt.title('关联组数：{}组，预测天数：{}天，回退天数：{}天'.format(
                time_step, input_size, forget_days))
            plt.show()

        # 实际净值、预测净值
        y_validation_plot = np.empty_like(testset)
        predict_validation_plot = np.empty_like(testset)
        y_validation_plot[:, :] = np.nan
        predict_validation_plot[:, :] = np.nan

        y = testset[look_back, 0]
        p = testset[look_back, 0]
        for i in range(look_back, len(testset)-1):
            y *= (1 + y_validation[i-look_back, 0] / 100)
            p *= (1 + predict_validation[i-look_back, 0] / 100)
            #print('{:.4f} {:.4f} {:.4f}'.format(testset[i+1,0], y, p))
            y_validation_plot[i+1, :] = y
            predict_validation_plot[i+1, :] = p

        print('未来{}天实际净值为：{}'.format(input_size, y_validation_plot[-6:,:].reshape(1,-1)))
        print('未来{}天预测净值为：{}'.format(input_size, predict_validation_plot[-6:,:].reshape(1,-1)))

        # 图表显示
        if draw:
            fig = plt.figure(figsize=(15, 6))
            plt.plot(y_validation_plot, color='blue', label='基金每日净值')
            plt.plot(predict_validation_plot, color='red', label='预测每日净值')
            plt.legend(loc='upper left')
            plt.title('关联组数：{}组，预测天数：{}天，回退天数：{}天'.format(
                time_step, input_size, forget_days))
            plt.show()

        #回归评价指标
        # calculate MSE 均方误差
        mse = mean_squared_error(inv_y,inv_y_predict)
        # calculate RMSE 均方根误差
        rmse = sqrt(mean_squared_error(inv_y, inv_y_predict))
        #calculate MAE 平均绝对误差
        mae = mean_absolute_error(inv_y,inv_y_predict)

        print('均方误差: %.6f' % mse)
        print('均方根误差: %.6f' % rmse)
        print('平均绝对误差: %.6f' % mae)

        
        if type == 1 and rmse < rmseDec:
            # 保存模型
            modelPathNew = modelPath.replace('rmse',str(rmse))
            print('save_model,modelPath:{}'.format(modelPathNew))
            model.save(modelPathNew)
        return rmse
