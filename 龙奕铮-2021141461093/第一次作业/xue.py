import akshare as ak
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def get_stock_data(symbol, start_date, end_date):
    data = ak.stock_zh_index_daily_em(symbol=symbol, start_date=start_date, end_date=end_date)
    return data

def calculate_pct_change(data):
    data['pct_change'] = data['close'].pct_change() * 100
    return data

def discretize_data(data, N):
    bins = np.linspace(-5, 5, N+1)
    labels = range(N)
    data['state'] = pd.cut(data['pct_change'], bins=bins, labels=labels, include_lowest=True)
    return data

def generate_transition_matrix(data, N):
    transition_matrix = np.zeros((N, N))
    for i in range(1, len(data)):
        current_state = data.iloc[i-1]['state']
        next_state = data.iloc[i]['state']
        if pd.notna(current_state) and pd.notna(next_state):
            transition_matrix[int(current_state), int(next_state)] += 1
    transition_matrix = transition_matrix / transition_matrix.sum(axis=1, keepdims=True)
    return transition_matrix

def plot_transition_matrix(transition_matrix):
    plt.figure(figsize=(8, 6))
    plt.imshow(transition_matrix, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.title("Steady State Transition Matrix After 100 Iterations")
    plt.xlabel("Next State")
    plt.ylabel("Current State")
    plt.show()

def plot_state_counts(data, bins):
    state_counts = data['state'].value_counts().sort_index()
    state_labels = [f"{(bins[i]+bins[i+1])/2}%" for i in range(len(bins)-1)]
    plt.figure(figsize=(10, 6))
    plt.plot(state_labels, state_counts.values, marker='o', linestyle='-')
    plt.title('Days Count for Each State (-5% to 5%)')
    plt.xlabel('State')
    plt.ylabel('Days Count')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.show()

def plot_stock_prices(data):
    stock_prices = data['close']
    plt.figure(figsize=(10, 6))
    plt.plot(stock_prices, color='blue')
    plt.title('Stock Price Change Over Time')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.grid(True)
    plt.show()

# 数据获取和准备
symbol = "sz000002"  # 万科A
start_date_train = "20000101"
end_date_train = "20181231"  # 修改训练集结束日期
start_date_test = "20190101"
end_date_test = "20191231"

data_train = get_stock_data(symbol, start_date_train, end_date_train)
data_test = get_stock_data(symbol, start_date_test, end_date_test)

# 计算百分比日变化
data_train = calculate_pct_change(data_train)

# 数据离散化
N = 10  # 状态数量
data_train = discretize_data(data_train, N)

# 频率统计与转移矩阵
transition_matrix = generate_transition_matrix(data_train, N)

# 绘制稳态转移矩阵
plot_transition_matrix(transition_matrix)

# 绘制每个状态的天数
bins = np.linspace(-5, 5, N+1)
plot_state_counts(data_train, bins)

# 绘制股票价格变化曲线
plot_stock_prices(data_train)


