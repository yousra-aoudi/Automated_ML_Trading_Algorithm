# Libraries
import tpqoa
import math
import time
import numpy as np
import pandas as pd
import datetime as dt

from matplotlib import pyplot as plt
from matplotlib.pylab import mpl
import scipy.stats as scs
import pickle

# Machine learning libraries
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier

# API
api = tpqoa.tpqoa('api.cfg')

# Vectorized Backtesting - Data collection

instrument = 'EUR_USD'

raw = api.get_history(instrument, start='2020-06-08', end='2020-06-13', granularity='M10', price='M')
raw.tail()
raw.info()

spread = 0.00012
mean = raw['c'].mean()

ptc = spread / mean
raw['c'].plot(figsize=(10, 6), legend=True)
plt.title('EUR/USD exchange rate 10-minute bars')
title = 'EUR USD exchange rate 10 minute bars.png'
plt.savefig(title)
plt.show()

data = pd.DataFrame(raw['c'])
data.columns = [instrument,]
window = 20
data['return'] = np.log(data / data.shift(1))
data['vol'] = data['return'].rolling(window).std()
data['mom'] = np.sign(data['return'].rolling(window).mean())
data['sma'] = data[instrument].rolling(window).mean()
data['min'] = data[instrument].rolling(window).min()
data['max'] = data[instrument].rolling(window).max()
data.dropna(inplace=True)

lags = 6
features = ['return', 'vol', 'mom', 'sma', 'min', 'max']
cols = []

for f in features:
    for lag in range(1, lags + 1):
        col = f'{f}_lag_{lag}'
        data[col] = data[f].shift(lag)
        cols.append(col)

data.dropna(inplace=True)
data['direction'] = np.where(data['return'] > 0, 1, -1)
data[cols].iloc[:lags, :lags]


# Machine learning
n_estimators = 15
random_state = 100
max_depth = 2
min_samples_leaf = 15
subsample = 0.33

dtc = DecisionTreeClassifier(random_state=random_state, max_depth=max_depth, min_samples_leaf=min_samples_leaf)
model = AdaBoostClassifier(base_estimator=dtc, n_estimators=n_estimators, random_state=random_state)
split = int(len(data) * 0.7)

train = data.iloc[:split].copy()
mu, std = train.mean(), train.std()
train_ = (train - mu) / std
model.fit(train_[cols], train['direction'])

accuracy_score(train['direction'], model.predict(train_[cols]))
test = data.iloc[split:].copy()
test_ = (test - mu) / std
test['position'] = model.predict(test_[cols])
accuracy_score(test['direction'], test['position'])

test['strategy'] = test['position'] * test['return']
sum(test['position'].diff() != 0)
test['strategy_tc'] = np.where(test['position'].diff() != 0, test['strategy'] - ptc, test['strategy'])
test[['return', 'strategy', 'strategy_tc']].sum().apply(np.exp)
test[['return', 'strategy', 'strategy_tc']].cumsum().apply(np.exp).plot(figsize=(10, 6))
plt.title('Gross performance of EUR/USD exchange rate and algorithmic trading strategy '
           'before and after transaction costs')
plt.savefig('Gross performance of EUR USD exchange rate and algorithmic trading strategy '
           'before and after transaction costs.png')
plt.show()

# Optimal leverage
mean = test[['return', 'strategy_tc']].mean() * len(data) * 52
print('Annualized mean returns :', mean)
var = test[['return', 'strategy_tc']].var() * len(data) * 52
print('Annualized variances :', var)
vol = var ** 0.5
print('Annualized volatilities :', var)
print('Optimal leverage according to the Kelly criterion (“full Kelly”) \n', mean / var)
print('Optimal leverage according to the Kelly criterion (“half Kelly”) \n', mean / var * 0.5)

to_plot = ['return', 'strategy_tc']

for lev in [10, 20, 30, 40, 50]:
    label = 'lstrategy_tc_%d' % lev
    test[label] = test['strategy_tc'] * lev
    to_plot.append(label)

test[to_plot].cumsum().apply(np.exp).plot(figsize=(10, 6))
plt.title('Gross performance of the algorithmic trading strategy for different leverage values')
plt.show()

# Risk Analysis

equity = 3333
risk = pd.DataFrame(test['lstrategy_tc_30'])
risk['equity'] = risk['lstrategy_tc_30'].cumsum().apply(np.exp) * equity
risk['cummax'] = risk['equity'].cummax()
risk['drawdown'] = risk['cummax'] - risk['equity']
risk['drawdown'].max()
t_max = risk['drawdown'].idxmax()
print('t_max :',t_max)

temp = risk['drawdown'][risk['drawdown'] == 0]
periods = (temp.index[1:].to_pydatetime() - temp.index[:-1].to_pydatetime())
print('timedelta values between all highs \n', periods[20:30])
t_per = periods.max()
print('The longest drawdown period in seconds :', t_per)
print('... transformed to hours :',t_per.seconds / 60 / 60)

risk[['equity', 'cummax']].plot(figsize=(10, 6))
plt.axvline(t_max, c='r', alpha=0.5)
plt.title('Maximum drawdown (vertical line) and drawdown periods (horizontal lines)')
plt.savefig('Maximum drawdown, vertical line, and drawdown periods, horizontal lines.png')
plt.show()

"""
VaR values based on the log returns of the equity position for the leveraged trading strategy over time for different
confidence levels. The time interval is fixed to the bar length of ten minutes:
"""

# Defines the percentile values to be used.
percs = [0.01, 0.1, 1., 2.5, 5.0, 10.0]

risk['return'] = np.log(risk['equity'] / risk['equity'].shift(1))

# Calculates the VaR values given the percentile values.
VaR = scs.scoreatpercentile(equity * risk['return'], percs)

# Translates the percentile values into confidence levels and the VaR values (nega‐ tive values) to positive values
# for printing.


def print_var():
    print('{} {}'.format('Confidence Level', 'Value-at-Risk'))
    print(33 * '-')
    for pair in zip(percs, VaR):
        print('{:16.2f} {:16.3f}'.format(100 - pair[0], -pair[1]))


print_var()


# Resamples the data from 10-minute to 1-hour bars.
hourly = risk.resample('1H', label='right').last()
hourly['return'] = np.log(hourly['equity'] / hourly['equity'].shift(1))

# Calculates the VaR values given the percentile values.
VaR = scs.scoreatpercentile(equity * hourly['return'], percs)
print_var()

# Persisting the Model Object
algorithm = {'model': model, 'mu': mu, 'std': std}
pickle.dump(algorithm, open('/algorithm.pkl', 'wb'))

