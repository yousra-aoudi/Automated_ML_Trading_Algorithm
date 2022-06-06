# import Automated_trading_strategy_v1
import tpqoa
import pickle
import pandas as pd
import numpy as np
# Online Algorithm
algorithm = pickle.load(open('/algorithm.pkl', 'rb' ))
algorithm['model']


class MLTrader(tpqoa.tpqoa):
    def __init__(self, config_file, algorithm):
        super(MLTrader, self).__init__(config_file)
        # The trained AdaBoost model object and the normalization parameters.
        self.model = algorithm['model']
        self.mu = algorithm['mu']
        self.std = algorithm['std']
        # The number of units traded.
        self.units = 100000
        # The initial, neutral position.
        self.position = 0
        # The bar length on which the algorithm is implemented.
        self.bar = '5s'
        # The length of the window for selected features.
        self.window = 2
        # The number of lags (must be in line with algorithm training).
        self.lags = 6
        self.min_length = self.lags + self.window + 1
        self.features = ['return', 'sma', 'min', 'max', 'vol', 'mom']
        self.raw_data = pd.DataFrame()

    # The method that generates the lagged features data.
    def prepare_features(self):
        self.data['return'] = np.log(self.data['mid'] / self.data['mid'].shift(1))
        self.data['sma'] = self.data['mid'].rolling(self.window).mean()
        self.data['min'] = self.data['mid'].rolling(self.window).min()
        self.data['mom'] = np.sign(self.data['return'].rolling(self.window).mean())
        self.data['max'] = self.data['mid'].rolling(self.window).max()
        self.data['vol'] = self.data['return'].rolling(self.window).std()
        self.data.dropna(inplace=True)
        self.data[self.features] -= self.mu
        self.data[self.features] /= self.std
        self.cols = []
        for f in self.features:
            for lag in range(1, self.lags + 1):
                col = f'{f}_lag_{lag}'
                self.data[col] = self.data[f].shift(lag)
                self.cols.append(col)

    # The redefined method that embodies the trading logic.
    def on_success(self, time, bid, ask):
        df = pd.DataFrame({'bid': float(bid), 'ask': float(ask)}, index=[pd.Timestamp(time).tz_localize(None)])
        self.raw_data = self.raw_data.append(df)
        self.data = self.raw_data.resample(self.bar, label='right').last().ffill()
        self.data = self.data.iloc[:-1]
        if len(self.data) > self.min_length:
            self.min_length +=1
            self.data['mid'] = (self.data['bid'] + self.data['ask']) / 2
            self.prepare_features()
            features = self.data[self.cols].iloc[-1].values.reshape(1, -1)
            signal = self.model.predict(features)[0]
            print(f'NEW SIGNAL: {signal}', end='\r')
            if self.position in [0, -1] and signal == 1: # Check for a long signal and long trade.
                print('*** GOING LONG ***')
                self.create_order(self.stream_instrument, units=(1 - self.position) * self.units)
                self.position = 1
            elif self.position in [0, 1] and signal == -1: # Check for a short signal and short trade.
                print('*** GOING SHORT ***')
                self.create_order(self.stream_instrument, units=-(1 + self.position) * self.units)
                self.position = -1


if __name__ == '__main__':
    # Instantiates the trading object.
    mlt = MLTrader('api.cfg', algorithm)
    # Starts the streaming, data processing, and trading.
    instrument = 'EUR_USD'
    mlt.stream_data(instrument, stop=500)
    # Closes out the final open position.
    print('*** CLOSING OUT ***')
    mlt.create_order(mlt.stream_instrument, units=-mlt.position * mlt.units)

