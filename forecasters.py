import itertools
from statsmodels.tsa.arima.model import ARIMA

class BaseForecaster:
    def fit(self, data):
        raise NotImplementedError
    def predict(self, steps):
        raise NotImplementedError

class ARIMAForecaster(BaseForecaster):
    def __init__(self):
        self.model_fit = None
        self.best_order = None
    def fit(self, data):
        p = d = q = range(0, 3)
        pdq = list(itertools.product(p, d, q))
        best_aic = float('inf')
        best_order = None
        for order in pdq:
            if order == (0, 0, 0):
                continue
            try:
                model = ARIMA(data, order=order)
                model_fit = model.fit()
                if model_fit.aic < best_aic:
                    best_aic = model_fit.aic
                    best_order = order
            except Exception:
                continue
        self.best_order = best_order
        print(f'ARIMA 최적 파라미터: {best_order}')
        self.model_fit = ARIMA(data, order=best_order).fit()
    def predict(self, steps):
        return self.model_fit.forecast(steps=steps)

class ProphetForecaster(BaseForecaster):
    def __init__(self):
        try:
            from prophet import Prophet
        except ImportError:
            raise ImportError('prophet 패키지가 필요합니다. 설치: pip install prophet')
        self.Prophet = Prophet
        self.model = None
        self.train_df = None
    def fit(self, data):
        df = data.reset_index()
        # 컬럼명을 동적으로 'ds', 'y'로 변경
        df = df.rename(columns={df.columns[0]: 'ds', df.columns[1]: 'y'})
        df = df[['ds', 'y']]
        self.train_df = df
        self.model = self.Prophet()
        self.model.fit(df)
    def predict(self, steps):
        import pandas as pd
        future = self.model.make_future_dataframe(periods=steps, freq='B')
        forecast = self.model.predict(future)
        return forecast.tail(steps).set_index('ds')['yhat']

class LSTMForecaster(BaseForecaster):
    def __init__(self, look_back=60, epochs=80, lstm_units=64, dropout_rate=0.1, batch_size=16, teacher_forcing_ratio=0):
        try:
            import tensorflow as tf
            from tensorflow import keras
        except ImportError:
            raise ImportError('tensorflow 패키지가 필요합니다. 설치: pip install tensorflow')
        from sklearn.preprocessing import MinMaxScaler
        self.tf = tf
        self.keras = keras
        self.MinMaxScaler = MinMaxScaler
        self.model = None
        self.scaler = None
        self.train_data = None
        self.look_back = look_back
        self.epochs = epochs
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.batch_size = batch_size
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.last_real_value = None  # for inverse differencing

    def fit(self, data):
        import numpy as np
        # 1차 차분
        diff_data = data.diff().dropna()
        self.last_real_value = data.iloc[-1]  # 예측 복원용
        diff_data = diff_data.values.reshape(-1, 1)
        scaler = self.MinMaxScaler()
        scaled = scaler.fit_transform(diff_data)
        X, y = [], []
        look_back = self.look_back
        for i in range(len(scaled) - look_back):
            X.append(scaled[i:i+look_back, 0])
            y.append(scaled[i+look_back, 0])
        X, y = np.array(X), np.array(y)
        X = X.reshape((X.shape[0], X.shape[1], 1))
        model = self.keras.models.Sequential([
            self.keras.layers.LSTM(self.lstm_units, return_sequences=True, input_shape=(look_back, 1)),
            self.keras.layers.Dropout(self.dropout_rate),
            self.keras.layers.LSTM(self.lstm_units, return_sequences=False),
            self.keras.layers.Dropout(self.dropout_rate),
            self.keras.layers.Dense(32, activation='relu'),
            self.keras.layers.Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        model.fit(X, y, epochs=self.epochs, batch_size=self.batch_size, verbose=0)
        self.model = model
        self.scaler = scaler
        self.train_data = scaled

    def predict(self, steps):
        import numpy as np
        last_seq = self.train_data[-self.look_back:].reshape(1, self.look_back, 1)
        preds = []
        for i in range(steps):
            pred = self.model.predict(last_seq, verbose=0)[0, 0]
            preds.append(pred)
            # teacher forcing: 실제값 일부 혼합
            if self.teacher_forcing_ratio > 0 and i < self.look_back:
                n_real = int(self.look_back * self.teacher_forcing_ratio)
                n_pred = self.look_back - n_real
                real_part = self.train_data[-n_real:].flatten() if n_real > 0 else np.array([])
                # pred_part 길이 보정
                if n_pred > 0:
                    if len(preds) >= n_pred:
                        pred_part = np.array(preds[-n_pred:])
                    else:
                        pad = n_pred - len(preds)
                        # 부족한 부분은 real_part에서 앞쪽 일부로 채움, 그래도 부족하면 0으로 패딩
                        if pad > 0:
                            pad_arr = np.zeros(pad)
                            pred_part = np.concatenate([
                                real_part[:pad] if len(real_part) >= pad else pad_arr,
                                np.array(preds)
                            ])
                        else:
                            pred_part = np.array(preds)
                    new_seq = np.concatenate([real_part, pred_part])
                else:
                    new_seq = real_part
                # 길이 보정: 항상 look_back
                if len(new_seq) > self.look_back:
                    new_seq = new_seq[-self.look_back:]
                elif len(new_seq) < self.look_back:
                    pad = self.look_back - len(new_seq)
                    pad_arr = np.zeros(pad)
                    new_seq = np.concatenate([pad_arr, new_seq])
                last_seq = new_seq.reshape(1, self.look_back, 1)
            else:
                last_seq = np.append(last_seq[:, 1:, :], [[[pred]]], axis=1)
        # 역정규화 및 누적합(역차분) 복원
        preds = self.scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()
        # 누적합: 마지막 실제값부터 예측 변화량을 더함
        restored = []
        last = self.last_real_value
        for diff in preds:
            val = last + diff
            restored.append(val)
            last = val
        return np.array(restored).flatten()
