# 삼성전자 주가 예측 (ARIMA)
# 필요한 라이브러리 설치: yfinance, pandas, matplotlib, statsmodels

import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
from forecasters import ARIMAForecaster, ProphetForecaster, LSTMForecaster
import numpy as np
import plotly.graph_objects as go
warnings.filterwarnings('ignore')

def main(model_name='arima'):
    # 데이터 준비
    ticker = '005930.KS'
    today = datetime.today().strftime('%Y-%m-%d')
    data = yf.download(ticker, start='2015-01-01', end=today)
    data_close = data['Close']
    # 시각화
    plt.figure(figsize=(12,6))
    plt.plot(data_close)
    plt.title('Samsung Electronics (005930.KS) Closing Price')
    plt.xlabel('Date')
    plt.ylabel('Close Price (KRW)')
    plt.show()
    # 모델 선택 및 예측
    if model_name == 'arima':
        forecaster = ARIMAForecaster()
    elif model_name == 'prophet':
        forecaster = ProphetForecaster()
    elif model_name == 'lstm':
        forecaster = LSTMForecaster()
    else:
        raise ValueError('지원하지 않는 모델입니다.')
    forecaster.fit(data_close)
    forecast = forecaster.predict(steps=30)
    # 예측 결과 시각화 (예측구간 확대)
    if model_name == 'prophet':
        forecast_index = forecast.index
    else:
        forecast_index = pd.bdate_range(data_close.index[-1] + pd.Timedelta(days=1), periods=30)
    zoom_days = 90  # 최근 90일 + 예측 30일
    plot_start = max(0, len(data_close) - zoom_days)
    # 예측값을 float으로 변환, NaN/inf는 평균값으로 대체
    forecast = np.array(forecast, dtype=np.float64)
    if np.any(np.isnan(forecast)) or np.any(np.isinf(forecast)):
        mean_val = np.nanmean(forecast[np.isfinite(forecast)])
        forecast = np.where(np.isfinite(forecast), forecast, mean_val)
    plt.figure(figsize=(12,6))
    plt.plot(data_close.iloc[plot_start:], label='Actual')
    plt.plot(forecast_index, forecast, label='Forecast')
    plt.title(f'Samsung Electronics (005930.KS) Forecast ({model_name.upper()}) - Zoomed')
    plt.xlabel('Date')
    plt.ylabel('Close Price (KRW)')
    # y축을 예측값 범위에 맞게 자동 조정
    actual_vals = data_close.iloc[plot_start:].to_numpy(dtype=np.float64).flatten()
    all_vals = np.concatenate([actual_vals, forecast])
    plt.ylim(np.min(all_vals)*0.98, np.max(all_vals)*1.02)
    plt.legend()
    plt.show()

    # 예측값 출력
    forecast_rounded = forecast.round(0).astype(int)
    forecast_df = pd.DataFrame({'Date': forecast_index, 'Predicted_Close': forecast_rounded})
    forecast_df.set_index('Date', inplace=True)
    print(forecast_df)

    # 예측값 차트로 시각화 (막대그래프, 확대)
    plt.figure(figsize=(12,6))
    plt.bar(forecast_df.index.strftime('%Y-%m-%d'), forecast_df['Predicted_Close'], color='orange')
    plt.title(f'Samsung Electronics (005930.KS) 30-Day Forecasted Close Price ({model_name.upper()})')
    plt.xlabel('Date')
    plt.ylabel('Predicted Close Price (KRW)')
    plt.ylim(min(forecast_rounded)*0.98, max(forecast_rounded)*1.02)
    plt.xticks(rotation=45, fontsize=8)
    plt.tight_layout()
    plt.show()

    # plotly로 예측 결과 라인차트 (툴팁, 값 라벨 포함)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data_close.iloc[plot_start:].index, y=actual_vals, mode='lines', name='Actual'))
    fig.add_trace(go.Scatter(x=forecast_index, y=forecast, mode='lines+markers+text', name='Forecast',
                             text=[f'{v:.0f}' for v in forecast], textposition='top center'))
    fig.update_layout(title=f'Samsung Electronics (005930.KS) Forecast ({model_name.upper()}) - Zoomed (Plotly)',
                     xaxis_title='Date', yaxis_title='Close Price (KRW)',
                     legend=dict(x=0, y=1),
                     hovermode='x unified')
    fig.show()
    
    # plotly로 예측값 막대그래프 (툴팁, 값 라벨 포함)
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(x=forecast_df.index.strftime('%Y-%m-%d'), y=forecast_df['Predicted_Close'],
                          text=forecast_df['Predicted_Close'], textposition='outside', name='Forecast'))
    fig2.update_layout(title=f'Samsung Electronics (005930.KS) 30-Day Forecasted Close Price ({model_name.upper()}) (Plotly)',
                      xaxis_title='Date', yaxis_title='Predicted Close Price (KRW)',
                      xaxis_tickangle=-45)
    fig2.show()

if __name__ == '__main__':
    print('예측 모델을 선택하세요:')
    print('1. arima (기본)')
    print('2. prophet')
    print('3. lstm')
    # 추후 다른 모델 추가 시 여기에 안내    .\\my_312_project_env\\Scripts\\activate
    model_input = input('모델명을 입력하세요 (기본: arima): ').strip().lower()
    if not model_input:
        model_input = 'arima'
    main(model_input)
