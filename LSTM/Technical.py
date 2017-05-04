# coding: utf-8

# In[4]:

import numpy as np
import pandas as pd
from pandas import DataFrame



# In[7]:

def moving_average(values, period):
    """
    移動平均を計算するのです。
    * values: 調整後終値を指定するのです。
    * period: 期間なのです。
    """
    return DataFrame(values).rolling(period).mean()


# In[8]:

if __name__ == '__main__':
    print(moving_average(adj_end, 25))
    print(moving_average(adj_end, 75))


# In[9]:

def moving_average_deviation_rate(values, period):
    """
    移動平均乖離率を計算するのです。
    * values: 調整後終値を指定するのです。
    * period: 期間なのです。
    """
    _values = DataFrame(values)
    ma = moving_average(_values, period)
    return (_values - ma) / ma


# In[10]:

if __name__ == '__main__':
    print(moving_average_deviation_rate(adj_end, 5))
    print(moving_average_deviation_rate(adj_end, 25))


# In[11]:

def macd(values, short_period, long_period, signal_period):
    """
    MACD とその signal を計算するのです。
    * values: 調整後終値を指定するのです。
    * short_period: 短期の期間なのです。
    * long_period: 長期の期間なのです。
    * signal_period: signal の期間なのです。
    * return: MACD と MACD Signal を返すのです。
    """
    _values = DataFrame(values)
    shorts = _values.ewm(span=short_period).mean()
    longs = _values.ewm(span=long_period).mean()
    _macd = shorts - longs
    return _macd, _macd.ewm(span=signal_period).mean()


# In[12]:

if __name__ == '__main__':
    print(macd(adj_end, 12, 26, 9))


# In[13]:

def momentum(values, period):
    """
    モメンタムを計算するのです。
    * values: 調整後終値を指定するのです。
    * period: 期間なのです。
    * return: Momentum を返すのです。
    """
    _values = DataFrame(values)
    pasts = _values.shift(period)
    return (_values - pasts) / period


# In[14]:

if __name__ == '__main__':
    print(momentum(adj_end, 9))


# In[15]:

def roc(values, period):
    """
    ROC を計算するのです。
    * values: 調整後終値を指定するのです。
    * period: 期間なのです。
    * return: 終値ベースの ROC を返すのです。
    """
    _values = DataFrame(values)
    pasts = _values.shift(period)
    return (_values - pasts) / _values


# In[16]:

if __name__ == '__main__':
    print(roc(adj_end, 12))


# In[17]:

def rsi(values, period):
    """
    Wilder の RSI を計算するのです。
    * values: 調整後終値を指定するのです。
    * period: 期間なのです。
    * return: Wilder の RSI の値なのです。
    """
    _values = DataFrame(values)
    # 前日との差
    _diff = _values.diff(1)
    # 上がったやつ
    _posi = _diff.clip_lower(0).ewm(alpha=1/period).mean()
    # 下がったやつ
    _nega = _diff.clip_upper(0).ewm(alpha=1/period).mean()
    return _posi / (_posi - _nega)


# In[18]:

if __name__ == '__main__':
    print(rsi(adj_end, 14))


# In[19]:

def stochastic_K(values_end, values_high, values_low, period):
    """
    ストキャスティクス の %K を計算するのです。
    * values_end: 終値を指定するのです。
    * values_high: 高値を指定するのです。
    * values_low: 安値を指定するのです。
    * period: 期間なのです。
    * return: %K の値なのです。
    """
    """
    %K＝{ （C－L9）÷（H9－L9） }×100％
    C：当日終値
    L9：過去x日間の最安値。xとしては、14, 9, 5 などが使用されることが多い。
    H9：過去x日間の最高値
    """
    end = DataFrame(values_end)
    high = DataFrame(values_high)
    low = DataFrame(values_low)
    
    hline = high.rolling(period).max()
    lline = low.rolling(period).min()
    return (end - lline) / (hline - lline)


def stochastic_D(values_end, values_high, values_low, period_K, period):
    """
    ストキャスティクス の %D を計算するのです。
    * values_end: 終値を指定するのです。
    * values_high: 高値を指定するのです。
    * values_low: 安値を指定するのです。
    * period_K: %K の期間なのです。
    * period: 期間なのです。
    * return: %D の値なのです。
    """
    """
    %D＝（H3÷L3）×100％
    H3：（C－L9）のy日間合計。（C－L9）の単純移動平均。yとしては3が使われることが多い。
    L3：（H9－L9）のy日間合計。（H9－L9）の単純移動平均。
    """
    end = DataFrame(values_end)
    high = DataFrame(values_high)
    low = DataFrame(values_low)
    
    hline = high.rolling(period_K).max()
    lline = low.rolling(period_K).min()
    
    sumlow = (end - lline).rolling(period).sum()
    sumhigh = (hline - lline).rolling(period).sum()
    
    return sumlow / sumhigh

def stochastic_slowD(values_end, values_high, values_low, period_K, period_D, period):
    d = stochastic_D(values_end, values_high, values_low, period_K, period_D)
    return d.rolling(period).mean()


# In[21]:

if __name__ == '__main__':
    print(stochastic_K(end, high, low, 5))
    print(stochastic_D(end, high, low, 5, 3))

    print(stochastic_D(end, high, low, 15, 3))
    print(stochastic_slowD(end, high, low, 15, 3, 3))

