
def getrsi(df):
  period = 14                           #Standard period 14
  dif = df['Close'].diff()              #Difference in Closing prices
  dif.dropna(inplace=True)              #Remove Nan from Data frame
  
  gain = dif.copy()
  loss = dif.copy()
  
  gain[gain<0] = 0
  loss[loss>=0] = 0
  
  #Todo Add variable for period
  avg_gain = gain.rolling(period).mean()
  avg_loss = loss.rolling(period).mean().abs()
  
  rsi = 100*avg_gain/(avg_gain+avg_loss)
  
  return rsi


#Note: The first n-1 period of Data will be Nan because of Moving average
  

#TODO check if ewn(Exp Weighted MA) is better than Simple moving average


#----------------------------Sample Usage--------------------------------------
# import matplotlib.pyplot as plt
# import pandas as pd


# path = #local dir
# df = pd.read_csv(path + f'data/{name}-USD.csv', index_col = 'Date', parse_dates=True)

# rsi = getrsi(df)

# ax1 = plt.subplot2grid((10,1),(0,0), rowspan=4, colspan=1)
# ax2 = plt.subplot2grid((10,1),(5,0), rowspan=4, colspan=1)

# ax1.plot(df["Close"],linewidth=2)
# ax1.set_title('BTC Close price')

# ax2.plot(rsi, color='violet',linewidth=2)
# ax2.axhline(33.33, linestyle = '--', linewidth=1.5, color='green') #Oversold "BUY"
# ax2.axhline(66.66, linestyle = '--', linewidth=1.5, color='red')   #OverBought "SELL"
# ax2.set_title('Relative Strength Index')
# plt.show()