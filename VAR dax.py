# -*- coding: utf-8 -*-
"""
Created on Jul 2020

@author: Vanina Dominguez
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import datetime
from scipy import stats
import yfinance as yf
import math

#Importing Information
End = datetime.datetime.now()
Start = End - datetime.timedelta(365*5)#Quantity of years to extend backtest
Ticker =("^GDAXI")#securitie or index 
datayf = yf.download(Ticker, start=Start, end=End)
l=len(datayf['Close']) #lenght data

#Initial Parameters
p= 0.01 #percentile
Z_99 = stats.norm.ppf(p) #Confidence level
t= 1 #time period in days of stock market calendar
W=252 #Window for rolling by stock market calendar
y= int(252/t)#periods in a year by stock market calendar

#Returns
ret = (datayf['Close']/datayf['Close'].shift(t))-1 #Returns Total Serie
ret_y = ret[:-1][-y:] #Returns last year

"Value at Risk Single Day"

#Last Price t-1 VaR Parameters
mean = np.mean(ret_y)
std = np.std(ret_y, ddof=0)
price = datayf.iloc[-2]['Close']
print(mean, std, Z_99, price)

#Returns Distribution Securitie
fig, ax = plt.subplots(figsize=(15, 10))
plt.hist(ret.dropna(), bins=40 )
plt.xlabel('Returns')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()
named="Returns Distribution "+str(Ticker)+".png"
fig.savefig(named)

#Parametric VaR
ParamVAR = price*Z_99*std

#Historical Simulation VaR 
HistVAR = price*np.percentile(ret_y.dropna(), p*100)
print('Parametric VAR is {0:.3f} and Historical VAR is {1:.3f} '.format(ParamVAR, HistVAR))

#Monte Carlo VaR
np.random.seed(42)
n_sims = 1000000
sim_returns = np.random.normal(mean, std, n_sims)
SimVAR = price*np.percentile(sim_returns,1-p)
print('Simulated VAR is ', SimVAR)


"Value at Risk Backtesting"


#variables with moving W window
ret_1= ret[W:l] #Real Losses in T for backtesting without the first W days.
meanr=ret.rolling(window=W).mean().shift(1)[W:l] #mean t-1- Montecarlo
stdr =ret.rolling(window=W).std(ddof=0).shift(1)[W:l] #std t-1 - All VaR
pricer=datayf['Close'].shift(1)[W:l] #Price T-1 - Al VaR
qtl=ret.rolling(window=W).quantile(p).shift(1)[W:l]#1% t-1 for Historical Simulation VaR

#Merge Dataframe with variables
df= pd.merge(ret,stdr,left_index=True, left_on= ret.index, right_on=stdr.index).iloc[:,1:3]
df= pd.merge(df,pricer,left_index=True, left_on= df.index, right_on=pricer.index).iloc[:,1:4]
df=df.rename(columns={'Close_x':'Real Return', 'Close_y':'Std', 'Close': 'Pricet-1'})
df= pd.merge(df,qtl,left_index=True, left_on= df.index, right_on=qtl.index).iloc[:,1:5]
df=df.rename(columns={'Close': 'VarHist'})

#Create columns with VaR
df['VarHist']= df['Pricet-1']*df['VarHist']* math.sqrt(t)#Historical Simulation VaR Column
df['VarPAram']=df['Pricet-1']*df['Std']*Z_99 * math.sqrt(t)#Parametric VaR Column
df= pd.merge(df,meanr,left_index=True, left_on= df.index, right_on=meanr.index).iloc[:,1:7]
df=df.rename(columns={'Close': 'Mean'})
df['VarMC']=0 #Monte Carlo VaR Column
df= df[1:]

#Loop for Monte Carlo Simulation 

h=0

for h in range(len(df.index)):
    meanh=df.iloc[h]['Mean']
    stdh= df.iloc[h]['Std']
    sim_returnsh = np.random.normal(meanh, stdh, n_sims)
    SimVARh = np.percentile(sim_returnsh, 1-p)* math.sqrt(t)
    df.iloc[0+h:1+h,6:7] = SimVARh 
    continue 

#Multiply by price Monte Carlo 
df['VarMC']= df['VarMC']*df['Pricet-1']

#Create column Backtest
df['Backtest']=df['Real Return']*df['Pricet-1']
   
"Plot"
name= "VaR Comparison "+str(int((1-p)*100))+"% "+ str(Ticker)
fig, ax = plt.subplots(figsize=(20, 10))
ax.plot( df.index, df['Backtest'] )
ax.plot( df.index, df['VarPAram'])
ax.plot( df.index, df['VarHist'])
ax.plot( df.index, df['VarMC'])
legend = ax.legend(loc='best', shadow=True, fontsize='large')
ax.set(xlabel='time (s)', ylabel='VAR', 
       title=name)
ax.grid()
plt.show()
fig.savefig(name +".png")







