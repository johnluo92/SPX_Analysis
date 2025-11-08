import pandas as pd, numpy as np, matplotlib.pyplot as plt, matplotlib.dates as mpl_dates
import sys, os, math, dateutil.parser, datetime, sys, pandas_market_calendars as mcal

from statsmodels.stats.weightstats import DescrStatsW
from datetime import date

from scipy import stats
import yfinance as yf
import seaborn as sns

yf.pdr_override()
import pandas_datareader.data as pdr

def constructDF():
    vixDf = pd.read_csv('VIX.csv')
    vixDf.index = pd.to_datetime(vixDf['DATE'], format='mixed')
    vixDf.drop(columns=['DATE','OPEN','HIGH','LOW'], inplace=True)
    return vixDf

def update_data():
    nyse = mcal.get_calendar('NYSE')
    today = date.today()
    vixDf = constructDF()
    last_saved_date = vixDf.index[-1]
    last_date_forward = (last_saved_date + datetime.timedelta(days=1)).strftime("%Y-%m-%d")
    last_traded =  pd.to_datetime(today) - pd.tseries.offsets.CustomBusinessDay(1, holidays = nyse.holidays().holidays)
    
    if last_saved_date != last_traded:
        with open('VIX.csv', 'a+') as file:
            file.write('\n')
        print(last_saved_date, last_traded)
        vix_import_df = pdr.get_data_yahoo('^VIX', last_date_forward)
        vix_import_df = vix_import_df.reindex(columns=['Open', 'High', 'Low', 'Close'])
        vix_import_df = vix_import_df.round(2)
        vix_import_df.to_csv('VIX.csv', mode='a', header=False, encoding=None)
    else:
        print('data up to date')
    return vixDf

# ----------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    
    def signaturebar(fig,text,fontsize=10,pad=5,xpos=20,ypos=7.5,
                 rect_kw = {"facecolor":"grey", "edgecolor":None},
                 text_kw = {"color":"w"}):
        w,h = fig.get_size_inches()
        height = ((fontsize+2*pad)/72.)/h
        rect = plt.Rectangle((0,0),1,height, transform=fig.transFigure, clip_on=False,**rect_kw)
        fig.axes[0].add_patch(rect)
        fig.text(xpos/72./h, ypos/72./h, text,fontsize=fontsize,**text_kw)
        fig.subplots_adjust(bottom=fig.subplotpars.bottom+height)
    
    def plot_price_paths_whs(vixDf):
        
        mu = np.average(vixDf['CLOSE'])  # mean of distribution
        last_reading = vixDf['CLOSE'][-1]
#         sigma = np.std(vixDf['CLOSE'])  # standard deviation of distribution
        num_bins = 500
        
        _, sigma = stats.norm.fit(vixDf['CLOSE'])
        fig, ax = plt.subplots()
        
        sns.histplot(vixDf['CLOSE'], bins = 500,line_kws={'ls': ':', 'lw':2},stat='density')
        sns.kdeplot(vixDf['CLOSE'], color='crimson')

        ax.set_xlabel('VIX Level')
        ax.set_ylabel('Probability density')
        ax.set_title(f'VIX Positively Skewed Distribution: $\mu={round(mu, 2)},\ \sigma={round(sigma, 2)}$')
        
        fig.tight_layout() # Tweak spacing to prevent clipping of ylabel
        #1 std percentiles: 16 -> 84 
        #2 std percentiles: 2 --> 98
        ax.axvline(np.percentile(vixDf,2), color='g', linestyle='dashed', linewidth=2, label = '2nd-98th percentile')
        ax.axvline(np.percentile(vixDf,98), color='g', linestyle='dashed', linewidth=2)
        ax.axvline(mu, color='b', linestyle='dashed', linewidth=1, label = f'Mean: {round(mu, 2)}')
        ax.axvline(last_reading, color='y', linestyle='dashed', linewidth=2, label = f'Current: {last_reading}')
        plt.legend(loc ="upper right")
        signaturebar(fig,"The Horn of Helm Hammerhand will sound in the deep, one last time!")

        plt.show()
        #legend
        
#         plt.show()
    # ------------ method for plotting historical + price paths ------------
    
    vixDf = update_data()
    print(vixDf.tail(5))
    plot_price_paths_whs(vixDf)