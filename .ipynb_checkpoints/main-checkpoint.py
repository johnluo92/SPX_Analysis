import pandas as pd, numpy as np, matplotlib.pyplot as plt, matplotlib.dates as mpl_dates
import math, dateutil.parser, datetime, sys
from statsmodels.stats.weightstats import DescrStatsW
from scipy import stats


if __name__ == "__main__":


    SP500 = pd.read_csv('SPX_Data.csv')
    SP500["Date"] = pd.to_datetime(SP500["Date"])
    SP500['Log_Returns'] = np.log(SP500['Close*']) - np.log(SP500['Close*'].shift(1))
    SP500['Month'] = pd.DatetimeIndex(SP500['Date']).month

    def add_weight(row):
        return 1.5 if row['Month'] == 9 else 1

    SP500['Weight'] = SP500.apply(add_weight, axis=1)

#     # sim_paths
#     def plot_paths_whs(SPX_historical):
            
#         sim_ending_prices = []

#     #     df = pd.DataFrame.from_dict(ohlcv_historical)
#         date = pd.to_datetime(SPX_historical['Date'])
#         close = SPX_historical['Close*']
#         plt.plot(date, close, linestyle='--', )
#         plt.title('SPX Price')
#         plt.ylabel('Close Price')
        

# #     plot_paths_whs(SP500)
    SP500.drop(columns=['Volume','Adj Close**'], inplace=True)
    SP500.dropna(inplace=True)
    
    #calculate weighted standard deviation
    # print(SP500.to_string())

    # weighted_log_r = list(SP500['Log_Returns'])
    # a = SP500.memory_usage()
    a = SP500.groupby(pd.Grouper(key="Month")).mean()
    # print(mu)
    print(SP500.dtypes)

    def price_sim_ICDF(SP500, sims, periods):
        last_close = SP500['Close*'].iat[-1]
        last_date = SP500['Date'].iat[-1]
        
        log_returns = SP500['Log_Returns']
        mu = np.average(log_returns)
        std = np.std(log_returns)
        variance = np.var(log_returns)
        drift = mu - .5 * variance

        weighted_mu = np.average(SP500['Log_Returns'], weights=SP500['Weight'])
        weighted_std= DescrStatsW(SP500['Log_Returns'], weights=SP500['Weight']).std
        mu_by_month = SP500.groupby(pd.Grouper(key="Month")).mean()

        print('non-weighted average log returns: {0:.5%}'.format(mu))
        print('non-weighted standard deviation: {0:.5%}'.format(std))
        print('\nweighted average log returns: {0:.5%}'.format(weighted_mu))
        print('weighted standard deviation: {0:.5%}'.format(weighted_std))
        
        sim_paths = [0]*sims

        for s in range(0, sims):
            
            q = np.random.rand(periods)
            Z = stats.norm.ppf(q, loc=mu, scale=std)

            daily_returns = pow(math.e, (drift + std * Z))
            price_paths = [{} for i in range(periods)]
            price_paths[0]['Date'] = last_date
            price_paths[0]['Close'] = last_close
            previous_date = price_paths[0]['Date']
            
            for t in range(1, periods):
                
                previous_date = price_paths[t-1]['Date']
#                 print(previous_date)
#                 print(type(previous_date))
                new_date = previous_date + datetime.timedelta(days=1)
#                 print(new_date)
#                 print(type(new_date))
#                 sys.exit(1)
                price_paths[t]['Date'] = new_date
                price_paths[t]['Close'] = price_paths[t-1]['Close'] * daily_returns[t]

            sim_paths[s] = price_paths

        return sim_paths
        
    periods, sims = 30, 1000
    def myfunc():
        return price_sim_ICDF(SP500, sims, periods)

    sim_paths = myfunc()
    
    def plot_price_paths_whs(sim_paths):
        
        date = pd.to_datetime(SP500['Date'])
        close = SP500['Close*']
        plt.plot(date, close, linestyle='--', )
        plt.title('SPX Price')
        plt.ylabel('Close Price')

        sim_ending_prices = []
        
        for path in sim_paths:

            df = pd.DataFrame.from_dict(path)
            date = pd.to_datetime(df['Date'])
            close = df['Close']
            plt.plot(date, close, linestyle='-', )

            sim_ending_prices.append(path[-1]['Close'])

        date_format=mpl_dates.DateFormatter("%Y-%m-%d")
        plt.gcf().autofmt_xdate() #remove if needed
        plt.gca().xaxis.set_major_formatter(date_format)
        plt.show()

        plt.hist(sim_ending_prices, bins=50, density=True)
        plt.show()
    # ------------ method for plotting historical + price paths ------------

    plot_price_paths_whs(sim_paths)