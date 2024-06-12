# SPX_Analysis
 tinkering with datasets, displaying charts, simulations, indicators, etc

data and pipeline set up for real-time disgestion

created for personal entertainment and for helping with my personal investing/trading

SPX's chart has a monte carlo simulation that is currently commented out but the histogram shows the distribution based on the natural log of historical returns based on daily closing prices for all historically available data on yahoo finance. When ran, the data will be uploaded from the latest date and charted with some key metrics.

The other chart is for the VIX, an industry standard volatility reading based on the options chains of the SPX index. It too refreshes but the there is no simulation, instead it shows the current reading in the grand scheme of constantly-refreshing dataset (daily closing). It presents a normal distribution to help with understanding the current level vs on a gaussian, albeit positively skewed.

![VIX_Distribution](https://user-images.githubusercontent.com/7178449/235383039-05324519-150f-41ee-ac8b-cd5afb84b081.png)

 Franklin D. Roosevelt: "Nothing to Fear but Fear Itself"

The below is the import and plot jupyter note, simple yet elegant. It plots 4 types of data:
- the expected 10 year inflation
- the expected 10 year real interest rate
- the total of the above 2 rates
- and the current fed funds rate

what this shows is the Fed wants to close the gap between the 10 year projected inflation and the real rate, potentially keeping rates higher to allow rate rate to shoot past inflation expectations to bring down current economic activities shaped around expectations of real interest rate returns. A lot is happening in the world at the moment, and the risk free has already closed a lot ground with the effr.

![interest_rate_dynamics](https://github.com/johnluo92/SPX_Analysis/assets/7178449/9e8005d1-dec5-47b7-ab18-a2dbd9a7ad92)
