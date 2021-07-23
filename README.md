# Stock Market Prediction

![Capture](https://user-images.githubusercontent.com/75405590/120405085-015c2180-c305-11eb-9193-89674861be1d.JPG)

![Capture](https://user-images.githubusercontent.com/72773479/126729318-a823d57c-8348-4182-a3d0-5b97f45e3ab0.png)

David Alberghini, Mark Blankenship, John Mowry, & Bryce Wilkinson

## Project overview
For this project, we used stock market data from Yahoo Finance and then used K-means Clustering Algorithm to detech similar companies based on their movements in the stock market. It mainly deals with developing a pipeline that normalises the data and then run the algorithm to produce the lables that assign the companies to different clusters. Users can search historical relationship between 50 large cap stocks over the last 10 years. These stocks are clustered using KMeans clustering. We used 10 clusters because our dataset was limited to 50 large cap stocks. Historical prices are available between May 2011 and May 2021 currently. Companies that have merged or have been added into the index made it challenges to obtain more company historical price data. This could be expanded into the future which would be very helpful in selecting a truly diverse portfolio. Stock price data had to be normalized to ensure changes in price reflected evenly regardless of an underlying stocks price. In addition, 10 clusters returned inertia 22.02 which is rather high, ideally a larger dataset would allow more clusters and a lower inertia. We did use 30 clusters to challenge our initial thesis which was more clusters would lead to more individual relationships. This proved true as we had 50 stocks and 30 clusters with inertia 8.14. In the end, we felt the user would be better suited with 10 clusters. Look at for future updates as we look to get more historical large cap stock data for our dataset. 

## Use case:
Potential investors can use unsupervised learning techniques on 50 large cap stocks within the last 10 years. Over time stocks tend to rise and outpace inflation. Diversification is a key driver of constructing portfolios but true diversification requires a lot of capital, both financially and mentally. The idea is to pick stocks that behave differently over time. The stock market is often times momentum driven, it's always driven by future cash flows of the company, but markets are correlated. Often times markets will overheat or become spooked by some external shock. In the last 70 years, the S&P 500 has experienced 36 double-digit drawdowns. That's one every other year. A company could report great earnings, announce profit margins increased by hundreds of bps (basis points), and forecast tremendous growth in coming quarters. Yet it too would succumb to a market selloff. But this company may not fall as much as their peers. They may rebound faster as the market begins to bottom. A bottom is a point in time when a stocks price begins to trend upwards after trending downwards. This company would likely get taken down with the market, albeit temporarily before investors realize the discount they know have on a company growing profits, margins and future cash flows. So we need companies that historically aren't as correlated to each other.  

![non_normalized_data](https://user-images.githubusercontent.com/72773479/126731939-77b3898c-9c86-4645-94ec-7a5198028aa6.jpg)

![normalized_data](https://user-images.githubusercontent.com/72773479/126732075-7c780c20-2bdd-4674-990e-a3eae86c4a72.jpg)

![10_clusters](https://user-images.githubusercontent.com/72773479/126732095-1d235c7e-c2ea-4a0b-bbcf-f7bc13ec3e63.jpg)


## Data Analytic Tools Used
  * Python
  * Pandas
  * Matplotlib
  * Jupyter Notebook
  * HTML/CSS
  * Github

