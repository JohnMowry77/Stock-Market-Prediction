import pandas_datareader.data as web
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import datetime

# define instruments to download
def init_stocks():

	companies_dict = {
	 'Amazon': 'AMZN',
	 'Apple': 'AAPL',
	 'Walgreen': 'WBA',
	 'Northrop Grumman': 'NOC',
	 'Boeing': 'BA',
	 'Lockheed Martin':'LMT',
	 'McDonalds': 'MCD',
	 'Intel': 'INTC',
	 'NextEra Energy': 'NEE',
	 'IBM': 'IBM',
	 'Texas Instruments': 'TXN',
	 'MasterCard': 'MA',
	 'Microsoft': 'MSFT',
	 'General Electric': 'GE',
	 'Alphabet Inc. (Class A)': 'GOOGL',
	 'American Express': 'AXP',
	 'Pepsi': 'PEP',
	 'Coca Cola': 'KO',
	 'Johnson & Johnson': 'JNJ',
	 'General Motors': 'GM',
	 'HCA Healthcare': 'HCA',
	 'Amgen Inc.': 'AMGN',
	 'JPMorgan Chase & Co.': 'JPM',
	 'Netflix Inc.': 'NFLX',
	 'UnitedHealth Group Inc.': 'UNH',
	 'Visa Inc.': 'V',
	 'Vulcan Materials': 'VMC',
	 'Verizon Communications': 'VZ','Lincoln National': 'LNC',
	 'Waste Management Inc.': 'WM',  
	 'Target Corp.': 'TGT',
	 'Prologis': 'PLD',
	 'Chevron Corp.': 'CVX',
	 'Pioneer Natural Resources': 'PXD',
	 'Progressive Corp.': 'PGR',
	 'Nucor Corp.': 'NUE',
	 'TJX Companies Inc.': 'TJX',
	 '3M Company': 'MMM',
	 'Medtronic plc': 'MDT',
	 'Lilly (Eli) & Co.': 'LLY',
	 'Masco Corp.': 'MAS',
	 'Kroger Co.': 'KR',
	 'AmerisourceBergen': 'ABC',
	 'Applied Materials Inc.': 'AMAT',
	 'Deere & Co.': 'DE',
	 'United Parcel Service': 'UPS',
	 'Lennar Corp.': 'LEN',
	 'Whirlpool Corp.': 'WHR',
	 'Lincoln National': 'LNC',
	 'Adobe Inc.': 'ADBE',
	 'Celanese': 'CE'
	}

	companies = sorted(companies_dict.items(), key=lambda x: x[1])

	# Define which online source to use
	data_source = 'yahoo'

	# define start and end dates
	# start_date = '2016-05-20'
	# start_date = '2018-05-20'
	start_date = startDate
	end_date = endDate
	# end_date = '2021-05-20'

	# Use pandas_datareader.data.DataReader to load the desired data list(companies_dict.values()) used for python 3 compatibility
	panel_data = web.DataReader(list(companies_dict.values()), data_source, start_date, end_date)

	# print(panel_data.axes)


	# Find Stock Open and Close Values
	stock_close = panel_data['Close']
	stock_open = panel_data['Open']


	# df=panel_data.dropna()
	# df=panel_data.isnull()

	# print(stock_close)
	# print(stock_close.iloc[0])
	# print(df)


	# Calculate daily stock movement
	stock_close = np.array(stock_close).T
	stock_open = np.array(stock_open).T

	row, col = stock_close.shape

	# create movements dataset filled with 0's
	movements = np.zeros([row, col])

	for i in range(0, row):
	 movements[i,:] = np.subtract(stock_close[i,:], stock_open[i,:])


	 # len(companies)

	# for i in range(0, len(companies)):
	# 	print('Company: {}, Change: {}'.format(companies[i][0], sum(movements[i][:])))


	# plt.figure(figsize=(18,16))
	# ax1 = plt.subplot(221)
	# plt.plot(movements[0][:])
	# plt.title(companies[0])

	# plt.subplot(222, sharey=ax1)
	# plt.plot(movements[1][:])
	# plt.title(companies[1])
	# plt.show()

	# import Normalizer
	from sklearn.preprocessing import Normalizer
	# create the Normalizer
	normalizer = Normalizer()

	new = normalizer.fit_transform(movements)

	# print(new.max())
	# print(new.min())
	# print(new.mean())


	# import machine learning libraries
	from sklearn.pipeline import make_pipeline
	from sklearn.cluster import KMeans

	# define normalizer
	normalizer = Normalizer()

	# create a K-means model with 10 clusters
	kmeans = KMeans(n_clusters=10, max_iter=1000)

	# make a pipeline chaining normalizer and kmeans
	pipeline = make_pipeline(normalizer,kmeans)

	# fit pipeline to daily stock movements
	pipeline.fit(movements)


	# To check how well the algorithm did use print(kmeans.inertia_)
	#Intertia is a score of how close each cluster is, so a lower inertia score is better. 

	# print(kmeans.inertia_)

	# predict cluster labels
	labels = pipeline.predict(movements)

	# create a DataFrame aligning labels & companies
	df = pd.DataFrame({'labels': labels, 'companies': companies})

	# display df sorted by cluster labels
	# print(df.sort_values('labels'))

	# PCA
	from sklearn.decomposition import PCA 

	# visualize the results
	reduced_data = PCA(n_components = 2).fit_transform(new)

	# run kmeans on reduced data
	kmeans = KMeans(n_clusters=10)
	kmeans.fit(reduced_data)
	labels = kmeans.predict(reduced_data)

	# create DataFrame aligning labels & companies
	df = pd.DataFrame({'labels': labels, 'companies': companies})

	# Display df sorted by cluster labels

	sorted_df=(df.sort_values('labels'))
	return sorted_df



if __name__=='__main__':
	app.run(debug=True)