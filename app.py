from flask import Flask, render_template, redirect, request, make_response, jsonify
# import stock_data
import pandas_datareader.data as web
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import datetime
from sklearn.decomposition import PCA 
# import Normalizer
from sklearn.preprocessing import Normalizer
# import machine learning libraries
from sklearn.pipeline import make_pipeline
from sklearn.cluster import KMeans

import trading_calendars as tc
import pytz

# Create an instance of Flask
app=Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
	return render_template("index.html")

@app.route("/Charts")
def charts():
	return render_template('Charts.html', title="page")

@app.route("/clusters")
def clusters():
	return render_template('clusters.html', title="page")


@app.route("/stocks", methods=["GET", "POST"])
def stocks():
	# sector_df=read_csv('')
	# if request.method=="POST":
	# define start and end dates
	# start_date = '2016-05-20'
	# start_date = '2018-05-20'
	# start_date = startDate
	# end_date = endDate
	# end_date = '2021-05-20'
	start_date=request.form.get("startdate")
	end_date=request.form.get("enddate")
	print(start_date)
	print(end_date)
	# def get_stocks():
	# 	sorted_df = stock_data.init_stocks()
	# 	print(sorted_df)
	# 	print('hi')
	# 	# start_date=request.form.get("startdate")
	# 	# end_date=request.form.get("enddate")
	# 	# print(start_date)
	# 	# print(end_date)
	# 	return redirect("/stocks")
	# get_stocks()

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

	xnys = tc.get_calendar("XNYS")
	date=xnys.sessions_in_range(
		# pd.Timestamp("2011-05-20", tz=pytz.UTC),
		# pd.Timestamp("2021-05-20", tz=pytz.UTC)
		pd.Timestamp(start_date, tz=pytz.UTC), 
		pd.Timestamp(end_date, tz=pytz.UTC)
	)

	movements_df= pd.DataFrame(movements, index=companies_dict.keys(), columns=date)
	movements_df.to_csv('output/movements_df.csv')
	# movements_df=movements_df.transpose()
	columns=list(movements_df.columns)
	new_columns=[]
	for each_column in columns: 
	    new_columns.append(each_column.tz_convert(None))#.replace('00:00:00+00:00', ''))
	movements_df.columns=new_columns
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


	# create the Normalizer
	normalizer = Normalizer()

	new = normalizer.fit_transform(movements)

	# print(new.max())
	# print(new.min())
	# print(new.mean())



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
	

	# visualize the results
	reduced_data = PCA(n_components = 2).fit_transform(new)

	# run kmeans on reduced data
	kmeans = KMeans(n_clusters=10)
	kmeans.fit(reduced_data)
	labels = kmeans.predict(reduced_data)

	# create DataFrame aligning labels & companies
	df = pd.DataFrame({'labels': labels, 'companies': companies})

	# Display df sorted by cluster labels

	sorted_df=df.sort_values('labels')

	sorted_df[['company_name', 'company_symbol']]=sorted_df['companies'].astype('str').str.replace('(', '').str.replace(')', '').str.replace("'", '').str.split(', ', expand=True)#[1]
	sorted_df=sorted_df.replace({'Alphabet Inc. Class A': 'Alphabet Inc. (Class A)', 
                   'Lilly Eli & Co.': 'Lilly (Eli) & Co.'})
	# combined_df=pd.merge(movements_df, sorted_df, left_index=True, right_on='company_name', how='outer')
	# combined_df=combined_df.drop(columns=['companies', 'company_name', 'company_symbol'])

	# for each_cluster in range(0, 10): 
	#     graph_df=combined_df[combined_df['labels']==each_cluster].drop(columns=['labels'])
	#     graph_df=graph_df.transpose()#.plot()
	#     # columns=graph_df.iloc[0]
	#     # graph_df.columns=columns
	#     columns=graph_df.iloc[0]
	#     graph_df.columns=columns
	#     graph_df.iloc[1:].plot(rot=90, legend=False)
	#     graph_df.plot(rot=90)
	#     plt.legend(loc='upper left')
	#     image_links= plt.savefig(f'static/charts/cluster_{each_cluster}.png')

	# combined_df=pd.merge(movements_df, sorted_df, left_on='Unnamed: 0', right_on='company_name', how='outer')
	# combined_df=combined_df.drop(columns=['companies', 'company_name', 'company_symbol'])
	# combined_df
	# for each_cluster in range(0, 10):
	#     graph_df=combined_df[combined_df['labels']==each_cluster].drop(columns=['labels'])
	#     graph_df=graph_df.transpose()#.plot()
	#     columns=graph_df.iloc[0]
	#     graph_df.columns=columns
	#     graph_df.iloc[1:].plot(rot=90, legend=False)
	# #     plt.subplots(figsize=(6, 2))
	# #     plt.figure(figsize=(10,10))
	# #     plt.legend(bbox_to_anchor=(1, 1))
	#     plt.legend(loc = 'upper left')
	#     plt.savefig(f'static/charts/cluster_{each_cluster}')

	# pd.merge(sorted_df, sector_df)

	# sorted_json =sorted_df.to_json()
	# sorted
	# return sorted_json
	# print('gothere')
		# print(request.form.dict())
		# response=make_response()
		# return response
	print('done training')
	# return sorted_json
	# table in html string
	table_html=sorted_df.to_html(index=False)

	# fig, ax=plt.subplots()
	# plt.plot([1, 2, 3, 4, 5], [1, 2, 3, 4, 5])

	########KEVIN WE NEED YOUR HELP HERE
	# plt.plot(date, movements_df["Amazon"])
	# plt.xlabel("Date")
	# plt.ylabel("Movements")
	# plt.title("Amazon Stock Movement")
	# plt.plot(date, movements_df) #We want to do it on the whole movements_df
	# fig.savefig('static/charts/cluster_move_one.png')


	# sorted_df.companies.str.split(expand = True)
	# sorted_df[["labels","companies", "symbol"]]=sorted_df.companies.str.split(",", expand =True,)
	sorted_df.to_csv('output/sorted_df.csv')
	# print(sorted_df['companies'].str.split(',', expand=True))
	#sorted_df["companies"]= sorted_df['companies'].str.split(",").str[:3]
	
	# merged_df = pd.merge(movements_df, sorted_df, on = companies)
	return render_template('predict.html', output=table_html)#, images=image_links)
#		return redirect('/' , data=sorted_json)
		# print ("complete")
	# return render_template("index.html", sorted_json)
	# else: 
	# 	return 'hi'






if __name__=='__main__':
	app.run(debug=True)