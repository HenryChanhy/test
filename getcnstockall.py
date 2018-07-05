import tushare as ts
import datetime
from os.path import join, dirname, isfile
BASE_DIR = dirname(__file__)

def savestock(stocks):
	for i in range(len(stocks)):
		code=stocks.index[i]
		marketdate=stocks.ix[i]['timeToMarket']
		try:
			begindate=datetime.datetime.strptime(str(marketdate),'%Y%m%d').strftime('%Y-%m-%d')
		except ValueError:
			print ('stock:{} error marketdate {}'.format(code,marketdate))
			begindate='2000-01-01'
		filename=join(BASE_DIR, "cndata", code + ".csv")
		if not isfile(filename):
			stockkdata=ts.get_k_data(code,start=begindate)
			stockkdata.to_csv(filename)
			print('stock: {} saved: {} of {} total'.format(code,i,len(stocks)) )
		#else:
		#	print('stock: {} exist, querying passed {} of {} total'.format(code,i,len(stocks)))

if __name__ == "__main__":
	stock_base = ts.get_stock_basics()
	stock_base.sample(n=100).to_csv('cn_stocksample.csv',columns=['name'])
	#savestock(stock_base)
