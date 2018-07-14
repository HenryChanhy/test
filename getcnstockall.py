import tushare as ts
import datetime
from sys import argv
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
	filename = argv[1] if len(argv) >1 else 'cn_stocksample.csv'
	nums = int(argv[2]) if len(argv) > 2 else None
	stock_base = ts.get_stock_basics()
	if nums == None:
		stock_base.to_csv(filename,columns=['name'],header=None)
	else:
		stock_base.sample(n=nums).to_csv(filename,columns=['name'],header=None)
	#stock_base.to_csv('cn_stocksample.csv',columns=['name'],header=None)
	#savestock(stock_base)
