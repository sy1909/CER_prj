
def simple_forecast(data_frame_forecast):

	from sklearn.metrics import mean_squared_error
	from math import sqrt

	data_frame_forecast = data_frame_forecast.set_index('기준일')
	X = data_frame_forecast.values
	X = X.astype('float32')
	train_size = int(len(X) * 0.50)
	train, test = X[0:train_size], X[train_size:]

	history = [x for x in train]
	predictions = list()
	for i in range(len(test)):
	    yhat = history[-1]
	    predictions.append(yhat)
	        
	    obs = test[i]
	    history.append(obs)
	    print('>Predicted=%.3f, Expected=%3.f' % (yhat, obs))

	mse = mean_squared_error(test, predictions)
	rmse = sqrt(mse)
	print('RMSE: %.3f' % rmse)


	# In[ ]:


	split_point = len(data_frame_forecast) - 8
	dataset, validation = data_frame_forecast[0:split_point], data_frame_forecast[split_point:]
	print('Dataset %d, Validation %d' % (len(dataset), len(validation)))
	dataset.to_csv('dataset.csv')
	validation.to_csv('validation.csv')


	# In[ ]:


	import warnings
	from statsmodels.tsa.arima_model import ARIMA
	from sklearn.metrics import mean_squared_error

	data_frame_forecast.dropna(inplace=True)

	def evaluate_arima_model(X, arima_order):

		X = data_frame_forecast.astype('float32')
		train_size = int(len(X) * 0.50)
		train, test = X[0:train_size], X[train_size:]
		history = [x for x in train]
		
		predictions = list()
		for t in range(len(test)):
			model = ARIMA(history, order=arima_order)
			
			model_fit = model.fit(trend='nc', disp=0)
			yhat = model_fit.forecast()[0]
			predictions.append(yhat)
			history.append(test[t])

		mse = mean_squared_error(test, predictions)
		rmse = sqrt(mse)
		return rmse

	def evaluate_models(dataset, p_values, d_values, q_values):
		dataset = dataset.astype('float32')
		best_score, best_cfg = float("inf"), None
		for p in p_values:
			for d in d_values:
				for q in q_values:
					order = (p,d,q)
					try:
						mse = evaluate_arima_model(dataset, order)
						if mse < best_score:
							best_score, best_cfg = mse, order
						print('ARIMA%s RMSE=%.3f' % (order,mse))
					except:
						continue
		print('Best ARIMA%s RMSE=%.3f' % (best_cfg, best_score))
	 
	p_values = range(0, 5)
	d_values = range(0, 3)
	q_values = range(0, 5)
	warnings.filterwarnings("ignore")
	evaluate_models(data_frame_forecast.values, p_values, d_values, q_values)


	# In[ ]:


	# 적정 arima 모델을 찾아내지 못했으므로 수동으로 찾아야 함
	from statsmodels.tsa.stattools import adfuller
	data_frame_forecast.dropna(inplace=True)

	X = data_frame_forecast['1인배출량'].values
	result = adfuller(X)
	print('ADF Statistic: %f' % result[0])
	print('p-value: %f' % result[1])
	print('Critical Values:')
	for key, value in result[4].items():
		print('\t%s: %.3f' % (key, value))

	# 이 테스트의 p-값을 보면 귀무 가설을 기각할 수 없으며 
	# 데이터 세트가 비정상적일 가능성이 높습니다. 
	# 따라서 ARIMA 모델의 첫 번째 매개변수(p)를 1로 선택합니다. 
	# 데이터 세트가 고정적이지 않지만 연간 데이터로 작업하기 때문에 
	# 계절성이 존재할 가능성이 없기 때문에 더 높은 숫자를 선택하지 않습니다.


	# In[ ]:


	from statsmodels.graphics.tsaplots import plot_acf
	from statsmodels.graphics.tsaplots import plot_pacf
	from matplotlib import pyplot

	pyplot.figure()
	pyplot.subplot(211)
	plot_acf(data_frame_forecast, ax=pyplot.gca())
	pyplot.subplot(212)
	plot_pacf(data_frame_forecast, ax=pyplot.gca())
	pyplot.show()


	# In[ ]:


	from pandas import DataFrame
	from sklearn.metrics import mean_squared_error
	from statsmodels.tsa.arima_model import ARIMA
	from math import sqrt

	import warnings
	warnings.filterwarnings("ignore")

	data_frame_forecast.dropna(inplace=True)

	X = data_frame_forecast.values
	X = X.astype('float32')
	train_size = int(len(X) * 0.50)
	train, test = X[0:train_size], X[train_size:]

	history = [x for x in train]
	predictions = list()
	for i in range(len(test)):

		model = ARIMA(history, order=(1,1,0))
		model_fit = model.fit(trend='nc', disp=0)
		yhat = model_fit.forecast()[0]
		predictions.append(yhat)

		obs = test[i]
		history.append(obs)

	residuals = [test[i]-predictions[i] for i in range(len(test))]
	residuals = DataFrame(residuals)
	print(residuals.describe())


	# In[ ]:


	from statsmodels.tsa.arima_model import ARIMA
	from scipy.stats import boxcox
	import numpy

	warnings.filterwarnings("ignore")

	def __getnewargs__(self):
		return ((self.endog),(self.k_lags, self.k_diff, self.k_ma))
	 
	ARIMA.__getnewargs__ = __getnewargs__

	data_frame_forecast.dropna(inplace=True)
	X = data_frame_forecast.values
	X = X.astype('float32')

	model = ARIMA(X, order=(1,1,0))
	model_fit = model.fit(trend='nc', disp=0)

	bias = 0.153408

	model_fit.save('model.pkl')
	numpy.save('model_bias.npy', [bias])


	# In[ ]:


	warnings.filterwarnings("ignore")
	from statsmodels.tsa.arima_model import ARIMAResults

	model_fit = ARIMAResults.load('model.pkl')
	bias = numpy.load('model_bias.npy')
	yhat = bias + float(model_fit.forecast()[0])
	print('Predicted: %.3f' % yhat)


	# In[ ]:


	from statsmodels.tsa.arima_model import ARIMAResults
	from sklearn.metrics import mean_squared_error
	from matplotlib import pyplot
	from math import sqrt

	def difference(dataset):
	        diff = list()
	        for i in range(1, len(dataset)):
	                value = dataset[i] - dataset[i - 1]
	                diff.append(value)
	        return diff

	data_frame_forecast.dropna(inplace=True)
	X = data_frame_forecast.values
	X = X.astype('float32')

	stationary = difference(X)

	df_X = pd.DataFrame({'year': data_frame_forecast.index.values, 'X':X.flatten()})

	validation = pd.read_csv('validation.csv')
	display(validation)
	validation = validation.set_index('기준일')

	def difference(validation):
	        diff = list()
	        for i in range(1, len(validation)):
	                value = validation[i] - validation[i - 1]
	                diff.append(value)
	        return diff

	y = validation.values
	y = y.astype('float32')
	      
	stationary = difference(y)
	display(validation.index.values)
	display(y.flatten())
	df_y = pd.DataFrame({'year': validation.index.values, 'y':y.flatten()})

	history = [x for x in X]

	model_fit = ARIMAResults.load('model.pkl')
	bias = numpy.load('model_bias.npy')

	predictions = list()
	yhat = bias + float(model_fit.forecast()[0])
	predictions.append(yhat)
	history.append(y[0])
	print('>Predicted=%.3f, Expected=%3.f' % (yhat, y[0]))

	for i in range(1, len(y)):
	  model = ARIMA(history, order=(1,1,0))
	  model_fit = model.fit(trend='nc', disp=0)
	  yhat = bias + float(model_fit.forecast()[0])
	  predictions.append(yhat)
		
	  obs=y[i]
	  history.append(obs)
	  print('>Predicted=%.3f, Expected=%3.f' % (yhat, obs))

	mse = mean_squared_error(y, predictions)
	rmse = sqrt(mse)
	print('RMSE: %.3f' % rmse)
	pyplot.plot(y)
	pyplot.plot(predictions, color='red')
	pyplot.show()


	# In[ ]:


	model_fit = ARIMAResults.load('model.pkl')
	bias = numpy.load('model_bias.npy')
	forecast = model_fit.forecast(steps=12)[0]
	print (forecast)

