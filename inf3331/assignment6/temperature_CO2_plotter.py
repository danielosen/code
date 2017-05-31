#Temperature and Carbondioxide plotter
#Python 3

#Import plotting package for plotting and converting to html
import matplotlib
matplotlib.use('Agg') #to ensure that plots can be saved properly
import matplotlib.pyplot as plt

#Import CSV package for reading files
import csv

#Import numpy for arrays
import numpy as np

#import for embedding to html using base64
import base64

#Import IO for writing file to buffer
from io import BytesIO

#define data reader function
def fetch_data(which,month=-1):
	'''
	Fetches and processes data from .csv files in the assignment folder.

	Requires arguments which (the file to be read) and month (default -1).

	Returns arrays of the processed data in the file(s).

	'''

	#read csv files
	#these are some size constants used for arrays
	num_co2 = 262
	num_temp = 197
	num_emiss = 264+1
	num_emiss2 = 2016-1960+1

	#read CO2 values
	if which=="co2":
		co2_year = np.zeros(num_co2,dtype='int')
		co2 = np.zeros(num_co2,dtype='int')
		i = 0
		with open('co2.csv', newline='') as csvfile:
			csvreader= csv.reader(csvfile, delimiter=' ',quotechar='|') #comma delimiter will not work here
			next(csvreader)
			for row in csvreader:
				for data in row:
					datalist = data.split(',')
					co2_year[i] = datalist[0]
					co2[i] = datalist[1]
					i += 1
		return co2_year,co2

	elif which=="temp":
		#read temperature values
		temp_year = np.zeros(num_temp,dtype='int')
		if month == -1:
			temp = np.zeros((num_temp,12),dtype='float')
		else:
			temp = np.zeros((num_temp,1),dtype='float')
		i = 0
		with open('temperature.csv',newline='') as csvfile:
			csvreader = csv.reader(csvfile, delimiter=' ', quotechar='|') #same as above for delimiter
			next(csvreader)
			for row in csvreader:
				for data in row:
					datalist = data.split(',')
					temp_year[i] = datalist[0]
					if month == -1:
						temp[i,:] = datalist[1:]
					else:
						temp[i] = datalist[month]

					i += 1
		return temp_year,temp
		#read emission values
	elif which=="country":
		country_codes = np.chararray((num_emiss,1),itemsize=5)
		co2_emissions = np.zeros((num_emiss,num_emiss2),dtype=float)
		i= 0
		with open('CO2_by_country.csv') as csvfile:
			#it was more work than usual to get csvreader to 
			#behave as I wanted, and so it was done from scratch
			for line in csvfile:
				line = line.split('",')
				country_code = line[1]
				country_code = country_code[1:] #remove leading "
				country_codes[i] = country_code
				emissions = line[4:-1] #dont grab endline or endfile
				j = 0
				for number in emissions:
					if number != '"':
						number = number[1:] #remove leading "
					else:
						number = -1 #missing entry value
					co2_emissions[i,j] = number
					j+=1
				i+=1
		return country_codes[1:],co2_emissions #the first row of country codes contains literal "Country Code"

#predictor function
def make_prediction(finalyear = 2016,month=-1):
	'''
	Predict future temperature in USA using linear interpolation on emission data.

	Requires arguments finalyear (default 2016) and month (default -1).

	Returns arrays of predicted temperature and the years for which temperature was predicted.

	Usually, temperature is regarded as depending logarithmically on CO2 emissions, which themselves grow exponentially with time,
	This results in temperature increasing linearly with time: temp = a*log(co2(time))+b = a'*time + b'.

	Instead, what we do here is use very few (recent) points of CO2 levels, so that the approximate change is linear w.r.t. to time,
	And then assume that temperature is linearly dependent on Co2, which is an OK approximation for small changes in CO2.
	'''
	pts = 5

	co2_year, co2 = fetch_data("co2") #fetch co2 data for all years
	temp_year, temp = fetch_data("temp") #fetch temp data for all monthsr
	#use only data from last 10 years
	#predict future co2 from linear model in time
	coeff_co2 = np.polyfit(co2_year[-pts:],co2[-pts:],1)
	future_year = np.arange(2012,finalyear+1)
	future_co2 = coeff_co2[0]*future_year+coeff_co2[1]

	if month==-1:
		temp = np.mean(temp,1) #choosing axis=1 averages over months for each year, choosing axis=0 averages over year for each month
	else:
		temp = temp[:,month-1]


	#predict future temp from linear model in co2
	coeff_temp = np.polyfit(co2[-pts:],temp[-pts:],1)
	future_temp = coeff_temp[0]*future_co2+coeff_temp[1]
	#return results
	return future_year, future_temp

#define plot functions

#plot co2
def plot_CO2(axis_list=[]):
	'''
Function: plot_CO2(axis_list=[])
Description: Plots CO2-time data (blue).
Arguments: axis_list, a list specifying the plotting ranges [x_min,x_max,y_min,y_max].
On the web page these are given as Min Year, Max Year, Min Carbon and Max Carbon respectively.
Returns a pyplot plot in .png html-ready image format.
	'''
	co2_year,co2 = fetch_data('co2')
	#make image
	fig = plt.figure(figsize=(5,4),dpi=100)
	axes = fig.add_subplot(1,1,1)
	axes.set_xlabel("Year")
	axes.set_ylabel("Carbon")
	axes.set_title('Carbon vs. Time Plot')
	axes.plot(co2_year,co2,'-')
	if axis_list:
		axes.axis(axis_list)
	#Open bytestream
	f = BytesIO()
	#Save image to bytestream
	plt.savefig(f,format="png")
	#seek to start of file
	f.seek(0)
	#first encode the byestream into an image string
	#then remove python's byte encoding, by decoding to ascii
	return base64.b64encode(f.getvalue()).decode('ascii')

#plot temperature
def plot_temperature(axis_list=[], finalyear= 2016, month=-1):
	'''
Function: plot_temperature(axis_list=[],finalyear=2016.month=-1)
Description: Plots Temperature-time data (blue), including prediction of future temperatures (green).
Arguments: axis_list, a list specifying the plotting ranges [x_min,x_max,y_min,y_max].
           On the web page, these are given as Min Year, Max Year, Min Temp and Max Temp respectively.
           finalyear, the last year of prediction. Accepts an integer value larger than 2012.
           month, the month number to plot and predict temperature data for. Accepts -1, 1,...,12,
           where -1: average of all months, and 1=Jan,..., 12= Dec.
Returns a pyplot plot in .png html-ready image format.
	'''
	temp_year,temp = fetch_data('temp',month)
	fig = plt.figure(figsize=(5,4),dpi=100)
	axes = fig.add_subplot(1,1,1)
	if month==-1:
		temp = np.mean(temp,1) #average over all months
	else:
		mean_axis = 0
	future_years, predicted_temp = make_prediction(finalyear,month)
	axes.plot(temp_year,temp,future_years,predicted_temp,'--')
	if axis_list:
		axes.axis(axis_list)
	axes.set_xlabel('Year')
	month_dict = {-1: 'All Months', 1: 'January', 2: 'February', 3: 'March', 4: 'April', 5: 'May',
	6: 'June', 7: 'July', 8: 'August', 9: 'September', 10: 'October', 11: 'November', 12: 'December'}
	axes.set_ylabel('Temperature in ' + month_dict[month])
	axes.set_title('Temperature vs Time Plot')

	#include prediction	

	#Open bytestream
	f = BytesIO()
	#Save image to bytestream
	plt.savefig(f,format="png")
	#seek to start of file
	f.seek(0)
	#first encode the byestream into an image string
	#then remove python's byte encoding, by decoding to ascii
	return base64.b64encode(f.getvalue()).decode('ascii')

#plot country co2 emissions
def plot_country_emissions(min_threshold=10,max_threshold=100,year=1994):
	'''
Function: plot_country_emissions(min_threshold=10,max_threshold=100,year=1994)
Description: Plots CO2 Emission-Country data as a horizontal bar chart.
Arguments: min_threshold, the minimum CO2 emission to be considered.
		   max_threshold, the maximum CO2 emission to be considered.
		   year, the year to plot CO2 emission data for.
Note that some countries lack emission data for certain years. These are not plotted.
Returns a pyplot plot in.png html-ready image format.
	'''
	#fetch data
	country_codes, co2_emissions = fetch_data("country")
	#grab data from specified year by finding year in first row and using that index for all rows
	#and remove the first row only containing the yeas (like we removed the literal "Country Code")
	co2_emissions = co2_emissions[1:,co2_emissions[0,:] == year]
	#now ignore missing points, make sure to do operation always on country code first
	country_codes = country_codes[co2_emissions > -1]
	co2_emissions = co2_emissions[co2_emissions > -1]
	#now ignore all above max threshold
	country_codes = country_codes[co2_emissions < max_threshold]
	co2_emissions = co2_emissions[co2_emissions < max_threshold]
	#now ignore all below min threshold
	country_codes = country_codes[co2_emissions > min_threshold]
	co2_emissions = co2_emissions[co2_emissions > min_threshold]
	#sort the results
	sortindex = np.argsort(co2_emissions)
	country_codes = country_codes[sortindex]
	co2_emissions = co2_emissions[sortindex]
	#prepare plot figure
	fig = plt.figure(figsize=(10,8),dpi=100)
	axes = fig.add_subplot(1,1,1)
	rgbamat = np.ones((co2_emissions.shape[0],3))
	mymax = np.max(co2_emissions)
	rgbamat[:,0]=1-co2_emissions[:]/mymax
	rgbamat[:,1]=1-co2_emissions[:]/mymax
	rgbamat[:,2]=1-co2_emissions[:]/mymax
	y_vals = np.arange(co2_emissions.shape[0])
	axes.barh(y_vals,co2_emissions,
		height=1, color=rgbamat, tick_label=country_codes.decode('ascii'),align='center')
	axes.set_xlabel('CO2 emissions (metric tons per capita)')
	axes.set_ylabel('Country Code')
	axes.set_title('CO2 emission of countries in ' + str(year))
	axes.tick_params(axis='both', which='major', labelsize=7)
	axes.tick_params(axis='both', which='minor', labelsize=7)
	axes.axis([0,mymax,y_vals[0]-1.5,y_vals[-1]+1.5])
	#Open bytestream
	f = BytesIO()
	#Save image to bytestream
	plt.savefig(f,format="png")
	#seek to start of file
	f.seek(0)
	#first encode the byestream into an image string
	#then remove python's byte encoding, by decoding to ascii
	return base64.b64encode(f.getvalue()).decode('ascii')

def grab_docstring(function_name):
	'''
	Function: grab_docstring(function_name)
	Arguments: a valid function name as string.
	Description: returns the docstring for function of functioname.
	Returns: list of chars (noted as strings)
	'''
	if function_name=="plot_CO2":
		return plot_CO2.__doc__
	elif function_name=="plot_temperature":
		return plot_temperature.__doc__
	elif function_name=="plot_country_emissions":
		return plot_country_emissions.__doc__
	else:
		return ""


#testing area