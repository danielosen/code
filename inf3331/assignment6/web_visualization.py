#Web Visualization Flask App
#For plotting temperature CO2 data
#Python 3
#Webpage http://localhost:5000/ 


#Import flask
from flask import Flask,render_template,request

#import temperature_plotter
import temperature_CO2_plotter as my_plot

#global plot data
global_plot_data = [1750,2016,0,10000,1800,2100,-8,26,-1,10,100,1994,2100]


#init app
app = Flask(__name__)

@app.route("/")
def main():
	#fetch from global plot data
	return render_template('index.html',
	CO2plot_xmin = global_plot_data[0],
	CO2plot_xmax = global_plot_data[1],
	CO2plot_ymin = global_plot_data[2],
	CO2plot_ymax = global_plot_data[3],
	Tplot_xmin = global_plot_data[4],
	Tplot_xmax = global_plot_data[5],
	Tplot_ymin = global_plot_data[6],
	Tplot_ymax = global_plot_data[7],
	Tplot_month = global_plot_data[8],
	Tplot_finalyear = global_plot_data[12],
	CO2plot = my_plot.plot_CO2(global_plot_data[0:4]),
	Tplot = my_plot.plot_temperature(global_plot_data[4:8],global_plot_data[12],global_plot_data[8]),
	Cplot_xmin = global_plot_data[9],
	Cplot_xmax = global_plot_data[10],
	Cplot_year = global_plot_data[11],
	Cplot = my_plot.plot_country_emissions(global_plot_data[9],global_plot_data[10],global_plot_data[11])
	)

@app.route('/image',methods=['POST'])
def image():
	global_plot_data[0] = int(request.form["x_min"])
	global_plot_data[1] = int(request.form["x_max"])
	global_plot_data[2] = int(request.form["y_min"])
	global_plot_data[3] = int(request.form["y_max"])
	return main()

@app.route('/image2',methods=['POST'])
def image2():
	global_plot_data[4] = int(request.form["x_min"])
	global_plot_data[5] = int(request.form["x_max"])
	global_plot_data[6] = int(request.form["y_min"])
	global_plot_data[7] = int(request.form["y_max"])
	global_plot_data[8] = int(request.form["month"])
	global_plot_data[12] = int(request.form["finalyear"])
	return main()

@app.route('/help')
def help():
	return render_template('help.html',
		co2_doc =my_plot.grab_docstring("plot_CO2"),
		temp_doc = my_plot.grab_docstring("plot_temperature"),
		bar_doc = my_plot.grab_docstring("plot_country_emissions"),
		)

@app.route('/image3',methods=['POST'])
def image3():
	global_plot_data[9] = int(request.form["x_min"])
	global_plot_data[10] = int(request.form["x_max"])
	global_plot_data[11] = int(request.form["emission_year"])
	return main()


if __name__ == '__main__':
	app.run()