# INF3331-danieleo

The python code for generating the plots and reading .csv files can be found in
	/temperature_CO2_plotter.py

The python code for generating the webpage using the plots can be found in
	/web_visualization.py

Note: the python scripts require the packages numpy, pyplot, flask, csv, base64, io

To start the web app server run:
	python3 web_visualization.py,

The webpage can be accessed at:
	http://localhost:5000/

Note: Do not attempt to access other sub-pages directly (except help) as these are methods.

The .html pages can be found in:
	/templates/index.html
	/templates/help.html

The .css and .js scripts can be found in:
	/static/bootstrap.min.css
	/static/bootstrap.min.js
	/static/jquery-3.1.1.min.js

Check the help page for docstrings, and make sure to input sensical values. The graphs will be
re-generated once any changes are submitted, refreshing the page. Changes to graphs are saved.

The server was tested on using Mozilla Firefox running on VMWare Ubuntu 14.04 (64-Bit).
