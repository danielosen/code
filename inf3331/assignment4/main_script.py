#main_script.py
import sys,os
import subprocess
show_help = False


if sys.argv.__len__() > 1:
	if sys.argv[1] == "--help":
		show_help = True
	if sys.argv.__len__() > 7 and not show_help:
		try:
			_name = sys.argv[1]
			_a = float(sys.argv[2])
			_b = float(sys.argv[3])
			_c = float(sys.argv[4])
			_d = float(sys.argv[5])
			_nx = int(sys.argv[6])
			_ny = int(sys.argv[7]) 
			_filename = sys.argv[8]
		except:
			raise ValueError("Unable to parse input.")
else:
	raise ValueError("Usage: main_script.py NAME a b c d Nx Ny FILENAME, or main_script.py --help")


if show_help:
	print("\n[How-To]")
	print("main_script.py NAME a b c d Nx Ny FILENAME\n")
	print("NAME is replaced by mandelbrot_1.py, mandelbrot_2.py, mandelbrot_3.py or mandelbrot_4.py\n")
	print("a,b,c,d define a rectangle in the complex space")
	print("a: min value of reals")
	print("b: max value of reals")
	print("c: min value of imaginaries")
	print("d: max value of imaginaries\n")
	print("FILENAME is the name of the image file. A .png suffix will be appended when saving.")
	print("The image file is saved in current folder.")
	print("In addition, the computation time is printed to the terminal.")
else:
	_filename = "{}.png".format(_filename)
	run_string = "python3 {} {} {} {} {} {} {} {}".format(_name,_a,_b,_c,_d,_nx,_ny,_filename)
	os.system(run_string)
