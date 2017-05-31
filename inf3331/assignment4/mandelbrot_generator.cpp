/* mandelbrot_generator.cpp */
#include "mandelbrot_generator.h"

// Implementation essentially the same as with weave, only more thourough
// Note we are not using any index matrix here, because it is not needed
// May get tons of compiler warnings due to being c++ and not c

//Methods

void generate_Image(unsigned short int* image,int dimy, int dimx, double a, double b, double c, double d,int escapetime){
	int index; //for linear indexing
	// Sanity checking:
	int lenx = dimx-1;
	int leny = dimy-1;
	if (dimx <= 0){
		lenx = 1;
		b = a;
	}
	if (dimy <= 0){
		leny = 1;
		d = c;
	}
	//assume we get unintialized image, iterate from top left a,d to bottom right b,c
	double _Complex **z = new double _Complex*[dimy]; //allocate complex numbers
	for(int i=0;i<dimy;i++){
		z[i] = new double _Complex[dimx];
		for(int j=0;j<dimx;j++){
			z[i][j] = a+(b-a)/(lenx)*j + (d+(c-d)/(leny)*i)*_Complex_I;
			index=i*dimx+j;
			image[index] = 0;
		}
	}
	for(int i=0;i<dimy;i++){
		for(int j=0;j<dimx;j++){
			for(int t=0;t<escapetime; t++){
				if (pow(creal(z[i][j]),2)+pow(cimag(z[i][j]),2) <= 4){
					z[i][j] = z[i][j]*z[i][j]+ a+(b-a)/(lenx)*j + (d+(c-d)/(leny)*i)*_Complex_I;
					index = i*dimx+j;// linear indexing
					image[index] += 1;
				}else{
					break;
				}
			}
		}
	}
	//Make the python array
	//Delete and free up allocated memory to avoid memory leaks
	for(int i=0; i<dimy;++i){
		delete[] z[i];
	}
	delete[] z;
}

/*
bool is_Feasible(double a, double b, double c, double d){
	// Want to know if the rectangle intersects the set {z complex : (re z)^2 + (im z)^2 <= 4}
	// We assume that a<b gives real range and c<d gives imaginary range
	// i.e. the corners are (a,c), (a,d), (b,c), (b,d)
	bool intersectx = (a<=0.0 && b=>0.0);
	bool intersecty = (c<=0.0 && d=>0.0);
	if (intersectx && intersecty){
		return true; // (0,0) is in the rectangle
	}else if(intersectx){
		if (c>2 || d<-2){ //the rectangle above or below the line re z = 0 is too far away
			return false;
		}else{
			return true;
		}
	}else if(intersecty){
		if(b<-2 || a>2){
			return false; //the rectangle to the left or right of the line im z = 0 is too far away
		}else{
			return true;
		} 
	}else{ //the rectangle does not intersect either coordinate axes and is confined to a quadrant of the complex plane
	}
}
*/
