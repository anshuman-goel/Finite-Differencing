/******************************************************************************
* FILE: p2.c
* DESCRIPTION:
*
* Users will supply the functions
* i.) fn(x) - the polynomial function to be analyized
* ii.) dfn(x) - the true derivative of the function
* iii.) degreefn() - the degree of the polynomial
*
* The function fn(x) should be a polynomial.
*
* AUTHOR: Tyler Stocksdale
* LAST REVISED: 8/23/2017
******************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <limits.h>
#include <sys/time.h>

// The number of grid points
#define   NGRID			1000000
// first grid point
#define   XI			-1.0
// last grid point
#define   XF			1.5
// the value of epsilon
#define EPSILON			0.000005
// the degree of the function fn()
#define DEGREE			3

// function declarations
void        print_function_data(int, double*, double*, double*);
void        print_error_data(int np, double, double, double*, double*, double*);
int         main(int, char**);

//returns the function y(x) = fn
double fn(double x)
{
  return pow(x, 3) - pow(x,2) - x + 1;
  //return pow(x, 2);
  //return x;
}

//returns the derivative d(fn)/dx = dy/dx
double dfn(double x)
{
  return (3*pow(x,2)) - (2*x) - 1;
  //return (2 * x);
  //return 1;
}

int main (int argc, char *argv[])
{
  //loop index
  int		i, count;

  //domain array and step size
  double	x[NGRID + 2], dx;

  //function array and derivative
  //the size will be dependent on the
  //number of processors used
  //to the program
  double	*y, *dy;

  //local minima/maxima array
  double	local_min_max[DEGREE-1];

  //"real" grid indices
  int		imin, imax;

  //error analysis array
  double	*err;

  //error analysis values
  double	avg_err, std_dev;

  struct timeval start, end;
  gettimeofday(&start, NULL);
  imin = 1;
  imax = NGRID;

  //construct grid
  for (i = 1; i <= NGRID ; i++)
  {
	x[i] = XI + (XF - XI) * (double)(i - 1)/(double)(NGRID - 1);
  }
  //step size and boundary points
  dx = x[2] - x[1];
  x[0] = x[1] - dx;
  x[NGRID + 1] = x[NGRID] + dx;

  //allocate function arrays
  y  =   (double*) malloc((NGRID + 2) * sizeof(double));
  dy =   (double*) malloc((NGRID + 2) * sizeof(double));

  //define the function
  for( i = imin; i <= imax; i++ )
  {
	y[i] = fn(x[i]);
  }

  //set boundary values
  y[imin - 1] = fn(x[0]);
  y[imax + 1] = fn(x[NGRID + 1]);

  //initialize local min/max to dummy values
  for(i=0; i<DEGREE-1; i++)
  {
	local_min_max[i]=INT_MAX;
  }

  //compute the derivative using first-order finite differencing
  //
  //  d           f(x + h) - f(x - h)
  // ---- f(x) ~ --------------------
  //  dx                 2 * dx
  //
  gettimeofday(&start, NULL);

  for (i = imin; i <= imax; i++)
  {
	dy[i] = (y[i + 1] - y[i - 1])/(2.0 * dx);
  }
  gettimeofday(&end, NULL);
  printf("Finite Diff %0.6e\n", (end.tv_usec-start.tv_usec)/pow(10,6));


  print_function_data(NGRID, &x[1], &y[1], &dy[1]);


  gettimeofday(&start, NULL);
  //compute the error, average error of the derivatives
  err = (double*)malloc(NGRID * sizeof(double));

  //compute the errors
  for(i = imin; i <= imax; i++)
  {
	err[i-imin] = fabs( dy[i] - dfn(x[i]) );
  }

  gettimeofday(&end, NULL);
  printf("Error Time %0.6e\n", (end.tv_usec-start.tv_usec)/pow(10,6));
  //find the average error
  avg_err = 0.0;
  for(i = 0; i < NGRID ; i++)
	avg_err += err[i];

  avg_err /= (double)NGRID;

  //find the standard deviation of the error
  //standard deviation is defined to be
  //
  //                   ____________________________
  //          __      /      _N_
  // \omega =   \    /  1    \
  //             \  /  --- *  >  (x[i] - avg_x)^2
  //              \/    N    /__
  //                        i = 1
  //
  std_dev = 0.0;
  for(i = 0; i< NGRID; i++)
  {
	std_dev += pow(err[i] - avg_err, 2);
  }
  std_dev = sqrt(std_dev/(double)NGRID);


  //find the local minima/maxima
  //
  //  |  dy  |
  //  | ---- | < epsilon
  //  |  dx  |
  //
  gettimeofday(&start, NULL);
  count = 0;
  for(i = imin; i <= imax; i++)
  {
	if(fabs(dy[i]) < EPSILON)
	{
		if(count >= DEGREE-1)
		{
			printf("Warning: You have detected more than the maximum possible local minima/maxima.\n");
			printf("Ensure that DEGREE is accurate or reduce your EPSILON.\n");
			printf("Reseting count to zero.\n");
			count = 0;
		}
		local_min_max[count++] = x[i];
	}
  }
  gettimeofday(&end, NULL);
  printf("local_min_max Time %0.6e\n", (end.tv_usec-start.tv_usec)/pow(10,6));
  //free allocated memory
  free(y);
  free(dy);
  free(err);

  return 0;
}

//prints out the function and its derivative to a file
void print_function_data(int np, double *x, double *y, double *dydx)
{
  int   i;

  FILE *fp = fopen("fn.dat", "w");

  for(i = 0; i < np; i++)
  {
	fprintf(fp, "%f %f %f\n", x[i], y[i], dydx[i]);
  }

  fclose(fp);
}

void print_error_data(int np, double avgerr, double stdd, double *x, double *err, double *local_min_max)
{
  int   i;
  FILE *fp = fopen("err.dat", "w");

  fprintf(fp, "%e\n%e\n", avgerr, stdd);

  for(i = 0; i<DEGREE-1; i++)
  {
	if (local_min_max[i] != INT_MAX)
		fprintf(fp, "(%f, %f)\n", local_min_max[i], fn(local_min_max[i]));
	else
		fprintf(fp, "(UNDEF, UNDEF)\n");
  }

  for(i = 0; i < np; i++)
  {
	fprintf(fp, "%f %e \n", x[i], err[i]);
  }
  fclose(fp);
}
