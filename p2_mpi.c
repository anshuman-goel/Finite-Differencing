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
* Group Info:
* agoel5 Anshuman Goel
* kgondha Kaustubh Gondhalekar
* ndas Neha Das
*
* LAST REVISED: 8/27/2017
******************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <limits.h>
#include <string.h>
#include "mpi.h"

#define BLOCKING 0 // 1: Blocking communication, 0: Non-Blocking communication

#define MANUALREDUCE 1 // 1: Manual Reduction; 0 : MPI_REDUCE

// The number of grid points
#define   NGRID     1000
// first grid point
#define   XI      -1.0 // -1.0
// last grid point
#define   XF      1.5
// the value of epsilon
#define EPSILON     0.005
// the degree of the function fn()
#define DEGREE      3

// function declarations
void print_function_data(int, double*, double*, double*);
void print_error_data(int np, double, double, double*, double*, double*);
int  main(int, char**);
void send();
double receive();
void send_blocking();
void send_nonblocking();
double receive_blocking();
double receive_nonblocking();
void do_vector_reduction();
void CustomReduce();

//returns the function y(x) = fn
double fn(double x)
{
        return pow(x, 3) - pow(x,2) - x + 1;
        // return pow(x, 2);
        // return x;
}

//returns the derivative d(fn)/dx = dy/dx
double dfn(double x)
{
        return (3*pow(x,2)) - (2*x) - 1;
        // return (2 * x);
        //  return 1;
}

int main (int argc, char *argv[])
{
        //loop index
        int i;

        //domain array and step size

        // double	x[NGRID + 2], dx;
        double dx;
        //function array and derivative
        //the size will be dependent on the
        //number of processors used
        //to the program
        double  *y, *dy;

        //local minima/maxima array
        double local_min_max[DEGREE-1];

        //"real" grid indices
        int imin, imax;

        //error analysis array
        double  *err;
        // To store the actual error at root processor
        double error[NGRID+1];
        //error analysis values
        double avg_err=0.0, std_dev=0.0;
        imin = 1;
        imax = NGRID;

        double error_end, error_start, start_finite_der, end_finite_der, start_dips, end_dips;
        double start_send_call_measure, end_send_call_measure;

        int numproc, rank;
        MPI_Init(&argc, &argv);
        MPI_Comm_size(MPI_COMM_WORLD, &numproc);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);

        if(numproc == 1){
            printf("Please use more than one processors for the parallel program to run.\n");
            MPI_Finalize();
            return 0;
        }

        int chunk_max, chunk_min; //local max and min for each chunk
        int chunk_size = (NGRID+2)/numproc;
        double *x; //array for the local grid points
        if(rank == numproc-1) {
                //Calculate chunk endpoints
                chunk_min = rank*chunk_size;
                //to handle if NGRID isn't evenly divisible by numproc
                if( (NGRID+2) % numproc != 0) {
                        chunk_size += (NGRID+2) % numproc; //chunksize for last process is different
                }
                chunk_max = NGRID+1;
        }
        else{
                //Calculate chunk endpoints
                chunk_min = rank*chunk_size;
                chunk_max = chunk_min + chunk_size - 1;
        }

        //Allocate space for gridpoints
        x = (double*) malloc(chunk_size*sizeof(double));
        //Calculate grid points
        //construct grid
        for(int i=1; i<chunk_size; i++) {
                x[i] = XI + (XF - XI) * (double)(i + chunk_min - 1)/ (double)(NGRID - 1);
        }
        dx = x[2] - x[1];
        x[0] = x[1] - dx;

        //Calculate y[i] for every value of x in the chunk
        y = (double*) malloc(chunk_size*sizeof(double));
        dy = (double*) malloc(chunk_size*sizeof(double));
        if(rank == 0) {
                //actual grid starts from 1 to chunk_max
                imin = 1;
                imax = chunk_size - 1;
                y[imin - 1] = fn(x[imin - 1]);
        }
        else if(rank == numproc-1) {
                //actual grid starts from chunk_min to chunk_max-1
                imin = 0;
                imax = chunk_size - 2;
                y[imax + 1] = fn(x[imax+1]);
                // printf("%f\n", y[imax + 1]);
        }
        else{
                //actual grid starts from chunk_min to chunk_max
                imin = 0;
                imax = chunk_size - 1;
        }

        //define the function
        for( i = imin; i <= imax; i++ ) {
                y[i] = fn(x[i]);
        }

        //initialize local min/max to dummy values
        for(i=0; i<DEGREE-1; i++) {
                local_min_max[i]=INT_MAX;
        }

        //Start measuring time for the send call
        start_send_call_measure = MPI_Wtime();

        // Storing the boundary f(x) values
        double left_end = 99, right_end = 99;
        if(rank == 0) {
                //Send y[chunk_max] to rank+1
                send(y[imax], rank+1);
                //Don't send y[chunk_min]

                //Don't receive left_end
                left_end = y[imin - 1];
                //Receive right_end from rank + 1
                right_end = receive(rank + 1);
        }
        else if(rank == numproc - 1) {
                //Don't send y[chunk_max]
                //Send y[chunk_min] to rank - 1
                send(y[imin], rank - 1);
                //Receive left_end from rank - 1
                left_end = receive(rank - 1);
                //Don't receive right_end from rank + 1
                right_end = y[imax + 1];
        }
        else{
                //Send y[chunk_max] to rank + 1
                //Send y[chunk_min] to rank - 1
                send(y[imax], rank + 1);
                send(y[imin], rank - 1);

                //Receive left_end from rank - 1
                //Receive right_end from rank + 1
                right_end = receive(rank+1);
                left_end = receive(rank-1);
        }
        //End measurement for send-call
        end_send_call_measure = MPI_Wtime();

        //Start measurement for finite-derivative calculation
        start_finite_der = MPI_Wtime();

        // Calculate approximate derivative and local_min_max
        int count = 0;
        // At imin
        dy[imin] = (y[imin + 1] - left_end)/(2.0 * dx);
        


        if (fabs(dy[imin])<EPSILON)
        {
                if(count >= DEGREE-1)
                {
                        printf("Warning: You have detected more than the maximum possible local minima/maxima.\n");
                        printf("Ensure that DEGREE is accurate or reduce your EPSILON.\n");
                        printf("Reseting count to zero.\n");
                        count = 0;
                }
                local_min_max[count++] = x[imin];
        }

        // Between imin and imax
        for(int i = imin+1; i<=imax-1; i++)
        {
                dy[i] = (y[i + 1] - y[i - 1])/(2.0 * dx);
                if (fabs(dy[i])<EPSILON)
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

        // At imax
        dy[imax] = (right_end - y[imax - 1])/ (2.0 * dx);
        if (fabs(dy[imax])<EPSILON )
        {
                if(count >= DEGREE-1)
                {
                        printf("Warning: You have detected more than the maximum possible local minima/maxima in process %d.\n", rank);
                        printf("Ensure that DEGREE is accurate or reduce your EPSILON.\n");
                        printf("Reseting count to zero.\n");
                        count = 0;
                }

                local_min_max[count++] = x[imax];
                // printf("-----------Local min max at proc %d: %f, dy/dx = %f\n",rank,x[imax], dy[imax] );

        }
        if(rank == 0)
        {
                //End measurement for finite-derivative calculation
                end_finite_der = MPI_Wtime();

                printf("Total send-call Time(s) %0.6e\n", end_send_call_measure - start_send_call_measure);
                printf("Total Finite Derivative Time(s) %0.6e\n", end_finite_der - start_finite_der);
        }

        error_start = MPI_Wtime();
        // Calculate error
        err = (double*)malloc((imax - imin + 1) * sizeof(double));
        for(i = imin; i <= imax; i++)
        {

                err[i-imin] = fabs( dy[i] - dfn(x[i]) );                
        }


        if (rank!=0)
        {
                // Sending size and error to root
                int err_size = (imax - imin + 1);
                MPI_Send(&err_size, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
                MPI_Send(err, (imax - imin + 1), MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
                if (MANUALREDUCE == 1)
                {
                        // Sending local_min_max to root processor for Manual Reduction
                        MPI_Send(local_min_max, DEGREE - 1, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD);
                }

        }

        // Initializing global_min_max with local minima and maxima of function f(x) at root
        double global_min_max[DEGREE - 1];
        memset(global_min_max, INT_MAX, (DEGREE - 1)*sizeof(double));

        if(rank==0)
        {
                // Concatenating error array from all the processors
                int proc_chunk_size; //ptr = imax - imin + 2;
                int ptr = 1;
                for(int i=0; i<imax; i++)
                {
                        error[i+1] = err[i];
                        ptr++;
                }
                for(int i=1; i<numproc; i++) {
                        // Receiving err array size from each processor
                        MPI_Recv(&proc_chunk_size, 1, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                        double temp_error[proc_chunk_size];
                        // Receiving err array from each processor

                        MPI_Recv(temp_error, proc_chunk_size, MPI_DOUBLE, i, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                        for(int j=0; j<proc_chunk_size; j++)
                        {
                                error[ptr] = temp_error[j];
                                ptr++;
                        }

                }

                error_end = MPI_Wtime();
                printf("Total Error Time(s) %0.6e\n", error_end - error_start);

                // Calculating average error at root
                for(int i=1; i<NGRID+1; i++)
                {
                        avg_err += error[i];
                }
                avg_err /= NGRID;

                // Calculating Standard deviation of error at root
                for(int i=1; i<NGRID+1; i++)
                {
                        std_dev += pow(avg_err - error[i],2);
                }
                std_dev = sqrt(std_dev/NGRID);

                // Calculate local minima and maxima with Manual Reduction
                if (MANUALREDUCE == 1) {
                        start_dips = MPI_Wtime();
                        int global_ptr = 0;

                        for (int i = 0; i < DEGREE - 1; i++) {
                                if ((int)(local_min_max[i]) < INT_MAX - 1 ) {

                                        if (global_ptr >= DEGREE-1 ) {
                                                printf("Bounds exceeded\n" );
                                                global_ptr = 0;
                                        }

                                        global_min_max[global_ptr] = local_min_max[i];
                                        global_ptr++;

                                }
                        }

                        for(int i=1; i<numproc; i++) {
                                double temp_min_max[DEGREE-1];
                                // Receiving local_min_max from each processor
                                MPI_Recv(temp_min_max, DEGREE-1, MPI_DOUBLE, i, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                                for(int j=0; j<DEGREE-1; j++) {
                                        if ((int)(temp_min_max[j]) < INT_MAX - 1) {

                                                if (global_ptr >= DEGREE-1 ) {
                                                        printf("Bounds exceeded\n" );
                                                        global_ptr = 0;
                                                }

                                                global_min_max[global_ptr] = temp_min_max[j];
                                                global_ptr++;
                                        }
                                }
                        }
                        end_dips = MPI_Wtime();
                        printf("Total local_min_max Derivative Time(s) %0.6e\n", end_dips - start_dips);
                }
        }

        // Calculating local minima and maxima using MPI_Reduce
        if (MANUALREDUCE == 0) {
                start_dips = MPI_Wtime();
                do_vector_reduction(&local_min_max, &global_min_max);
                if(rank==0)
                {                end_dips = MPI_Wtime();
                                 printf("Total local_min_max Derivative Time(s) %0.6e\n", end_dips - start_dips);
                }
        }

        if(rank==0)
        {
                double x_all[NGRID + 2];
                // Can be changed with MPI_Gather but haven't asked to do in Homework Question
                for (i = 1; i <= NGRID; i++)
                {
                        x_all[i] = XI + (XF - XI) * (double)(i - 1)/(double)(NGRID - 1);
                }
                // Writing to err.dat
                print_error_data(NGRID, avg_err, std_dev, &x_all[0], error, global_min_max);
        }

        free(y);
        free(dy);
        free(err);

        MPI_Finalize();

        return 0;
}

// Sending value to processor with rank = proc
void send(double value, int proc){
        if(BLOCKING == 1) {
                send_blocking(value, proc);
        }
        else{
                send_nonblocking(value, proc);
        }
}

// Receiving value from processor with rank = proc
double receive( int proc){
        if(BLOCKING == 1) {
                return receive_blocking(proc);
        }
        else{
                return receive_nonblocking(proc);
        }
}

// Sending value to processor with rank = proc using blocking communication
void send_blocking(double value, int proc){
        MPI_Send(&value, 1, MPI_DOUBLE, proc, 0, MPI_COMM_WORLD);
}

// Sending value to processor with rank = proc using non-blocking communication
void send_nonblocking(double value, int proc){
        MPI_Request req;
        MPI_Isend(&value, 1, MPI_DOUBLE, proc, 0, MPI_COMM_WORLD, &req);
        MPI_Request_free(&req);
}

// Receiving value from processor with rank = proc using blocking communication
double receive_blocking(int proc){
        double value;
        MPI_Recv(&value, 1, MPI_DOUBLE, proc, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        return value;
}

// Receiving value from processor with rank = proc using non-blocking communication
double receive_nonblocking(int proc){
        MPI_Request req;
        int flag = 0;
        MPI_Status status;
        double value;
        MPI_Irecv(&value, 1, MPI_DOUBLE, proc, 0, MPI_COMM_WORLD, &req);
        MPI_Test(&req, &flag, &status);
        while(!flag) {
                MPI_Test(&req, &flag, &status);
        }
        return value;
}

// Customized Reduction operation for MPI_Reduce to do Vector Reduction of local minima and maxima
void CustomReduce(void * local, void * global, int * len, MPI_Datatype *datatype){

        double *local_arr = (double *)local;
        double *global_arr = (double *)global;
        double *head = &global_arr[0];
        for (int i = 0; i < (DEGREE - 1); i++) {
                int break_flag = 0;
                if ((int)(local_arr[i]) < INT_MAX - 1 ) {
                        // put the value at a valid location in global ar
                        while((int) *global_arr < INT_MAX-1) {

                                int index = (global_arr - head)/sizeof(double);
                                //check if bounds exceeded
                                if( index >= DEGREE - 1) {
                                        printf("Bounds exceeded\n" );
                                        break_flag = 1;
                                        break;
                                }
                                global_arr++;
                        }
                        //check break flag
                        if(break_flag == 1) {
                                break;
                        }
                        *global_arr = local_arr[i];
                        global_arr++;
                }
        }
        //move global_arr back to starting position
        global_arr = head;
}

// Perform Vector Reduction via MPI_Reduce call
void do_vector_reduction(double *local_arr, double *global_arr)
{
        MPI_Op myOp;
        MPI_Op_create((MPI_User_function *) CustomReduce, 0, &myOp);
        MPI_Reduce (local_arr, global_arr, DEGREE-1, MPI_DOUBLE, myOp, 0, MPI_COMM_WORLD);
}

//prints out the function and its derivative to a file
void print_function_data(int np, double *x, double *y, double *dydx)
{
        int i;

        FILE *fp = fopen("fn.dat", "w");

        for(i = 0; i < np; i++)
        {
                fprintf(fp, "%f %f %f\n", x[i], y[i], dydx[i]);
        }

        fclose(fp);
}

//prints out the average error, standard deviation, x, error array and local minima and maxima to a file
void print_error_data(int np, double avgerr, double stdd, double *x, double *err, double *local_min_max)
{
        int i;
        FILE *fp = fopen("err.dat", "w+");

        fprintf(fp, "%0.6e\n%0.14e\n", avgerr, stdd);

        for(i = 0; i<DEGREE-1; i++)
        {
                if (local_min_max[i] != INT_MAX)
                        fprintf(fp, "(%f, %f)\n", local_min_max[i], fn(local_min_max[i]));
                else
                        fprintf(fp, "(UNDEF, UNDEF)\n");
        }

        for(i = 1; i <= np; i++)
        {
                fprintf(fp, "%f %0.6e \n", x[i], err[i]);
        }
        fclose(fp);
}
