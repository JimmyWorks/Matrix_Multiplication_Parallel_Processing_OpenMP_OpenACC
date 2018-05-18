///////////////////////////////////////////////////////////////////////////////
// matmul.c
// Author: Dr. Richard A. Goodrum, Ph.D.
// Modified by: Jimmy Nguyen
//
// Description:
// Matrix multiplication is performed on a predetermined LxM and MxN matrix.
// OpenACC is utilized to demonstrate how parallel processing can be leveraged
// for improved performance.
//
// Procedures:
// main   generates matrices and tests matmul
// matmul   basic, brute force matrix multiply
///////////////////////////////////////////////////////////////////////////////

#include <stdio.h>
#include <sys/time.h>
#include <stdlib.h>
#include <string.h>


// Matmul function signature
int matmul( int, int, int, float*, float*, float* );

///////////////////////////////////////////////////////////////////////////////
// int main( int argc, char *argv[] )
// Author: Dr. Richard A. Goodrum, Ph.D.
// Date:  16 September 2017
// Description: Generates two matrices and then calls matmul to multiply them.
//    Finally, it verifies that the results are correct.
//
// Modification History:
// 2/25/2018 - Jimmy Nguyen
// Added OpenACC directives to matmul fuction
//
// Parameters:
//   argc   I/P   int   The number of arguments on the command line
//   argv   I/P   char *[]   The arguments on the command line
//   main   O/P   int   Status code
///////////////////////////////////////////////////////////////////////////////
int main( int argc, char *argv[] )
{
   // Verify that the correct number of arguments defined by command-line
   if(argc < 4 || argc > 5)
   {
      printf("Must input 3 integers\n");
      return 1;
   }

   // Get matrix parameters from command-line
   int L = atoi(argv[1]);
   int M = atoi(argv[2]);
   int N = atoi(argv[3]);
   //printf("Args are: %d, %d, %d\n", L, M, N);
  
   int DEBUG = 0;
   // If debug mode defined by command-line args, turn on debug mode  
   if(argc == 5 && !strcmp(argv[4], "debug"))
   {
      printf("Enabling debug\n");
      DEBUG = 1;
   //   printf("Set: %d\n", DEBUG);
   }   

   // Dynamically allocate memory for matrices A, B, and C
   float* A = (float*)calloc((L*M), sizeof(float));
   float* B = (float*)calloc((M*N), sizeof(float));
   float* C = (float*)calloc((L*N), sizeof(float));

   int i, j, k;

   // Initialize Matrix A
   for( i=0; i<L; i++ )
     for( j=0; j<M; j++ )
     {
       if( i <= j )
       {
      A[i*M+j] = (float) (i*M+j+1);
       }
       else
       {
      A[i*M+j] = 0.0;
      A[i*M+j] = (float) (i*M+j+1);
       }
     }
   // Initialize Matrix B
   for( j=0; j<M; j++ )
     for( k=0; k<N; k++ )
     {
       if( j <= k )
       {
         if( k < M )
      B[j*N+k] = 1.0;
         else
      B[j*N+k] = B[j*N+k-1] + 1.0;
       }
       else
       {
      B[j*N+k] = 0.0;
       }
     }
   // Initialize Matrix C
   for( i=0; i<L; i++ )
     for( k=0; k<N; k++ )
     {
       C[i*N+k] = - (float) L*M*N;
     }
	 
   //Start timer
   struct timeval start, stop;
   gettimeofday( &start, NULL );
   
   //Call matmul matrix multiplier
   matmul( L, M, N, A, B, C );
   
   //End timer
   gettimeofday( &stop, NULL );
   float elapsed = ( (stop.tv_sec-start.tv_sec) +
         (stop.tv_usec-start.tv_usec)/(float)1000000 );

   float flops = ( 2 * (float)L * (float)M * (float)N ) / elapsed;

   //Print results
   printf( "L=%d, M=%d, N=%d, elapsed=%g, flops=%g\n",
      L, M, N, elapsed, flops );

// If debug sentinel is true, this region will execute
if(DEBUG)
{
// Print all elements of matrix A
   printf( "A:\n" );
   for( i=0; i<L; i++ )
   {
     printf( "%g", A[i*M] );
     for( j=1; j<M; j++ )
     {
       printf( " %g", A[i*M+j] );
     }
     printf( "\n" );
   }
// Print all elements of matrix B
   printf( "B:\n" );
   for( j=0; j<M; j++ )
   {
     printf( "%g", B[j*N] );
     for( k=1; k<N; k++ )
     {
       printf( " %g", B[j*N+k] );
     }
     printf( "\n" );
   }
// Print all elements of matrix C
   printf( "C:\n" );
   for( i=0; i<L; i++ )
   {
     printf( "%g", C[i*N] );
     for( k=1; k<N; k++ )
     {
       printf( " %g", C[i*N+k] );
     }
     printf( "\n" );
   }
}
   // Free all dynamic memory allocations
   free(A);
   free(B);
   free(C);
}

///////////////////////////////////////////////////////////////////////////////
// int main( int argc, char *argv[] )
// Author: Dr. Richard A. Goodrum, Ph.D.
// Date:  16September 2017
// Description: Generates two matrices and then calls matmul to multiply them.
//    Finally, it verifies that the results are correct.
//
//
// Modification History:
// 2/25/2018 - Jimmy Nguyen
// Added int variable, sum, and OpenACC directives:
// 		- added OpenACC data region with copyin/copyout clauses
//		- added OpenACC kernels loop with designated loops for gang, worker, vector
// 		- added reduction clause for sum variable
//
// Parameters:
//   l   I/P   int   The first dimension of A and C
//   m   I/P   int   The second dimension of A and  first of B
//   n   I/P   int   The second dimension of B and C
//   A   I/P   float *   The first input matrix
//   B   I/P   float *   The second input matrix
//   C   O/P   float *   The output matrix
//   matmul   O/P   int   Status code

//  [ A ] * [ B ] = [ C ]
//  where A = [l * m] matrix
//  where B = [m * n] matrix
//  and   C = [l * n] matrix
///////////////////////////////////////////////////////////////////////////////
// Matmul //
int matmul( int l, int m, int n, float *A, float *B, float * restrict C )
{
   int i, j, k;   
   
   // OpenACC data region
   #pragma acc data copyin(A, B) copyout(C)
   {
	  // OpenACC processing region 
      #pragma acc kernels loop gang
      for( i=0; i<l; i++ )            // Loop over the rows of A and C.
      {
        #pragma acc worker
        for( k=0; k<n; k++ )            // Loop over the columns of B and C
        {
         int sum = 0;
         #pragma acc vector reduction (+:sum)
        // Initialize the output element for the inner
        // product of row i of A with column j of B
         for( j=0; j<m; j++ )            // Loop over the columns of A and C
         {
         sum += A[i*m+j] * B[j*n+k];   // Compute the inner product
         }
         C[i*n+k] = sum;
        }
      }
   }
}

