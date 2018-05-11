///////////////////////////////////////////////////////////////////////////////
// matmul.c
// Author: Dr. Richard A. Goodrum, Ph.D.
//
// Modification History:
// 1/30/2018 - Jimmy Nguyen (Jimmy@Jimmyworks.net)
// Added OpenMP directive to matmul function for parallel for-loop
// Defined OMP_MATMUL_TRIGGER as global constant
// Set OMP_MATMUL_TRIGGER to 2000 as tested to be minimum problem size for (M*N)
// Moved matmul function to top of program
//
// Procedures:
// main:	generates matrices and tests matmul
// matmul:	basic, brute force matrix multiply
///////////////////////////////////////////////////////////////////////////////

#include <omp.h>
#include <stdio.h>
#include <sys/time.h>

#define L (1*1024/256)          // Rows of Matrix A (Resulting Rows of C)
#define M (1*1024/256)          // Rows of Matrix B (Matching Col of A)
#define N (1*1024/256)          // Columns of Matrix B (Resulting Col of C)
#define OMP_MATMUL_TRIGGER 2000 // Minimum problem size (M*N) before OpenMP use
float A[L*M], B[M*N], C[L*N];

///////////////////////////////////////////////////////////////////////////////
// int matmul( int l, int m, int n, float *A, float *B, float *C )
// Author: Dr. Richard A. Goodrum, Ph.D.
// Date:  16 September 2017
//
// Modification History:
// 1/30/2018 - Jimmy Nguyen
// Added OpenMP pragma parallel for if(ompTrigger >= OMP_MATMUL_TRIGGER) private (k, j)
//
// Description: For each row of matrix A, for each column of matrix B, and for
//              each multiple, reduce the summation and store in matrix C.
//              Repeat this process till all elements completed.
// OpenMP Addition: If m*n is greater than the OMP_MATMUL_TRIGGER as defined, 
//                  then use OpenMP directive to parallelize the function using
//                  using additional threads to speed up the process.
// Parameters:
//	l	I/P	int	The first dimension of A and C
//	m	I/P	int	The second dimension of A and  first of B
//	n	I/P	int	The second dimension of B and C
//	A	I/P	float *	The first input matrix
//	B	I/P	float *	The second input matrix
//	C	O/P	float *	The output matrix
//	matmul	O/P	int	Status code
///////////////////////////////////////////////////////////////////////////////
int matmul( int l, int m, int n, float *A, float *B, float *C )
{
    int i, j, k;
    int ompTrigger = m*n;

    // If problem size large enough, execute all i tasks in parallel keeping
    // private copies of k and j in each thread
    #pragma omp parallel for if(ompTrigger >= OMP_MATMUL_TRIGGER) private (k, j)
    for( i=0; i<l; i++ )			// Loop over the rows of A and C.
    {
        for( k=0; k<n; k++ )			// Loop over the columns of B and C
        {
            // Initialize the output element for the inner
            // product of row i of A with column j of B
            C[i*n+k] = 0;						
            for( j=0; j<m; j++ )		// Loop over the columns of A and C
            {
                C[i*n+k] += A[i*m+j] * B[j*n+k];// Compute the inner product
            }
        }
    }
}

///////////////////////////////////////////////////////////////////////////////
// int main( int argc, char *argv[] )
// Author: Dr. Richard A. Goodrum, Ph.D.
// Date:  16 September 2017
// Description: Generates two matrices and then calls matmul to multiply them.
// 	Finally, it verifies that the results are correct.
//
// Parameters:
//	argc	I/P	int	The number of arguments on the command line
//	argv	I/P	char *[]	The arguments on the command line
//	main	O/P	int	Status code
///////////////////////////////////////////////////////////////////////////////
int main( int argc, char *argv[] )
{
        // Counters
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
        
	struct timeval start, stop;
	gettimeofday( &start, NULL );   // Record start time
	matmul( L, M, N, A, B, C );     // Matmul function
	gettimeofday( &stop, NULL );    // Record end time
	float elapsed = ( (stop.tv_sec-start.tv_sec) +
			(stop.tv_usec-start.tv_usec)/(float)1000000 );

	float flops = ( 2 * (float)L * (float)M * (float)N ) / elapsed;

        // Print result
	printf( "L=%d, M=%d, N=%d, elapsed=%g, flops=%g\n",
		L, M, N, elapsed, flops );

#ifdef DEBUG
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
#endif
}