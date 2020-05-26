#include <limits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <fstream>
#include <string>
#include <sys/time.h>
#include <unistd.h>

#include <Kokkos_Core.hpp>
#include "hdf5.h"
#include "dataspaces.h"
#include "mpi.h"

void checkSizes( int &N, int &M, int &S, int &nrepeat );

int main( int argc, char* argv[] )
{
  int N = -1;         // number of rows 2^12
  int M = -1;         // number of columns 2^10
  int S = -1;         // total size 2^22
  int nrepeat = 10;  // number of repeats of the test

  // Read command line arguments.
  for ( int i = 0; i < argc; i++ ) {
    if ( ( strcmp( argv[ i ], "-N" ) == 0 ) || ( strcmp( argv[ i ], "-Rows" ) == 0 ) ) {
      N = pow( 2, atoi( argv[ ++i ] ) );
      printf( "  User N is %d\n", N );
    }
    else if ( ( strcmp( argv[ i ], "-M" ) == 0 ) || ( strcmp( argv[ i ], "-Columns" ) == 0 ) ) {
      M = pow( 2, atof( argv[ ++i ] ) );
      printf( "  User M is %d\n", M );
    }
    else if ( ( strcmp( argv[ i ], "-S" ) == 0 ) || ( strcmp( argv[ i ], "-Size" ) == 0 ) ) {
      S = pow( 2, atof( argv[ ++i ] ) );
      printf( "  User S is %d\n", S );
    }
    else if ( strcmp( argv[ i ], "-nrepeat" ) == 0 ) {
      nrepeat = atoi( argv[ ++i ] );
    }
    else if ( ( strcmp( argv[ i ], "-h" ) == 0 ) || ( strcmp( argv[ i ], "-help" ) == 0 ) ) {
      printf( "  y^T*A*x Options:\n" );
      printf( "  -Rows (-N) <int>:      exponent num, determines number of rows 2^num (default: 2^12 = 4096)\n" );
      printf( "  -Columns (-M) <int>:   exponent num, determines number of columns 2^num (default: 2^10 = 1024)\n" );
      printf( "  -Size (-S) <int>:      exponent num, determines total matrix size 2^num (default: 2^22 = 4096*1024 )\n" );
      printf( "  -nrepeat <int>:        number of repetitions (default: 100)\n" );
      printf( "  -help (-h):            print this message\n\n" );
      exit( 1 );
    }
  }

  // Check sizes.
  checkSizes( N, M, S, nrepeat );
  int rank, nprocs; 
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Comm gcomm = MPI_COMM_WORLD;
    
  
  dspaces_init(nprocs, 2, &gcomm, NULL);

  Kokkos::initialize( argc, argv );
  {
  
  // Allocate Matrix A on device.
  typedef Kokkos::View<double**>  ViewMatrixType;
  ViewMatrixType A( "A", N, M );

  // Timer products.
  Kokkos::Timer timer;
  

  for(int timestep=0; timestep<nrepeat; timestep++) {

    //printf( "Timestep %d Dataspace Get....\n", timestep);
    // DataSpaces: Read-Lock Mechanism
    // Usage: Prevent other processies from changing the 
    // 	  data while we are working with it
    dspaces_lock_on_read("A_lock", NULL);

    //Name the Data that will be read
    char var_name[128];
    sprintf(var_name, "A_rank%d",rank);
    // ndim: Dimensions for application data domain
    // In this case, our data array is 1 dimensional
    int ndim = 2; 

    uint64_t lb[2] = {0}, ub[2] = {0};

    ub[0] = N-1; 
    ub[1] = M-1;
  
    dspaces_get(var_name, timestep, sizeof(double), ndim, lb, ub, A.data());
    // DataSpaces: Release our lock on the data
    dspaces_unlock_on_read("A_lock", NULL);

    //printf( "Timestep %d Dataspace Get Ends....\n", timestep);

  }
  // Calculate time.
  double time = timer.seconds();

  // Calculate bandwidth.
  // Each matrix A row (each of length M) is read once.
  
  double Gbytes = 1.0e-9 * double( sizeof(double) * M * N );

  // Print results (problem size, time and bandwidth in GB/s).
  printf( "  N( %d ) M( %d ) nrepeat ( %d ) problem( %g MB ) time( %g s ) bandwidth( %g GB/s )\n",
          N, M, nrepeat, Gbytes * 1000, time, Gbytes * nrepeat / time );


  Kokkos::Timer timer1;
  for (int timestep = 0; timestep < nrepeat; timestep++)
  {
      std::string path = "../comparison_writer/cppio_rank";
      path += std::to_string(rank) + "_ts";
      path += std::to_string(timestep)+".dat";
      std::ifstream input;
      input.open(path, std::ios::in | std::ios::binary);
      input.read(reinterpret_cast<char *>(A.data()), sizeof(double)*N*M);
      input.close();
  }
  double time1 = timer1.seconds();
  printf( "CppIO Reader:  N( %d ) M( %d ) nrepeat ( %d ) problem( %g MB ) time( %g s ) bandwidth( %g GB/s )\n",
          N, M, nrepeat, Gbytes * 1000, time1, Gbytes * nrepeat / time1 );


  Kokkos::Timer timer2;

  std::string filepath = "../comparison_writer/HDf5_rank";
  filepath += std::to_string(rank) + ".h5";

  hid_t file_id = H5Fopen(filepath.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);

  for(int timestep=0; timestep<nrepeat; timestep++)
  {
      std::string path = "timestep_";
      path += std::to_string(timestep);

      hid_t dataset_id = H5Dopen2(file_id, path.c_str(), H5P_DEFAULT);

      herr_t dataset_status = H5Dread(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, A.data());

      dataset_status = H5Dclose(dataset_id);
  }

  herr_t status = H5Fclose(file_id);

  double time2 = timer2.seconds();
  printf( "HDF5 Reader:  N( %d ) M( %d ) nrepeat ( %d ) problem( %g MB ) time( %g s ) bandwidth( %g GB/s )\n",
          N, M, nrepeat, Gbytes * 1000, time2, Gbytes * nrepeat / time2 );
   

  }
  Kokkos::finalize();

  
  MPI_Barrier(gcomm);
  MPI_Finalize();

  return 0;

}

void checkSizes( int &N, int &M, int &S, int &nrepeat ) {
  // If S is undefined and N or M is undefined, set S to 2^22 or the bigger of N and M.
  if ( S == -1 && ( N == -1 || M == -1 ) ) {
    S = pow( 2, 22 );
    if ( S < N ) S = N;
    if ( S < M ) S = M;
  }

  // If S is undefined and both N and M are defined, set S = N * M.
  if ( S == -1 ) S = N * M;

  // If both N and M are undefined, fix row length to the smaller of S and 2^10 = 1024.
  if ( N == -1 && M == -1 ) {
    if ( S > 1024 ) {
      M = 1024;
    }
    else {
      M = S;
    }
  }

  // If only M is undefined, set it.
  if ( M == -1 ) M = S / N;

  // If N is undefined, set it.
  if ( N == -1 ) N = S / M;

  printf( "  Total size S = %d N = %d M = %d\n", S, N, M );

  // Check sizes.
  if ( ( S < 0 ) || ( N < 0 ) || ( M < 0 ) || ( nrepeat < 0 ) ) {
    printf( "  Sizes must be greater than 0.\n" );
    exit( 1 );
  }

  if ( ( N * M ) != S ) {
    printf( "  N * M != S\n" );
    exit( 1 );
  }
}