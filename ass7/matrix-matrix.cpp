#include <mpi.h>
#include <iostream>
#include <vector>

int main(int argc, char **argv)
{
  MPI_Init(&argc, &argv);

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int N = 500;
  std::vector<long long> A(N * N);
  std::vector<long long> B(N * N);
  std::vector<long long> C(N * N, 0);

  if (rank == 0)
  {
    for (int i = 0; i < N * N; i++)
      A[i] = i + 1;
    for (int i = 0; i < N * N; i++)
      B[i] = i + 1;
  }

  MPI_Bcast(B.data(), N * N, MPI_LONG_LONG, 0, MPI_COMM_WORLD);

  int rows_per_proc = N / size;
  std::vector<long long> sub_A(rows_per_proc * N);

  MPI_Scatter(A.data(), rows_per_proc * N, MPI_LONG_LONG, sub_A.data(), rows_per_proc * N, MPI_LONG_LONG, 0, MPI_COMM_WORLD);

  std::vector<long long> sub_C(rows_per_proc * N, 0);

  double start_time = MPI_Wtime();

  for (int i = 0; i < rows_per_proc; i++)
    for (int j = 0; j < N; j++)
      for (int k = 0; k < N; k++)
        sub_C[i * N + j] += sub_A[i * N + k] * B[k * N + j];

  MPI_Gather(sub_C.data(), rows_per_proc * N, MPI_LONG_LONG, C.data(), rows_per_proc * N, MPI_LONG_LONG, 0, MPI_COMM_WORLD);

  double end_time = MPI_Wtime();

  if (rank == 0)
  {
    std::cout << "C[0][0] = " << C[0] << std::endl;
    std::cout << "Execution Time: " << (end_time - start_time) << " seconds" << std::endl;
  }

  MPI_Finalize();
  return 0;
}
