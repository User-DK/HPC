#include <mpi.h>
#include <iostream>
#include <vector>

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int N = 800; // larger size to see meaningful performance differences
    std::vector<int> matrix(N * N);
    std::vector<int> vector(N);
    std::vector<int> result(N, 0);

    if (rank == 0)
    {
        for (int i = 0; i < N; i++)
            vector[i] = i + 1;
        for (int i = 0; i < N * N; i++)
            matrix[i] = i + 1;
    }

    MPI_Bcast(vector.data(), N, MPI_INT, 0, MPI_COMM_WORLD);

    int rows_per_proc = N / size;
    std::vector<int> sub_matrix(rows_per_proc * N);
    MPI_Scatter(matrix.data(), rows_per_proc * N, MPI_INT, sub_matrix.data(), rows_per_proc * N, MPI_INT, 0, MPI_COMM_WORLD);

    std::vector<int> sub_result(rows_per_proc, 0);

    double start_time = MPI_Wtime(); // start timing

    for (int i = 0; i < rows_per_proc; i++)
        for (int j = 0; j < N; j++)
            sub_result[i] += sub_matrix[i * N + j] * vector[j];

    MPI_Gather(sub_result.data(), rows_per_proc, MPI_INT, result.data(), rows_per_proc, MPI_INT, 0, MPI_COMM_WORLD);

    double end_time = MPI_Wtime(); // end timing

    if (rank == 0)
    {
        std::cout << "Result (first 10 elements): ";
        for (int i = 0; i < std::min(10, N); i++)
            std::cout << result[i] << " ";
        std::cout << "\nExecution Time: " << (end_time - start_time) << " seconds" << std::endl;
    }

    MPI_Finalize();
    return 0;
}
