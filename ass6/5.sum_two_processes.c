#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv) {
    MPI_Init(&argc,&argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    
    int n = 10;
    int half = n / 2;

    if (rank == 0) {
        int *A = malloc(n * sizeof(int));
        for (int i = 0; i < n; ++i) A[i] = i+1;
        // sum first half
        int sum0 = 0;
        for (int i = 0; i < half; ++i) sum0 += A[i];

        // send second half to process 1
        MPI_Send(A + half, half, MPI_INT, 1, 0, MPI_COMM_WORLD);

        // receive partial sum from P1
        int sum1;
        MPI_Recv(&sum1, 1, MPI_INT, 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        printf("P0: sum0=%d, sum1=%d, total=%d\n", sum0, sum1, sum0 + sum1);
        free(A);
    } else if (rank == 1) {
        // receive half from P0
        int *buf = malloc(half * sizeof(int));
        MPI_Recv(buf, half, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        int sum1 = 0;
        for (int i = 0; i < half; ++i) sum1 += buf[i];

        // send partial sum back
        MPI_Send(&sum1, 1, MPI_INT, 0, 1, MPI_COMM_WORLD);
        free(buf);
    } else {
        //do nothing or exit
    }

    MPI_Finalize();
    return 0;
}
