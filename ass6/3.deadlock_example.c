#include <mpi.h>
#include <stdio.h>

int main(int argc, char** argv) {
    MPI_Init(&argc,&argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&size);

    if (size != 2) {
        if (rank == 0) {
            fprintf(stderr, "This deadlock demo requires exactly 2 processes.\n");
        }
        MPI_Finalize();
        return 1;
    }

    int tag = 0;
    int send_val = (rank == 0) ? 100 : 200;
    int recv_val = -1;

    // Both processes perform a synchronous send first -> this will cause deadlock
    printf("Rank %d: calling MPI_Ssend to %d\n", rank, 1-rank);
    fflush(stdout);
    MPI_Ssend(&send_val, 1, MPI_INT, 1-rank, tag, MPI_COMM_WORLD);

    // This line will not be reached because both are blocked in the Ssend
    MPI_Recv(&recv_val, 1, MPI_INT, 1-rank, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    printf("Rank %d: received %d\n", rank, recv_val);

    MPI_Finalize();
    return 0;
}
