#include <mpi.h>
#include <stdio.h>

int main(int argc, char** argv) {
    MPI_Init(&argc,&argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&size);

    int send_val = rank;                    // send my rank
    int recv_val = -1;
    int left = (rank - 1 + size) % size;    // from which I receive
    int right = (rank + 1) % size;          // to which I send

    MPI_Sendrecv(&send_val, 1, MPI_INT, right, 0,
                 &recv_val, 1, MPI_INT, left, 0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    printf("Rank %d received %d from rank %d\n", rank, recv_val, left);

    MPI_Finalize();
    return 0;
}
