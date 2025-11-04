#include <mpi.h>
#include <stdio.h>

int main(int argc, char** argv) {
    MPI_Init(&argc,&argv);

    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    MPI_Group world_group, five_group;
    MPI_Comm newcomm = MPI_COMM_NULL;

    MPI_Comm_group(MPI_COMM_WORLD, &world_group);

    if (world_size >= 5) {
        int ranks[5] = {0,1,2,3,4};
        MPI_Group_incl(world_group, 5, ranks, &five_group);
        MPI_Comm_create_group(MPI_COMM_WORLD, five_group, 0, &newcomm);
        MPI_Group_free(&five_group);
    }

    if (newcomm != MPI_COMM_NULL) {
        int new_rank, new_size;
        MPI_Comm_rank(newcomm, &new_rank);
        MPI_Comm_size(newcomm, &new_size);
        printf("World rank %d -> In new communicator: rank %d of %d\n",
               world_rank, new_rank, new_size);
        MPI_Comm_free(&newcomm);
    } else {
        printf("World rank %d -> NOT in the new communicator (outside first 5)\n", world_rank);
    }

    MPI_Group_free(&world_group);
    MPI_Finalize();
    return 0;
}
