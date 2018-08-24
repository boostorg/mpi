#include <boost/mpi/nonblocking.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/serialization/string.hpp>
#include <iterator>
#include <algorithm>

int main(int argc, char* argv[])
{
  MPI_Init(&argc, &argv);
  int rank, nproc;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);
  int value = 42;
  int input;
  int next = (rank + 1) % nproc;
  int prev = (rank + nproc - 1) % nproc;
  int tag = 2;
  MPI_Request sreq;
  MPI_Isend(&value, 1, MPI_INT, next, tag, MPI_COMM_WORLD, &sreq);
  int probe = 0;
  int test  = 0;
  MPI_Message msg;
  do {
    if (!test) {
      MPI_Test(&sreq, &test, MPI_STATUS_IGNORE);
      if (test) {
        printf("Proc %i sent msg %i to Proc %i\n", rank, tag, next);
      } else {
        printf("Proc %i have not sent msg %i to Proc %i yet\n", rank, tag, next);
      }
    }
    if (!probe) {
      int err = MPI_Improbe(prev, tag,
                            MPI_COMM_WORLD, &probe,
                            &msg,
                            MPI_STATUS_IGNORE);
      if (probe) 
        printf("Proc %i got msg %i from proc %i\n", rank, tag, prev);
      else 
        printf("Proc %i haven't got msg %i from proc %i yet\n", rank, tag, prev);
    }
  } while(probe == 0 || test == 0);
  MPI_Finalize();
  return 0;
}
