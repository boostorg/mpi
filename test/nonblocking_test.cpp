#include <boost/mpi/nonblocking.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/serialization/string.hpp>
#include <iterator>
#include <algorithm>

using boost::mpi::communicator;
using boost::mpi::request;
using boost::mpi::status;

int main(int argc, char* argv[])
{
  boost::mpi::environment env(argc, argv);
  boost::mpi::communicator comm;
  std::string value = "Hello";
  std::string incoming;
  int next = (comm.rank() + 1) % comm.size();
  int prev = (comm.rank() + comm.size() - 1) % comm.size();
  int tag = 2;
  request sreq = comm.isend(next, tag, value);
  request rreq = comm.irecv(prev, tag, incoming);
  int probe = 0;
  int test  = 0;
  MPI_Message msg;
  do {
    if (!test) {
      MPI_Test(sreq.m_request.get(), &test, MPI_STATUS_IGNORE);
      if (test) {
        printf("Proc %i sent msg %i to Proc %i\n", comm.rank(), tag, next);
      } else {
        printf("Proc %i have not sent msg %i to Proc %i yet\n", comm.rank(), tag, next);
      }
    }
    if (!probe) {
      int err = MPI_Improbe(prev, tag,
                            comm, &probe,
                            &msg,
                            MPI_STATUS_IGNORE);
      if (probe) 
        printf("Proc %i got msg %i from proc %i\n", comm.rank(), tag, prev);
      else 
        printf("Proc %i haven't got msg %i from proc %i yet\n", comm.rank(), tag, prev);
    }
  } while(probe == 0 || test == 0);
  return 0;
}
