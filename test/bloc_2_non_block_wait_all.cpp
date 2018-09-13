#include <boost/mpi.hpp>
#include <chrono>
#include <string>

namespace mpi = boost::mpi;

int main()
{
  mpi::environment env;
  mpi::communicator world;
  
  if (world.rank() == 0) {
    std::string msgs[2] = {"Hello", "World"};
    world.send(1, 0, msgs[0]);
    world.send(1, 0, msgs[1]);
  } else {
    mpi::request reqs[2];
    std::string msgs[2];
    reqs[0] = world.irecv(0, 0, msgs[0]);
    reqs[1] = world.irecv(0, 0, msgs[1]);
    mpi::wait_all(std::begin(reqs), std::end(reqs));
  }
}
