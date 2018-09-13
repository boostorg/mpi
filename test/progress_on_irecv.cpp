#include <boost/mpi.hpp>
#include <chrono>
#include <vector>
#include <boost/serialization/vector.hpp>

namespace mpi = boost::mpi;

int main()
{
  mpi::environment env;
  mpi::communicator world;
  
  if (world.rank() == 0) {
    // make sure message is large enough so it is not transmitted eagerly
    std::vector<int> msg0(100000);
    std::vector<int> msg1(1);
    world.send(1, 0, msg0);
    world.send(1, 1, msg1);
  } else {
    mpi::request req;
    std::vector<int> msgs[2];
    req = world.irecv(0, 0, msgs[0]);
    world.recv(0, 1, msgs[1]);
    req.wait();
  }
}
