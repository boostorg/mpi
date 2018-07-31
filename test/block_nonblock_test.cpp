#include <vector>
#include <iostream>
#include <iterator>
#include <boost/mpi.hpp>
#include <boost/serialization/vector.hpp>

namespace mpi = boost::mpi;

bool test(mpi::communicator const& comm, std::vector<int> const& ref, bool iswap)
{
  int rank = comm.rank();
  
  if (rank == 0) {
    std::vector<int> data;
    if (iswap) {
      auto req = comm.irecv(1, 0, data);
      req.wait();
    } else {
      comm.recv(1, 0, data);
    }
    std::cout << "Process 0 received:" << std::endl;
    std::copy(std::begin(data), std::end(data), std::ostream_iterator<int>(std::cout, " "));
    std::cout << std::endl;
    std::cout << "While expecting:" << std::endl;
    std::copy(std::begin(ref), std::end(ref), std::ostream_iterator<int>(std::cout, " "));
    std::cout << std::endl;
    return (data == ref);
  } else if (rank == 1) {
    std::vector<int> vec = ref;
    if (iswap) {
      comm.send(0, 0, vec);
    } else {
      auto req = comm.isend(0, 0, vec);
      req.wait();
    }
    return true;
  } 
}

int main(int argc, char **argv)
{
  mpi::environment env(argc, argv);
  mpi::communicator world;
  std::vector<int> ref(13); // don't assume we're lucky
  for(int i = 0; i < int(ref.size()); ++i) {
    ref[i] = i;
  }
  bool send  = test(world, ref, true);
  bool isend = test(world, ref, false);
  return send && isend ? 0 : 1;
}
