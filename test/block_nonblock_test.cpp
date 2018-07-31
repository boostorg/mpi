#include <vector>
#include <iostream>
#include <iterator>
#include <boost/mpi.hpp>
#include <boost/serialization/vector.hpp>

namespace mpi = boost::mpi;

bool test(mpi::communicator const& comm, std::vector<int> const& ref, bool iswap, bool alloc)
{
  int rank = comm.rank();
  if (rank == 0) {
    if (iswap) {
      std::cout << "Blockin send, non blocking receive.\n";
    } else {
      std::cout << "Non blockin send, blocking receive.\n";
    }
    if (alloc) {
      std::cout << "Explicitly allocate space for the receiver.\n";
    } else {
      std::cout << "Do not explicitly allocate space for the receiver.\n";
    }
  }
  if (rank == 0) {
    std::vector<int> data;
    if (alloc) {
      data.resize(ref.size());
    }
    if (iswap) {
      mpi::request req = comm.irecv(1, 0, data);
      req.wait();
    } else {
      comm.recv(1, 0, data);
    }
    std::cout << "Process 0 received:" << std::endl;
    std::copy(data.begin(), data.end(), std::ostream_iterator<int>(std::cout, " "));
    std::cout << std::endl;
    std::cout << "While expecting:" << std::endl;
    std::copy(ref.begin(),  ref.end(),  std::ostream_iterator<int>(std::cout, " "));
    std::cout << std::endl;
    return (data == ref);
  } else {
    if (rank == 1) {
      std::vector<int> vec = ref;
      if (iswap) {
        comm.send(0, 0, vec);
      } else {
        mpi::request req = comm.isend(0, 0, vec);
        req.wait();
      }
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
  bool send_alloc  = test(world, ref, true,  true);
  bool isend_alloc = test(world, ref, false, true);
  bool send  = test(world, ref, true,  false);
  bool isend = test(world, ref, false, false);
  bool local_passed = send && isend && send_alloc && isend_alloc;
  bool passed = mpi::all_reduce(world, local_passed, std::logical_and<bool>());
  return passed ? 0 : 1;
}
