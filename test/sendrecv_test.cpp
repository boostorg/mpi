//          Copyright Alain Miniussi 20014.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE_1_0.txt or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

// A test of the sendrecv() operation.
#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <boost/test/minimal.hpp>
#include <vector>
#include <algorithm>
#include <boost/serialization/string.hpp>
#include <boost/iterator/counting_iterator.hpp>
#include <boost/lexical_cast.hpp>
#include <numeric>

namespace mpi = boost::mpi;

int test_main(int argc, char* argv[])
{

  mpi::environment env(argc, argv);
  mpi::communicator world;

  int const wrank = world.rank();
  int const wsize = world.size();
  int const wnext = (wrank + 1) % wsize;
  int const wprev = (wrank + wsize - 1) % wsize;
  int recv = -1;
  world.sendrecv(wnext, 1, wrank, wprev, 1, recv);
  for(int r = 0; r < wsize; ++r) {
    world.barrier();
    if (r == wrank) {
      std::cout << "rank " << wrank << " received " << recv << " from " << wprev << '\n';
    }
  }
  BOOST_CHECK(recv == wprev);
  return 0;
}
