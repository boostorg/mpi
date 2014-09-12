
//          Copyright Alain Miniussi 2014.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE_1_0.txt or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

// Authors: Alain Miniussi

#include <vector>
#include <iostream>
#include <sstream>
#include <iterator>
#include <algorithm>
#include <functional>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/collectives.hpp>
#include <boost/mpi/environment.hpp>
#include <boost/mpi/cartesian_communicator.hpp>

#include <boost/test/minimal.hpp>

namespace mpi = boost::mpi;

struct topo_minimum {
  mpi::cartesian_dimension
  operator()(mpi::cartesian_dimension const& d1,
             mpi::cartesian_dimension const& d2 ) const {
    return mpi::cartesian_dimension(std::min(d1.size, d2.size),
                                    d1.periodic && d2.periodic);
  }
};

std::string topology_description( mpi::cartesian_topology const&  topo ) {
  std::ostringstream out;
  std::copy(topo.begin(), topo.end(), std::ostream_iterator<mpi::cartesian_dimension>(out, " "));
  out << std::flush;
  return out.str();
}

// Check that everyone agrees on the coordinates
void test_coordinates_consistency( mpi::cartesian_communicator const& cc,
                                   std::vector<int> const& coords )
{
  for(int p = 0; p < cc.size(); ++p) {
    std::vector<int> min(cc.ndims());
    std::vector<int> local(cc.coords(p));
    mpi::reduce(cc, local.data(), local.size(),
                min.data(), mpi::minimum<int>(), p);
    if (p == cc.rank()) {
      BOOST_CHECK(std::equal(coords.begin(), coords.end(), 
                             min.begin()));
      std::ostringstream out;
      out << "proc " << p << " at (";
      std::copy(min.begin(), min.end(), std::ostream_iterator<int>(out, " "));
      out << ")\n";
      std::cout << out.str();
    }
  }
}

void test_topology_consistency( mpi::cartesian_communicator const& cc) 
{
  mpi::cartesian_topology itopo(cc.ndims());
  mpi::cartesian_topology otopo(cc.ndims());
  std::vector<int>  coords(cc.ndims());
  cc.topology(itopo, coords);

  // Check that everyone agrees on the dimensions
  mpi::all_reduce(cc, 
                  &(itopo[0]), itopo.size(), &(otopo[0]),
                  topo_minimum());
  BOOST_CHECK(std::equal(itopo.begin(), itopo.end(), 
                         otopo.begin()));
  if (cc.rank() == 0) {
    std::cout << topology_description(otopo) << '\n';
  }
  test_coordinates_consistency( cc, coords );
}

int test_main(int argc, char* argv[])
{
  mpi::environment env(argc, argv);

  mpi::communicator world;
  mpi::cartesian_topology topo(3);

  if (world.size() == 24) {
    topo[0].size = 2; topo[1].size = 3; topo[2].size = 4;
  } else {
    topo[0].size = 0; topo[1].size = 3; topo[2].size = 0;
  }
  topo[0].periodic = true; topo[1].periodic = false; topo[2].periodic = true;
  
  mpi::cartesian_communicator cc(world, topo, true);
  BOOST_CHECK(cc.has_cartesian_topology());
  BOOST_CHECK(cc.ndims() == 3);
  for( int r = 0; r < cc.size(); ++r) {
    cc.barrier();
    if (r == cc.rank()) {
      std::vector<int> coords = cc.coords(r);
      std::cout << "Process of cartesian rank " << cc.rank() 
                << " and global rank " << world.rank() 
                << " has coordinates (";
      std::copy(coords.begin(), coords.end(), std::ostream_iterator<int>(std::cout,","));
      std::cout << ")\n";
    }
  }
  test_topology_consistency(cc);
  std::vector<int> sub02;
  sub02.push_back(0);
  sub02.push_back(2);
  mpi::cartesian_communicator cc02(cc, sub02);
  test_topology_consistency(cc02);
  return 0;
}
