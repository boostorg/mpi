
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

struct bvecmin {
  std::vector<bool> operator()(std::vector<bool> const& v1,
                               std::vector<bool> const& v2) const {
    BOOST_ASSERT(v1.size() == v2.size());
    std::vector<bool> res(v1.size());
    std::transform(v1.begin(), v1.end(), v2.begin(), res.begin(),
                   std::logical_and<bool>());
    return res;
  }
};

std::string topology_description( std::vector<int> const&  dims,
                                  std::vector<bool> const& periodic ) {
  std::ostringstream out;
  for(int i = 0; i < dims.size(); ++i) {
    out << "(" << dims[i] << ',';
    if (periodic[i]) {
      out << "cyclic";
    } else {
      out << "bounded";
    }
    out << ") ";
  }
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
  std::vector<int>  idims(cc.ndims());
  std::vector<bool> iperiodic(cc.ndims());
  std::vector<int>  odims(cc.ndims());
  std::vector<bool> operiodic;
  std::vector<int>  coords(cc.ndims());
  cc.topology(idims, iperiodic, coords);

  // Check that everyone agrees on the dimensions
  mpi::all_reduce(cc, 
                  &(idims[0]), idims.size(), &(odims[0]),
                  mpi::minimum<int>());
  BOOST_CHECK(std::equal(idims.begin(), idims.end(), 
                         odims.begin()));
  // Check that everyone agree on the periodicities
  mpi::all_reduce(cc, iperiodic, operiodic, bvecmin());
  BOOST_CHECK(std::equal(iperiodic.begin(), iperiodic.end(), 
                         operiodic.begin()));
  if (cc.rank() == 0) {
    std::cout << topology_description(odims, operiodic) << '\n';
  }
  test_coordinates_consistency( cc, coords );
}

int test_main(int argc, char* argv[])
{
  mpi::environment env(argc, argv);

  mpi::communicator world;
  std::vector<int>  dims(3);
  std::vector<bool> periodic(3);
  if (world.size() == 24) {
    dims[0] = 2; dims[1] = 3; dims[2] = 4;
  } else {
    dims[0] = 0; dims[1] = 0; dims[2] = 0;
  }
  periodic[0] = true; periodic[1] = false; periodic[2] = true;
  
  mpi::cartesian_communicator cc(world, dims, periodic, true);
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
