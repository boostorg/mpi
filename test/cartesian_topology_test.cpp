// Copyright (C) 2007 Trustees of Indiana University

// Authors: Alain Miniussi

// Use, modification and distribution is subject to the Boost Software
// License, Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)

#include <vector>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <boost/mpi/cartesian_communicator.hpp>

#include <boost/test/minimal.hpp>

namespace mpi = boost::mpi;

int test_main(int argc, char* argv[])
{
  mpi::environment env(argc, argv);

  mpi::communicator world;
  std::vector<int>  dims(3);
  std::vector<bool> periodic(3);
  dims[0] = 2; dims[1] = 3; dims[2] = 4;
  periodic[0] = true; periodic[1] = false; periodic[2] = true;
  
  mpi::cartesian_communicator cc(world, dims, periodic, true);
  BOOST_CHECK(cc.has_cartesian_topology());
    
  return 0;
}
