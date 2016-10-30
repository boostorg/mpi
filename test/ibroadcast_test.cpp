// Copyright (C) 2005, 2006 Douglas Gregor.

// Use, modification and distribution is subject to the Boost Software
// License, Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)

// A test of the ibroadcast() collective.

#include <algorithm>
#include <boost/mpi/collectives/ibroadcast.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <boost/serialization/string.hpp>
#include <boost/serialization/list.hpp>
#include <boost/test/minimal.hpp>

#include "gps_position.hpp"
//#include "debugger.hpp"

namespace mpi = boost::mpi;

template<typename T>
void
ibroadcast_test(mpi::communicator const& comm, T const& bc_value,
                std::string kind, int root)
{
  T value;
  if (comm.rank() == root) {
    value = bc_value;
    std::cout << "Broadcasting " << kind << " from root " << root << "...";
    std::cout.flush();
  }
  mpi::request req = mpi::ibroadcast(comm, value, root);
  req.wait();
  BOOST_CHECK(value == bc_value);
  if (comm.rank() == root && value == bc_value) {
    std::cout << "OK." << std::endl;
  }
}

template<typename T>
void
ibroadcast_test_all(mpi::communicator const& comm, T const& bc_value,
                    std::string kind) 
{
  for(int root = 0; root < comm.size(); ++root) {
    ibroadcast_test(comm, bc_value, kind, root);
    comm.barrier();
  }
}

int test_main(int argc, char* argv[])
{
  mpi::environment env(argc, argv);

  mpi::communicator comm;
  if (comm.size() == 1) {
    std::cerr << "ERROR: Must run the broadcast test with more than one "
              << "process." << std::endl;
    comm.abort(-1);
  }

  //wait_for_debugger(extract_paused_ranks(argc, argv), comm);

  // Check transfer of individual objects
  ibroadcast_test_all(comm, 17, "integers");
  ibroadcast_test_all(comm, gps_position(39,16,20.2799), "GPS positions");
  ibroadcast_test_all(comm, gps_position(26,25,30.0), "GPS positions");
  ibroadcast_test_all(comm, std::string("Rosie"), "string");

  std::list<std::string> strings;
  strings.push_back("Hello");
  strings.push_back("MPI");
  strings.push_back("World");
  ibroadcast_test_all(comm, strings, "list of strings");

  return 0;
}
