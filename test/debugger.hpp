//          Copyright AlainMiniussi 20014 - 20015.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE_1_0.txt or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <vector>
#include "boost/mpi/communicator.hpp">

/*
#include "boost/mpi/debugger.hpp"
  std::vector<int> processes;
  for (int i=1; i < argc; ++i) {
    processes.push_back(atoi(argv[i]));
  }
  wait_for_debugger(processes, comm);
*/
void wait_for_debugger(std::vector<int> const& processes, boost::mpi::communicator comm);

