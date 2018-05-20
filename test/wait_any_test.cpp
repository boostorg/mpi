// Copyright (C) 2017 Alain Miniussi & Steffen Hirschmann 

// Use, modification and distribution is subject to the Boost Software
// License, Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)

#include <string>
#include <iostream>
#include <sstream>
#include <vector>
#include <set>

#include <boost/mpi.hpp>
#include <boost/mpi/nonblocking.hpp>
#include <boost/serialization/string.hpp>
#include <boost/test/minimal.hpp>

namespace mpi = boost::mpi;

int test_main(int argc, char* argv[]) 
{
  mpi::environment env(argc, argv);
  mpi::communicator world;
  
  std::vector<std::string> ss(world.size());
  typedef std::vector<mpi::request> requests;
  requests rreqs;
  
  std::set<int> pending_senders;
  for (int i = 0; i < world.size(); ++i) {
    rreqs.push_back(world.irecv(i, i, ss[i]));
    pending_senders.insert(i);
  }
  
  std::ostringstream fmt;
  std::string msg = "Hello, World! this is ";
  fmt << msg << world.rank();

  requests sreqs;
  for (int i = 0; i < world.size(); ++i) {
    sreqs.push_back(world.isend(i, world.rank(), fmt.str()));
  }
  
  for (int i = 0; i < world.size(); ++i) {
    mpi::status status;
    decltype(rreqs.begin()) it;
    std::tie(status, it) = mpi::wait_any(std::begin(rreqs), std::end(rreqs));
    int sender = status.source();
    std::ostringstream out;
    out << "Proc " << world.rank() << " got message from " << status.source() << '\n';
    std::cout << out.str();
  }
  
  for (int i = 0; i < world.size(); ++i) {
    std::ostringstream fmt;
    fmt << msg << i;
    auto found = std::find(ss.begin(), ss.end(), fmt.str());
    BOOST_CHECK(found != ss.end());
    fmt.str("");
    fmt << "Proc " << world.rank() << " Got msg from " << i << '\n';
    std::cout << fmt.str();
  }

  mpi::wait_all(std::begin(sreqs), std::end(sreqs));

  return 0;
}
