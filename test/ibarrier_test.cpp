//          Copyright Alain Miniussi 2014.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE_1_0.txt or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

// A test of the non blocking barrier operation.

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>


#define BOOST_TEST_MODULE mpi_ibarrier
#include <boost/test/included/unit_test.hpp>

namespace mpi = boost::mpi;

BOOST_AUTO_TEST_CASE(ibarrier_check) 
{
  mpi::environment  env;
  mpi::communicator world;

  std::ostringstream buf;  
  int rk = world.rank();
  buf << "rk" << rk << ": calling ibarrier.\n";
  std::cout << buf.str();
  mpi::request r = world.ibarrier();
  if (rk == 0) {
    while (!r.test()) {
      buf << "rk" << rk << ": not completed yet.\n";
      std::cout << buf.str();
    }
    buf << "rk" << rk << ": completed.\n";
    std::cout << buf.str();
  } else {
    buf << "rk" << rk << ": waiting...";
    std::cout << buf.str() << std::flush;
    r.wait();
    buf << "rk" << rk << ": done.\n";
    std::cout << buf.str();
  }
  BOOST_TEST(true);  
}
