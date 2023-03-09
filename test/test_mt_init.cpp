// Copyright (C) 2013 Alain Miniussi <alain.miniussi@oca.eu>

// Use, modification and distribution is subject to the Boost Software
// License, Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)

// test threading::level operations

#include <boost/mpi.hpp>
#include <iostream>
#include <sstream>

#include "mpi_test_utils.hpp"

namespace mpi = boost::mpi;

int
test_mt_init(std::string s)
{
  int failed = 0;
  mpi::threading::level required = mpi::threading::level(-1);
  std::istringstream in(s);
  in >> required;
  MPI_CHECK(!in.bad(), failed);
  MPI_CHECK(mpi::environment::thread_level() >= mpi::threading::single, failed);
  MPI_CHECK(mpi::environment::thread_level() <= mpi::threading::multiple, failed);
  return failed;
}

int main()
{
  int failed = 0;
  mpi::environment env;
  mpi::communicator comm;
  MPI_FAILED_CHECK(test_mt_init("single"), failed);
  MPI_FAILED_CHECK(test_mt_init("funneled"), failed);
  MPI_FAILED_CHECK(test_mt_init("serialized"), failed);
  MPI_FAILED_CHECK(test_mt_init("multiple"), failed);
  return failed;
}
