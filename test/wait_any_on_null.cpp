// Copyright (C) 2021 Steffen Hirschmann

// Use, modification and distribution is subject to the Boost Software
// License, Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)

#include <boost/mpi.hpp>
#include <boost/mpi/nonblocking.hpp>
#include <vector>

#define BOOST_TEST_NO_MAIN
#define BOOST_TEST_ALTERNATIVE_INIT_API
#define BOOST_TEST_MODULE mpi_wait_on_null
#include <boost/test/included/unit_test.hpp>

/** Check default constructed requests.
 * Must not deadlock.
 */
BOOST_AUTO_TEST_CASE(wait_any_default_constructed_request)
{
  std::vector<boost::mpi::request> req(1);
  boost::mpi::status s;
  decltype(req)::iterator it;
  std::tie(s, it) = boost::mpi::wait_any(req.begin(), req.end());
  BOOST_CHECK(it == req.end());
}

/** Check a trivial request twice.
 */
BOOST_AUTO_TEST_CASE(wait_any_all_trivial_and_done)
{
  boost::mpi::communicator comm;
  std::vector<boost::mpi::request> req;
  int dummy1 = 1, dummy2 = 0;

  req.push_back(comm.irecv(comm.rank(), 0, dummy2));
  comm.isend(comm.rank(), 0, dummy1);

  boost::mpi::status s;
  decltype(req)::iterator it;
  // All trivial requests
  std::tie(s, it) = boost::mpi::wait_any(req.begin(), req.end());
  BOOST_CHECK(it != req.end());
  BOOST_CHECK(it == req.begin());
  BOOST_CHECK(s.count<int>());
  BOOST_CHECK(*s.count<int>() == 1);

  // Call a second time
  std::tie(s, it) = boost::mpi::wait_any(req.begin(), req.end());
  BOOST_CHECK(it == req.end());
  BOOST_CHECK(s.count<int>());
  // empty status according to MPI 3.1 §3.7.3, l.39 (p. 52)
  BOOST_CHECK(*s.count<int>() == 0);
  BOOST_CHECK(!s.cancelled());
  BOOST_CHECK(s.source() == MPI_ANY_SOURCE);
  BOOST_CHECK(s.tag() == MPI_ANY_TAG);
  BOOST_CHECK(s.error() == MPI_SUCCESS);
}

int main(int argc, char **argv)
{
    boost::mpi::environment env(argc, argv);
    return boost::unit_test::unit_test_main(init_unit_test, argc, argv);
}
