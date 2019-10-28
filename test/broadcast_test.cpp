// Copyright (C) 2005, 2006 Douglas Gregor.

// Use, modification and distribution is subject to the Boost Software
// License, Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)

// A test of the broadcast() collective.
#include <boost/mpi/collectives/broadcast.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <algorithm>
#include "gps_position.hpp"
#include <boost/serialization/string.hpp>
#include <boost/serialization/list.hpp>
#include <boost/mpi/skeleton_and_content.hpp>
#include <boost/iterator/counting_iterator.hpp>
//#include "debugger.hpp"

#define BOOST_TEST_MODULE mpi_broadcast
#include <boost/test/included/unit_test.hpp>

namespace mpi = boost::mpi;

template<typename T>
void
broadcast_test(mpi::communicator const& comm, T const& bc_value,
               char const* kind, int root = -1)
{
  if (root == -1) {
    for (root = 0; root < comm.size(); ++root)
      broadcast_test(comm, bc_value, kind, root);
  } else {
    T value;
    if (comm.rank() == root) {
      value = bc_value;
      std::cout << "Broadcasting " << kind << " from root " << root << "...";
      std::cout.flush();
    }
    
    mpi::broadcast(comm, value, root);
    BOOST_CHECK(value == bc_value);
    if (comm.rank() == root && value == bc_value)
      std::cout << "OK." << std::endl;
  }

  comm.barrier();
}

template<typename T>
void
ibroadcast_test(mpi::communicator const& comm, T const& bc_value,
		char const* kind, int root = -1)
{
  if (root == -1) {
    for (root = 0; root < comm.size(); ++root)
      ibroadcast_test(comm, bc_value, kind, root);
  } else {
    T value;
    if (comm.rank() == root) {
      value = bc_value;
      std::cout << "Broadcasting " << kind << " from root " << root << "...";
      std::cout.flush();
    }

    mpi::request req = mpi::ibroadcast(comm, value, root);
    std::ostringstream buf;
    buf << "rk" << comm.rank() << ": Broadcasting " << value << " from " << root << "...";
    if (!req.test()) {
      buf << ".. not finished here. So we wait...";
      req.wait();
      buf << "done.\n";
    } else {
      buf << ".. which is already finished.\n";
    }
    std::cout << buf.str();
    BOOST_CHECK(value == bc_value);
    if (comm.rank() == root && value == bc_value)
      std::cout << "OK." << std::endl;
  }
}

void
test_skeleton_and_content(mpi::communicator const& comm, int root = 0)
{
  using boost::make_counting_iterator;

  int list_size = comm.size() + 7;
  if (comm.rank() == root) {
    // Fill in the seed data
    std::list<int> original_list;
    for (int i = 0; i < list_size; ++i)
      original_list.push_back(i);

    // Build up the skeleton
    mpi::packed_skeleton_oarchive oa(comm);
    oa << original_list;

    // Broadcast the skeleton
    std::cout << "Broadcasting integer list skeleton from root " << root
              << "..." << std::flush;
    mpi::broadcast(comm, oa, root);
    std::cout << "OK." << std::endl;

    // Broadcast the content
    std::cout << "Broadcasting integer list content from root " << root
              << "..." << std::flush;
    {
      mpi::content c = mpi::get_content(original_list);
      mpi::broadcast(comm, c, root);
    }
    std::cout << "OK." << std::endl;

    // Reverse the list, broadcast the content again
    std::reverse(original_list.begin(), original_list.end());
    std::cout << "Broadcasting reversed integer list content from root "
              << root << "..." << std::flush;
    {
      mpi::content c = mpi::get_content(original_list);
      mpi::broadcast(comm, c, root);
    }
    std::cout << "OK." << std::endl;

  } else {
    // Allocate some useless data, to try to get the addresses of the
    // list<int>'s used later to be different across processes.
    std::list<int> junk_list(comm.rank() * 3 + 1, 17);

    // Receive the skeleton
    mpi::packed_skeleton_iarchive ia(comm);
    mpi::broadcast(comm, ia, root);

    // Build up a list to match the skeleton, and make sure it has the
    // right structure (we have no idea what the data will be).
    std::list<int> transferred_list;
    ia >> transferred_list;
    BOOST_CHECK(int(transferred_list.size()) == list_size);

    // Receive the content and check it
    mpi::broadcast(comm, mpi::get_content(transferred_list), root);
    bool list_content_ok = std::equal(make_counting_iterator(0),
				      make_counting_iterator(list_size),
				      transferred_list.begin());
    BOOST_CHECK(list_content_ok);

    // Receive the reversed content and check it
    mpi::broadcast(comm, mpi::get_content(transferred_list), root);
    bool rlist_content_ok = std::equal(make_counting_iterator(0),
				       make_counting_iterator(list_size),
				       transferred_list.rbegin());
    BOOST_CHECK(rlist_content_ok);
    if (!(list_content_ok && rlist_content_ok)) {
      if (comm.rank() == 1) {
	std::cout
	  << "\n##### You might want to check for BOOST_MPI_BCAST_BOTTOM_WORKS_FINE "
	  << "in boost/mpi/config.hpp.\n\n";
      }
    }
  }

  comm.barrier();
}

BOOST_AUTO_TEST_CASE(broadcast_check)
{
  boost::mpi::environment env;
  mpi::communicator comm;

  BOOST_TEST_REQUIRE(comm.size() > 1);

  // Check transfer of individual objects
  broadcast_test(comm, 17, "integers");
  ibroadcast_test(comm, 17, "integers");
  broadcast_test(comm, gps_position(39,16,20.2799), "GPS positions");
  broadcast_test(comm, gps_position(26,25,30.0), "GPS positions");
  broadcast_test(comm, std::string("Rosie"), "string");

  std::list<std::string> strings;
  strings.push_back("Hello");
  strings.push_back("MPI");
  strings.push_back("World");
  broadcast_test(comm, strings, "list of strings");

  test_skeleton_and_content(comm, 0);
  test_skeleton_and_content(comm, 1);
}
