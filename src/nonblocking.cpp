//          Copyright Alain Miniussi 2014.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE_1_0.txt or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

// Authors: Alain Miniussi

#include <boost/mpi/nonblocking.hpp>

namespace boost { namespace mpi {
std::pair<status, std::vector<request>::iterator>
wait_any(std::vector<request>::iterator begin,
         std::vector<request>::iterator end) {
  
  typedef std::vector<request>::iterator iter_t;
  
  BOOST_ASSERT(begin < end);
  bool all_trivial_requests = true;
  iter_t current = begin;
  while (true) {
    // Trivial request are processed below.
    if (!current->trivial()) {
      // Skip if already completed
      if (!current->null_requests()) {
        optional<status> result = current->test();
        // completed:
        if (result) {
          return std::make_pair(*result, current);
        } else {
          // some non trivial works is left to do
          all_trivial_requests = false;
        }
      }
    }
    // Move to the next request.
    ++current;
    // Check if this request (and all others before it) are "trivial"
    // requests, e.g., they can be represented with a single
    // MPI_Request.
    if (current == end) {
      // We have reached the end of the list. If all requests thus far
      // have been trivial, or processed, so we can call MPI_Waitany directly.
      if (all_trivial_requests) {
        std::vector<MPI_Request> requests(end - begin);
        std::vector<MPI_Request>::iterator mpi_iter = requests.begin();
        for (iter_t rit = begin; rit != end; ++rit) {
          *mpi_iter++ = rit->m_handler->request(0);
        }
        // Let MPI wait until one of these operations completes.
        int index;
        status stat;
        BOOST_MPI_CHECK_RESULT(MPI_Waitany, 
                               (end-begin, requests.data(), &index, &stat.m_status));
        
        // We don't have a notion of empty requests or status objects,
        // so this is an error.
        if (index == MPI_UNDEFINED) {
          boost::throw_exception(exception("MPI_Waitany", MPI_ERR_REQUEST));
        }
        // Find the iterator corresponding to the completed request.
        current = begin + index;
        current->m_handler->request(0) = requests[index];
        return std::make_pair(stat, current);
      }

      // There are some nontrivial requests, so we must continue our
      // busy waiting loop.
      current = begin;
      all_trivial_requests = true;
    }
  }

  // We cannot ever get here
  BOOST_ASSERT(false);
}
}}
