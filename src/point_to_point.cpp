// Copyright 2005 Douglas Gregor.

// Use, modification and distribution is subject to the Boost Software
// License, Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)

// Message Passing Interface 1.1 -- Section 3. MPI Point-to-point

/* There is the potential for optimization here. We could keep around
   a "small message" buffer of size N that we just receive into by
   default. If the message is N - sizeof(int) bytes or smaller, it can
   just be sent with that buffer. If it's larger, we send the first N
   - sizeof(int) bytes in the first packet followed by another
   packet. The size of the second packet will be stored in an integer
   at the end of the first packet.

   We will introduce this optimization later, when we have more
   performance test cases and have met our functionality goals. */

#include <boost/mpi/detail/point_to_point.hpp>
#include <boost/mpi/datatype.hpp>
#include <boost/mpi/exception.hpp>
#include <boost/mpi/request.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi/detail/antiques.hpp>
#include <cassert>

namespace boost { namespace mpi { namespace detail {

void
packed_archive_send(communicator const& comm, int dest, int tag,
                    const packed_oarchive& ar)
{
  std::size_t const& size = ar.size();
  BOOST_MPI_CHECK_RESULT(MPI_Send,
                         (detail::unconst(&size), 1, 
                          get_mpi_datatype(size), 
                          dest, tag, comm));
  BOOST_MPI_CHECK_RESULT(MPI_Send,
                         (detail::unconst(ar.address()), size,
                          MPI_PACKED,
                          dest, tag, comm));
}

request
packed_archive_isend(communicator const& comm, int dest, int tag,
                     const packed_oarchive& ar)
{
  request req = request::make_dynamic();
  std::size_t const& size = ar.size();
  BOOST_MPI_CHECK_RESULT(MPI_Isend,
                         (detail::unconst(&size), 1, 
                          get_mpi_datatype(size),
                          dest, tag, comm, &req.size_request()));
  BOOST_MPI_CHECK_RESULT(MPI_Isend,
                         (detail::unconst(ar.address()), size,
                          MPI_PACKED,
                          dest, tag, comm, &req.payload_request()));
  
  return req;
}

request
packed_archive_isend(communicator const& comm, int dest, int tag,
                     const packed_iarchive& ar)
{
  request req = request::make_dynamic();
  std::size_t const& size = ar.size();
  BOOST_MPI_CHECK_RESULT(MPI_Isend,
                         (detail::unconst(&size), 1, 
                          get_mpi_datatype(size), 
                          dest, tag, comm, &req.size_request()));
  BOOST_MPI_CHECK_RESULT(MPI_Isend,
                         (detail::unconst(ar.address()), size,
                          MPI_PACKED,
                          dest, tag, comm, &req.payload_request()));

  return req;
}

void
packed_archive_recv(communicator const& comm, int source, int tag, packed_iarchive& ar,
                    MPI_Status& status)
{
  std::size_t count;
  BOOST_MPI_CHECK_RESULT(MPI_Recv,
                         (&count, 1, get_mpi_datatype(count),
                          source, tag, comm, &status));

  // Prepare input buffer and receive the message
  ar.resize(count);
  BOOST_MPI_CHECK_RESULT(MPI_Recv,
                         (ar.address(), count, MPI_PACKED,
                          status.MPI_SOURCE, status.MPI_TAG,
                          comm, &status));
}

} } } // end namespace boost::mpi::detail
