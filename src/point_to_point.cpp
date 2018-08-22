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
#include <boost/mpi/detail/antiques.hpp>
#include <cassert>

namespace boost { namespace mpi { namespace detail {

void
packed_archive_send(MPI_Comm comm, int dest, int tag,
                    const packed_oarchive& ar)
{
  BOOST_MPI_CHECK_RESULT(MPI_Send,
                         (detail::unconst(ar.address()), ar.size(),
                          MPI_PACKED,
                          dest, tag, comm));
}

int
packed_archive_isend(MPI_Comm comm, int dest, int tag,
                     const packed_oarchive& ar,
                     MPI_Request& out_requests)
{
  BOOST_MPI_CHECK_RESULT(MPI_Isend,
                         (detail::unconst(ar.address()), ar.size(),
                          MPI_PACKED,
                          dest, tag, comm, &out_requests));

  return 1;
}

int
packed_archive_isend(MPI_Comm comm, int dest, int tag,
                     const packed_iarchive& ar,
                     MPI_Request& out_requests)
{
  BOOST_MPI_CHECK_RESULT(MPI_Isend,
                         (detail::unconst(ar.address()), ar.size(),
                          MPI_PACKED,
                          dest, tag, comm, &out_requests));

  return 1;
}

void
packed_archive_recv(MPI_Comm comm, int source, int tag, packed_iarchive& ar,
                    MPI_Status& status)
{
  MPI_Message message;
  BOOST_MPI_CHECK_RESULT(MPI_Mprobe,
                         (source, tag, comm, 
                          &message, &status));
  int count;
  BOOST_MPI_CHECK_RESULT(MPI_Get_count, (&status, MPI_PACKED, &count));
  ar.resize(count);
  BOOST_MPI_CHECK_RESULT(MPI_Mrecv,
                         (ar.address(), count, MPI_PACKED,
                          &message, &status));
}

} } } // end namespace boost::mpi::detail
