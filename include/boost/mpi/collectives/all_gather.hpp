// Copyright (C) 2005, 2006 Douglas Gregor.

// Use, modification and distribution is subject to the Boost Software
// License, Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)

// Message Passing Interface 1.1 -- Section 4.5. Gather
#ifndef BOOST_MPI_ALLGATHER_HPP
#define BOOST_MPI_ALLGATHER_HPP

#include <cassert>
#include <cstddef>
#include <numeric>
#include <boost/mpi/exception.hpp>
#include <boost/mpi/datatype.hpp>
#include <vector>
#include <boost/mpi/packed_oarchive.hpp>
#include <boost/mpi/packed_iarchive.hpp>
#include <boost/mpi/detail/point_to_point.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <boost/mpi/detail/offsets.hpp>
#include <boost/assert.hpp>

namespace boost { namespace mpi {

namespace detail {
// We're all-gathering for a type that has an associated MPI
// datatype, so we'll use MPI_Gather to do all of the work.
template<typename T>
void
all_gather_impl(const communicator& comm, const T* in_values, int n, 
                T* out_values, mpl::true_)
{
  MPI_Datatype type = get_mpi_datatype<T>(*in_values);
  BOOST_MPI_CHECK_RESULT(MPI_Allgather,
                         (const_cast<T*>(in_values), n, type,
                          out_values, n, type, comm));
}

// We're all-gathering for a type that does not have an
// associated MPI datatype, so we'll need to serialize
// it.
template<typename T>
void
all_gather_impl(const communicator& comm, const T* in_values, int n, 
               T* out_values, mpl::false_)
{
  int tag = environment::collectives_tag();
  int nproc = comm.size();
  // first, gather all size, these size can be different for
  // each process
  packed_oarchive oa(comm);
  for (int i = 0; i < n; ++i) {
    oa << in_values[i];
  }
  std::vector<int> oasizes(nproc);
  int oasize = oa.size();
  BOOST_MPI_CHECK_RESULT(MPI_Allgather,
                         (&oasize, 1, MPI_INTEGER,
                          oasizes.data(), 1, MPI_INTEGER, 
                          MPI_Comm(comm)));
  // Gather the archives, which can be of different sizes, so
  // we need to use allgatherv.
  // Every thing is contiguous, so the offsets can be
  // deduced from the collected sizes.
  std::vector<int> offsets(nproc);
  sizes2offsets(oasizes, offsets);
  packed_iarchive::buffer_type recv_buffer(std::accumulate(oasizes.begin(), oasizes.end(), 0));
  BOOST_MPI_CHECK_RESULT(MPI_Allgatherv,
                         (const_cast<void*>(oa.address()), int(oa.size()), MPI_BYTE,
                          recv_buffer.data(), oasizes.data(), offsets.data(), MPI_BYTE, 
                          MPI_Comm(comm)));
  for (int src = 0; src < nproc; ++src) {
    if (src == comm.rank()) { // this is our local data
      std::copy(in_values, in_values + n, out_values + n * src);
    } else {
      packed_iarchive ia(comm,  recv_buffer, boost::archive::no_header, offsets[src]);
      for (int i = 0; i < n; ++i) {
        ia >> out_values[n*src + i];
      }
    }
  }
}

} // end namespace detail

template<typename T>
void
all_gather(const communicator& comm, const T& in_value, T* out_values)
{
  detail::all_gather_impl(comm, &in_value, 1, out_values, is_mpi_datatype<T>());
}

template<typename T>
void
all_gather(const communicator& comm, const T& in_value, std::vector<T>& out_values)
{
  out_values.resize(comm.size());
  ::boost::mpi::all_gather(comm, in_value, out_values.data());
}

template<typename T>
void
all_gather(const communicator& comm, const T* in_values, int n, T* out_values)
{
  detail::all_gather_impl(comm, in_values, n, out_values, is_mpi_datatype<T>());
}

template<typename T>
void
all_gather(const communicator& comm, const T* in_values, int n, std::vector<T>& out_values)
{
  out_values.resize(comm.size() * n);
  ::boost::mpi::all_gather(comm, in_values, n, out_values.data());
}

} } // end namespace boost::mpi

#endif // BOOST_MPI_ALL_GATHER_HPP
