// Copyright (C) 2018 Alain Miniussi

// Use, modification and distribution is subject to the Boost Software
// License, Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)

#include <boost/mpi/status.hpp>
#include <boost/mpi/exception.hpp>

namespace boost { namespace mpi {

namespace detail {
status
make_empty_status() {
  MPI_Status stat;
  MPI_Request req = MPI_REQUEST_NULL;
  BOOST_MPI_CHECK_RESULT(MPI_Wait, (&req, &stat));
  return status(stat);
}
}
status const&
status::empty_status() {
  static status empty = detail::make_empty_status();
  return empty;
}
}}
