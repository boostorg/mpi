// Copyright (C) 2007 Trustees of Indiana University

// Authors: Douglas Gregor
//          Andrew Lumsdaine

// Use, modification and distribution is subject to the Boost Software
// License, Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
#include <boost/mpi/exception.hpp>

namespace boost { namespace mpi {

void
exception::init() 
{
  // Query the MPI implementation for its reason for failure
  char buffer[MPI_MAX_ERROR_STRING];
  int len;
  MPI_Error_string(result_code_, buffer, &len);

  // Construct the complete error message
  message.append(routine_);
  message.append(": ");
  message.append(buffer, len);
}

exception::exception(const char* routine, int result_code)
  : routine_(routine), result_code_(result_code)
{
  init();
}

exception::exception(std::string const& routine, int result_code)
  : routine_(routine), result_code_(result_code)
{
  init();
}

exception::~exception() throw() { }

} } // end namespace boost::mpi
