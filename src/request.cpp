// Copyright (C) 2006 Douglas Gregor.

// Use, modification and distribution is subject to the Boost Software
// License, Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
#include <boost/mpi/request.hpp>
#include <boost/mpi/status.hpp>

namespace boost { namespace mpi {

request::handler::handler()
{
}

request::handler::~handler()
{
}

request::request()
  : m_handler(new simple_handler()) 
{
}

request::request(handler* h) 
  : m_handler(h) 
{
}

request::simple_handler::simple_handler() 
{
}

request::simple_handler::~simple_handler() 
{
}

status
request::simple_handler::wait()
{
  // This request is either a send or a receive for a type with an
  // associated MPI datatype
  status result;
  BOOST_MPI_CHECK_RESULT(MPI_Wait, (&m_request, &result.m_status));
  return result;
}

optional<status> 
request::simple_handler::test()
{
  status result;
  int flag = 0;
  BOOST_MPI_CHECK_RESULT(MPI_Test, 
                         (&m_request, &flag, &result.m_status));
  return bool(flag) ? optional<status>(result) : optional<status>();
}

void
request::handler::cancel()
{
  int nreq = nb_requests();
  MPI_Request* first = requests();
  for (MPI_Request* r = first; r < first + nreq; ++r) {
    if (*r != MPI_REQUEST_NULL) {
      BOOST_MPI_CHECK_RESULT(MPI_Cancel, (r) );  
    }
  }    
}

bool
request::handler::null_requests() const
{
  int nreq = nb_requests();
  MPI_Request const* first = requests();
  for (MPI_Request const* r = first; r < first + nreq; ++r) {
    if (*r != MPI_REQUEST_NULL) {
      return false;
    }
  }
  return true;
}

bool
request::simple_handler::null_requests() const
{
  return m_request == MPI_REQUEST_NULL;
}

namespace detail {
MPI_Status
report_test_wait_error(std::string fname, int error_code, MPI_Status* stats, int n) {
  MPI_Status res = stats[0];
  if (error_code != MPI_SUCCESS) {
    if (error_code == MPI_ERR_IN_STATUS) {
      // some specific request went wrong, signal the first failed one.
      for (int i = 0; i < n; ++i) {
        int err = stats[i].MPI_ERROR;
        // MPI_ERR_PENDING can only appear on Wait
        if (err != MPI_SUCCESS && err != MPI_ERR_PENDING) {
          res = stats[i];
          boost::throw_exception(exception(fname, err));
        }
      }
      boost::throw_exception(exception(fname + " -- intenal error", error_code));
    }
    // something else went wrong
    boost::throw_exception(exception(fname, error_code));
  }
  return res;
}
}

status
request::archive_handler::wait() 
{
  // This request is a send of a serialized type, broken into two
  // separate messages. Complete both sends at once.
  MPI_Status stats[2];
  int error_code = MPI_Waitall(2, m_requests, stats);
  return status(error_code == MPI_SUCCESS
                ? stats[0]
                // likely to throw
                : detail::report_test_wait_error("MPI_Waitall", error_code, stats, 2));
}

optional<status>
request::archive_handler::test()
{ 
  // This request is a send of a serialized type, broken into two
  // separate messages. We only get a result if both complete.
  MPI_Status stats[2];
  int flag = 0;
  int error_code = MPI_Testall(2, m_requests, &flag, stats);
  return (bool(flag)
          ? optional<status>(status(error_code == MPI_SUCCESS
                                    ? stats[0]
                                    // likely to throw:
                                    : detail::report_test_wait_error("MPI_Testall", error_code, stats, 2)))
          : optional<status>());
}

void
request::archive_handler::cancel() 
{
  this->handler::cancel();
}

} } // end namespace boost::mpi
