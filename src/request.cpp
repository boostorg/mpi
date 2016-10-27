// Copyright (C) 2006 Douglas Gregor.

// Use, modification and distribution is subject to the Boost Software
// License, Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
#include <boost/mpi/request.hpp>
#include <boost/mpi/status.hpp>

namespace boost { namespace mpi {

request::handler::handler(bool simple)
  : m_simple(simple)
{
  m_requests[0] = MPI_REQUEST_NULL;
  m_requests[1] = MPI_REQUEST_NULL;
}

request::handler::~handler()
{
}

request::request()
  : m_handler(new handler(true)) 
{
}

request::request(handler* h) 
  : m_handler(h) 
{
}

status
request::handler::wait()
{
  assert(m_requests[1] == MPI_REQUEST_NULL);
  // This request is either a send or a receive for a type with an
  // associated MPI datatype
  status result;
  BOOST_MPI_CHECK_RESULT(MPI_Wait, (&m_requests[0], &result.m_status));
  return result;
}

optional<status> 
request::handler::test()
{
  assert(m_requests[1] == MPI_REQUEST_NULL);
  status result;
  int flag = 0;
  BOOST_MPI_CHECK_RESULT(MPI_Test, 
                         (&m_requests[0], &flag, &result.m_status));
  return bool(flag) ? optional<status>(result) : optional<status>();
}

void
request::handler::cancel()
{
  BOOST_MPI_CHECK_RESULT(MPI_Cancel, (&m_requests[0]));
  if (m_requests[1] != MPI_REQUEST_NULL)
    BOOST_MPI_CHECK_RESULT(MPI_Cancel, (&m_requests[1]));
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
  if (m_requests[1] == MPI_REQUEST_NULL) {
    // This request could be a serialized datatype that has been
    // packed into a single message. Just wait on the one receive/send
    // and return the status to the user.
    // This is very unlikely though.
    return this->handler::wait();
  } else {
    // This request is a send of a serialized type, broken into two
    // separate messages. Complete both sends at once.
    MPI_Status stats[2];
    int error_code = MPI_Waitall(2, m_requests, stats);
    return status(error_code == MPI_SUCCESS
                  ? stats[0]
                  : detail::report_test_wait_error("MPI_Waitall", error_code, stats, 2));
  }
}

optional<status>
request::archive_handler::test()
{ 
  if (m_requests[1] == MPI_REQUEST_NULL) {
    // This request could be a serialized datatype that has been
    // packed into a single message. Just test the one receive/send
    // and return the status to the user if it has completed.
    // This is very unlikely though.
    return this->handler::test();
  } else {
    // This request is a send of a serialized type, broken into two
    // separate messages. We only get a result if both complete.
    MPI_Status stats[2];
    int flag = 0;
    int error_code = MPI_Testall(2, m_requests, &flag, stats);
    return (bool(flag)
            ? optional<status>(status(error_code == MPI_SUCCESS
                                      ? stats[0]
                                      : detail::report_test_wait_error("MPI_Testall", error_code, stats, 2)))
            : optional<status>());
  }
}

void
request::archive_handler::cancel() 
{
  this->handler::cancel();
}

} } // end namespace boost::mpi
