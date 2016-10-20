// Copyright (C) 2006 Douglas Gregor.

// Use, modification and distribution is subject to the Boost Software
// License, Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
#include <boost/mpi/request.hpp>
#include <boost/mpi/status.hpp>
#include <boost/foreach.hpp>

namespace boost { namespace mpi {

typedef std::vector<request::impl>::iterator request_iter;
/***************************************************************************
 * request                                                                 *
 ***************************************************************************/
request::request(int nb)
  : m_impl(nb)
{
  for (request_iter it = m_impl.begin(); it != m_impl.end(); ++ it) {
    it->m_requests[0] = MPI_REQUEST_NULL;
    it->m_requests[1] = MPI_REQUEST_NULL;
  }
}

status request::impl::wait()
{
  if (m_handler) {
    // This request is a receive for a serialized type. Use the
    // handler to wait for completion.
    return *m_handler(this, ra_wait);
  } else if (m_requests[1] == MPI_REQUEST_NULL) {
    // This request is either a send or a receive for a type with an
    // associated MPI datatype, or a serialized datatype that has been
    // packed into a single message. Just wait on the one receive/send
    // and return the status to the user.
    status result;
    BOOST_MPI_CHECK_RESULT(MPI_Wait, (&m_requests[0], &result.m_status));
    return result;
  } else {
    // This request is a send of a serialized type, broken into two
    // separate messages. Complete both sends at once.
    MPI_Status stats[2];
    int error_code = MPI_Waitall(2, m_requests, stats);
    if (error_code == MPI_ERR_IN_STATUS) {
      // Dig out which status structure has the error, and use that
      // one when throwing the exception.
      if (stats[0].MPI_ERROR == MPI_SUCCESS 
          || stats[0].MPI_ERROR == MPI_ERR_PENDING)
        boost::throw_exception(exception("MPI_Waitall", stats[1].MPI_ERROR));
      else
        boost::throw_exception(exception("MPI_Waitall", stats[0].MPI_ERROR));
    } else if (error_code != MPI_SUCCESS) {
      // There was an error somewhere in the MPI_Waitall call; throw
      // an exception for it.
      boost::throw_exception(exception("MPI_Waitall", error_code));
    } 

    // No errors. Returns the first status structure.
    status result;
    result.m_status = stats[0];
    return result;
  }
}

status request::wait() {
  status s;
  // will return the last status
  for (request_iter it = m_impl.begin(); it != m_impl.end(); ++ it) {
    s = it->wait();
  }
  return s;
}

optional<status> request::impl::test()
{
  if (m_handler) {
    // This request is a receive for a serialized type. Use the
    // handler to test for completion.
    return m_handler(this, ra_test);
  } else if (m_requests[1] == MPI_REQUEST_NULL) {
    // This request is either a send or a receive for a type with an
    // associated MPI datatype, or a serialized datatype that has been
    // packed into a single message. Just test the one receive/send
    // and return the status to the user if it has completed.
    status result;
    int flag = 0;
    BOOST_MPI_CHECK_RESULT(MPI_Test, 
                           (&m_requests[0], &flag, &result.m_status));
    return flag != 0? optional<status>(result) : optional<status>();
  } else {
    // This request is a send of a serialized type, broken into two
    // separate messages. We only get a result if both complete.
    MPI_Status stats[2];
    int flag = 0;
    int error_code = MPI_Testall(2, m_requests, &flag, stats);
    if (error_code == MPI_ERR_IN_STATUS) {
      // Dig out which status structure has the error, and use that
      // one when throwing the exception.
      if (stats[0].MPI_ERROR == MPI_SUCCESS 
          || stats[0].MPI_ERROR == MPI_ERR_PENDING)
        boost::throw_exception(exception("MPI_Testall", stats[1].MPI_ERROR));
      else
        boost::throw_exception(exception("MPI_Testall", stats[0].MPI_ERROR));
    } else if (error_code != MPI_SUCCESS) {
      // There was an error somewhere in the MPI_Testall call; throw
      // an exception for it.
      boost::throw_exception(exception("MPI_Testall", error_code));
    }

    // No errors. Returns the second status structure if the send has
    // completed.
    if (flag != 0) {
      status result;
      result.m_status = stats[1];
      return result;
    } else {
      return optional<status>();
    }
  }
}

optional<status> request::test()
{
  if (m_impl.size() == 1) {
    return m_impl.front().test();
  } else {
    std::vector<impl> incomplete;
    optional<status> res, s;
    // if we need to arbitraly choose one request, it
    // is more efficient to pick the first.
    // So we scan in reverser order to end up on that one.
    typedef std::vector<impl>::reverse_iterator riter;
    for (riter it = m_impl.rbegin(); it != m_impl.rend(); ++it) {
      s = it->test();
      if (!s) { // not completed, keep it
        incomplete.push_back(*it);
        res = s;
      }
    }
    if (incomplete.empty()) {
      // they all completed, keep the first one for 
      // future calls.
      m_impl.resize(1);
      return s;
    } else {
      m_impl.swap(incomplete);
      return res;
    }
    return s;
  }
}

void request::impl::cancel()
{
  if (m_handler) {
    m_handler(this, ra_cancel);
  } else {
    BOOST_MPI_CHECK_RESULT(MPI_Cancel, (&m_requests[0]));
    if (m_requests[1] != MPI_REQUEST_NULL)
      BOOST_MPI_CHECK_RESULT(MPI_Cancel, (&m_requests[1]));
  }
}

void request::cancel() {
  for (request_iter it = m_impl.begin(); it != m_impl.end(); ++ it) {
    it->cancel();
  }
}

} } // end namespace boost::mpi
