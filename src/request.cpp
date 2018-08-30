// Copyright (C) 2006 Douglas Gregor.

// Use, modification and distribution is subject to the Boost Software
// License, Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
#include <boost/mpi/request.hpp>
#include <boost/mpi/status.hpp>

namespace boost { namespace mpi {

/***************************************************************************
 * request                                                                 *
 ***************************************************************************/
request::request() 
  : m_handler(new legacy_handler()) {}

/***************************************************************************
 * handlers                                                                *
 ***************************************************************************/

request::handler::~handler() {}
    
optional<MPI_Request&>
request::legacy_handler::trivial() {
  if ((!bool(m_handler) && m_requests[1] == MPI_REQUEST_NULL)) {
    return m_requests[0];
  } else {
    return boost::none;
  }
}

bool
request::legacy_handler::active() const {
  return m_requests[0] != MPI_REQUEST_NULL || m_requests[1] != MPI_REQUEST_NULL;
}

status
request::legacy_handler::wait()
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

optional<status>
request::legacy_handler::test()
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

void
request::legacy_handler::cancel()
{
  if (m_handler) {
    m_handler(this, ra_cancel);
  } else {
    BOOST_MPI_CHECK_RESULT(MPI_Cancel, (&m_requests[0]));
    if (m_requests[1] != MPI_REQUEST_NULL)
      BOOST_MPI_CHECK_RESULT(MPI_Cancel, (&m_requests[1]));
  }
}

// trivial handler

request::trivial_handler::trivial_handler()
    : m_request(MPI_REQUEST_NULL), m_data() {}
  
status
request::trivial_handler::wait()
{
  status result;
  BOOST_MPI_CHECK_RESULT(MPI_Wait, (&m_request, &result.m_status));
  return result;  
}


optional<status>
request::trivial_handler::test() 
{
  status result;
  int flag = 0;
  BOOST_MPI_CHECK_RESULT(MPI_Test, 
                         (&m_request, &flag, &result.m_status));
  return flag != 0? optional<status>(result) : optional<status>();
}

void
request::trivial_handler::cancel()
{
  BOOST_MPI_CHECK_RESULT(MPI_Cancel, (&m_request));
}
  
bool
request::trivial_handler::active() const
{
  return m_request != MPI_REQUEST_NULL; 
}

optional<MPI_Request&>
request::trivial_handler::trivial() 
{ 
  return m_request; 
}
  
MPI_Request&
request::trivial_handler::size_request()
{
  std::abort(); 
  return m_request; // avoid warning
}

MPI_Request&
request::trivial_handler::payload_request()
{
  std::abort();
  return m_request;  // avoid warning
}
  
boost::shared_ptr<void>
request::trivial_handler::data() 
{
  return m_data; 
}

void
request::trivial_handler::set_data(boost::shared_ptr<void> d) 
{
  m_data = d; 
}

// dynamic handler


request::dynamic_handler::dynamic_handler()
  : m_data() {
  m_requests[0] = MPI_REQUEST_NULL;
  m_requests[1] = MPI_REQUEST_NULL;

}
  
status
request::dynamic_handler::wait()
{
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
 
optional<status>
request::dynamic_handler::test()
{
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

void
request::dynamic_handler::cancel()
{
  BOOST_MPI_CHECK_RESULT(MPI_Cancel, (&m_requests[0]));
  BOOST_MPI_CHECK_RESULT(MPI_Cancel, (&m_requests[1]));
}

bool
request::dynamic_handler::active() const
{
  return (m_requests[0] != MPI_REQUEST_NULL
          || m_requests[1] != MPI_REQUEST_NULL);
}

optional<MPI_Request&>
request::dynamic_handler::trivial() {
  return boost::none;
}
  
MPI_Request&
request::dynamic_handler::size_request()
{
  return m_requests[0];
}

MPI_Request&
request::dynamic_handler::payload_request()
{
  return m_requests[1];
}
  
boost::shared_ptr<void>
request::dynamic_handler::data()
{
  return m_data;
}

void 
request::dynamic_handler::set_data(boost::shared_ptr<void> d) 
{
  m_data = d;
}


} } // end namespace boost::mpi
