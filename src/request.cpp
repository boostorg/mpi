// Copyright (C) 2006 Douglas Gregor.

// Use, modification and distribution is subject to the Boost Software
// License, Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
#include <boost/mpi/request.hpp>
#include <boost/mpi/status.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi/error_string.hpp>

namespace boost { namespace mpi {

/***************************************************************************
 * request                                                                 *
 ***************************************************************************/
request::request()
  : m_request(new MPI_Request),
    m_probe_info(),
    m_data()
{
  *m_request = MPI_REQUEST_NULL;
}

status request::wait()
{
  status stat;
  if (bool(m_request)) {
    BOOST_MPI_CHECK_RESULT(MPI_Wait, (m_request.get(), &stat.m_status));
    m_request.reset();
    m_data.reset();
  } else if (bool(m_probe_info)) {
    BOOST_MPI_CHECK_RESULT(MPI_Mprobe, (m_probe_info->m_source, m_probe_info->m_tag,
                                        m_probe_info->m_comm,
                                        &m_probe_info->m_message,
                                        &stat.m_status));
    int count;
    BOOST_MPI_CHECK_RESULT(MPI_Get_count, (&stat.m_status, MPI_PACKED, &count));
    packed_iarchive& ar = m_probe_info->archive();
    ar.resize(count);
    BOOST_MPI_CHECK_RESULT(MPI_Mrecv,
                           (ar.address(), ar.size(), MPI_PACKED,
                            &m_probe_info->m_message, &stat.m_status));    
    m_probe_info->deserialize(stat);
    m_probe_info.reset();
    m_data.reset();
  } else {
    stat = status::empty_status();
  }
  return stat;
}

optional<status> request::test()
{
  status stat;
  int flag = 0;
  if (bool(m_request)) {
    BOOST_MPI_CHECK_RESULT(MPI_Test, (m_request.get(), &flag, &stat.m_status));
    if (flag) {
      m_request.reset();
      m_data.reset();
      return optional<status>(stat);
    } else {
      return optional<status>();
    }
  } else if (bool(m_probe_info)) {
    BOOST_MPI_CHECK_RESULT(MPI_Improbe, (m_probe_info->m_source, m_probe_info->m_tag,
                                         m_probe_info->m_comm, &flag,
                                         &m_probe_info->m_message,
                                         &stat.m_status));
    if (flag) { 
      // message is arrived
      int count;
      BOOST_MPI_CHECK_RESULT(MPI_Get_count, (&stat.m_status, MPI_PACKED, &count));
      packed_iarchive& ar = m_probe_info->archive();
      ar.resize(count);
      BOOST_MPI_CHECK_RESULT(MPI_Mrecv,
                             (ar.address(), ar.size(), MPI_PACKED,
                              &m_probe_info->m_message, &stat.m_status));    
      m_probe_info->deserialize(stat);
      m_probe_info.reset();
      m_data.reset();
      return optional<status>(stat);
    } else {
      return  optional<status>();
    }
  } else {
    return  optional<status>();
  }
}

void request::cancel()
{
  if (bool(m_request)) {
    BOOST_MPI_CHECK_RESULT(MPI_Cancel, (m_request.get()));
    m_request.reset();
    m_data.reset();
  } else if (bool(m_probe_info)) {
    m_probe_info.reset();
    m_data.reset();
  } else {
    BOOST_ASSERT(!m_data);
  }
}

} } // end namespace boost::mpi
