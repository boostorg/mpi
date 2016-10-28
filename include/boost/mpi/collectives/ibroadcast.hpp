//          Copyright Alain Miniussi 2014.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE_1_0.txt or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

// Authors: Alain Miniussi

#ifndef BOOST_MPI_IBROADCAST_HPP
#define BOOST_MPI_IBROADCAST_HPP

#include  <boost/mpi/collectives/broadcast.hpp>
#include  <boost/mpi/request.hpp>

namespace boost { namespace mpi {
namespace detail {
// We're sending a type that has an associated MPI datatype, so
// we'll use MPI_Ibcast to do all of the work and handle it with 
// a request::basic_handler

template<typename T>
request
ibroadcast_impl(const communicator& comm, T* values, int n, int root, 
                mpl::true_)
{
  shared_ptr<request::basic_handler> handler(new request::basic_handler());
  
  BOOST_MPI_CHECK_RESULT(MPI_Ibcast,
                         (values, n,
                          boost::mpi::get_mpi_datatype<T>(*values),
                          root, MPI_Comm(comm), &handler->request(0)));
  return request(handler);
}

class ibroadcast_root_handler_base
  : public request::handler {
public:
  typedef packed_oarchive::buffer_type buffer_type;
  ibroadcast_root_handler_base(const communicator& comm, int root)
    : m_comm(comm), m_buffer(), m_root(root) {
    m_requests[0] = MPI_REQUEST_NULL;
    m_requests[1] = MPI_REQUEST_NULL;
  }
  
  void share_size() {
    int size = m_buffer.size();
    BOOST_MPI_CHECK_RESULT(MPI_Ibcast,
                           (&size, 1,
                            MPI_INTEGER,
                            m_root, m_comm, m_requests));
  }
  void share_data() {
    BOOST_MPI_CHECK_RESULT(MPI_Ibcast,
                           (m_buffer.data(),
                            m_buffer.size(),
                            MPI_BYTE,
                            m_root, m_comm, m_requests+1));
  }

  virtual status wait() {
    MPI_Status stats[2];
    int error_code = MPI_Waitall(2, m_requests, stats);
    return status(error_code == MPI_SUCCESS
                  ? stats[0]
                  // likely to throw
                  : detail::report_test_wait_error("MPI_Waitall", error_code, stats, 2));
    m_buffer.resize(0);
  }
  
  virtual optional<status> test() {
    // This request is a send of a serialized type, broken into two
    // separate messages. We only get a result if both complete.
    MPI_Status stats[2];
    int flag = 0;
    int error_code = MPI_Testall(2, m_requests, &flag, stats);
    if (bool(flag)) {
      m_buffer.resize(0);
      return optional<status>(status(error_code == MPI_SUCCESS
                                     ? stats[0]
                                     // likely to throw:
                                     : detail::report_test_wait_error("MPI_Testall", 
                                                                      error_code, 
                                                                      stats, 2)));
    } else {
      return optional<status>();
    }
  }
  
  virtual MPI_Request* requests() { return m_requests; }
  virtual int  nb_requests() const { return 2; }
  virtual bool trivial() const { return false; }

protected:
  communicator const& m_comm;
  buffer_type         m_buffer;
  int                 m_root;
  MPI_Request         m_requests[2];
};

template<class T>
class ibroadcast_root_handler
  : public ibroadcast_root_handler_base {
  ibroadcast_root_handler(const communicator& comm, int root,
                          T const* values, int n)
    : ibroadcast_root_handler_base(comm, root) {
    packed_oarchive oa(m_comm, m_buffer);
    for (int i = 0; i < n; ++i) {
      oa << values[i];
    }
    share_size();
    share_data();
  }
};
  
class ibroadcast_target_handler_base
  : public request::handler 
{
public:
  typedef packed_iarchive::buffer_type buffer_type;
  ibroadcast_target_handler_base(const communicator& comm, int root)
    : m_comm(comm), m_buffer(), m_root(root), m_size(0) {
    m_requests[0] = MPI_REQUEST_NULL;
    m_requests[1] = MPI_REQUEST_NULL;
    share_size();
  }
  
  void share_size() {
    BOOST_MPI_CHECK_RESULT(MPI_Ibcast,
                           (&m_size, 1,
                            MPI_INTEGER,
                            m_root, m_comm, m_requests));
  }
  void share_data() {
    assert(m_requests[0] == MPI_REQUEST_NULL);
    m_buffer.resize(m_size);
    BOOST_MPI_CHECK_RESULT(MPI_Ibcast,
                           (m_buffer.data(), m_buffer.size(),
                            MPI_BYTE,
                            m_root, m_comm, m_requests+1));
  }
  
  virtual status wait() {
    status stat;
    if (m_requests[1] == MPI_REQUEST_NULL) {
      // Wait for the count message to complete
      BOOST_MPI_CHECK_RESULT(MPI_Wait,
                             (m_requests, &stat.m_status));
      share_data();      
      // Wait until we have received the entire message
      BOOST_MPI_CHECK_RESULT(MPI_Wait,
                             (m_requests + 1, &stat.m_status));
      
      this->load(stat);
    }
    return stat;
  }
  virtual void load(status& stat) = 0;
  
  virtual optional<status> test() {
    int flag = 0;
    // This request is a send of a serialized type, broken into two
    // separate messages. We only get a result if both complete.
    {
      MPI_Status stats[2];
      int error_code = MPI_Testall(2, m_requests, &flag, stats);
      if (bool(flag)) {
        // Resize our buffer and get ready to receive its data
        share_data();
      } else {
        if (error_code == MPI_SUCCESS) {
          return optional<status>(); // We have not finished yet
        } else {
          return status(detail::report_test_wait_error("MPI_Testall", 
                                                       error_code, 
                                                       stats, 2));
        }
      }
    }
    status stat;
    // Check if we have received the message data
    BOOST_MPI_CHECK_RESULT(MPI_Test,
                           (m_requests + 1, &flag, &stat.m_status));
    if (flag) {
      load(stat);
      return stat;
    } else {
      return optional<status>();
    }
  }
  
  virtual MPI_Request* requests() { return m_requests; }
  virtual int  nb_requests() const { return 2; }
  virtual bool trivial() const { return false; }
  
  private:
  communicator const& m_comm;
  buffer_type         m_buffer;
  MPI_Request         m_requests[2];
  int                 m_root;
  int                 m_size;
};

template<class T>
class ibroadcast_target_handler
  : public ibroadcast_target_handler_base {
public:
  ibroadcast_target_handler(const communicator& comm, int root, T* ptr, int n)
    : ibroadcast_target_handler_base(comm,root),m_buffer(ptr), m_nslots(n) {}
  
  void load(status& stat) {
    packed_iarchive ia(m_comm, m_buffer);
    for (int i = 0; i < m_nslots; ++i) {
      ia >> buffer[i];
    }
    stat.m_count = m_nslosts;
    m_buffer.resize(0);
  }
};

// We're sending a type that has an no associated MPI datatype,
// so we delegate the work to more complex handler capable of 
// performing a two stage seliaization.

template<typename T>
request
ibroadcast_impl(const communicator& comm, T* values, int n, int root, 
                mpl::false_)
{
  shared_ptr<request::basic_handler> handler;
  if (comm.rank() == root) {
    handler.reset(new ibroadcast_root_handler<T>(comm, root, values, n));
  } else {
    handler.reset(new ibroadcast_target_handler<T>(comm, root, values, n));
  }
  return request(handler);
}
}

template<typename T>
request ibroadcast(const communicator& comm, T* values, int n, int root)
{
  detail::ibroadcast_impl(comm, values, n, root, is_mpi_datatype<T>());
}

template<typename T>
request ibroadcast(const communicator& comm, T& value, int root)
{
  detail::ibroadcast(comm, &value, 1, root, is_mpi_datatype<T>());
}

template<typename T>
request ibroadcast(const communicator& comm, std::vector<T>& values, int root)
{
  detail::ibroadcast(comm, values.data(), values.size(), root, is_mpi_datatype<T>());
}

}}
#endif // BOOST_MPI_IBROADCAST_HPP

