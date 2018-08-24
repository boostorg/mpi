// Copyright (C) 2006 Douglas Gregor <doug.gregor -at- gmail.com>.

// Use, modification and distribution is subject to the Boost Software
// License, Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)

/** @file request.hpp
 *
 *  This header defines the class @c request, which contains a request
 *  for non-blocking communication.
 */
#ifndef BOOST_MPI_REQUEST_HPP
#define BOOST_MPI_REQUEST_HPP

#include <boost/mpi/config.hpp>
#include <boost/optional.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/mpi/packed_iarchive.hpp>
#include <boost/mpi/skeleton_and_content_fwd.hpp>

namespace boost { namespace mpi {

class status;
class communicator;
class packed_oarchive;

/**
 *  @brief A request for a non-blocking send or receive.
 *
 *  This structure contains information about a non-blocking send or
 *  receive and will be returned from @c isend or @c irecv,
 *  respectively.
 */
class BOOST_MPI_DECL request 
{
 public:
  /**
   *  Constructs a request for non serialized type.
   */
  request();

  /**
   *  Constructs a request for serialized type.
   */
  template<class T>
  request(int source, int tag, MPI_Comm const& comm, T& value) 
    : m_request(), 
      m_probe_info(new probe_info<T>(source, tag, comm, &value, 1)),
      m_data() {}

  template<class T>
  request(int source, int tag, MPI_Comm const& comm, const skeleton_proxy<T>& proxy);

  template<class T>
  request(int source, int tag, MPI_Comm const& comm, skeleton_proxy<T>& proxy);

  /**
   *  Constructs a request for serialized type array.
   */
  template<class T>
  request(int source, int tag, MPI_Comm const& comm, T* values, int nb) 
    : m_request(), m_probe_info(new probe_info<T>(source, tag, comm, values, nb)) {}

  request(int source, int tag, MPI_Comm const& comm, packed_iarchive& ar) 
    : m_request(), m_probe_info(new probe_info_iarchive(source, tag, comm, ar)) {}

  /**
   *  Wait until the communication associated with this request has
   *  completed, then return a @c status object describing the
   *  communication.
   */
  status wait();

  /**
   *  Determine whether the communication associated with this request
   *  has completed successfully. If so, returns the @c status object
   *  describing the communication. Otherwise, returns an empty @c
   *  optional<> to indicate that the communication has not completed
   *  yet. Note that once @c test() returns a @c status object, the
   *  request has completed and @c wait() should not be called.
   */
  optional<status> test();

  /**
   *  Cancel a pending communication, assuming it has not already been
   *  completed.
   */
  void cancel();

  bool active() const { return bool(m_request) || bool(m_probe_info); }
  
 private:
  struct probe_info_base {
    int                 m_source;
    int                 m_tag;
    MPI_Message         m_message;
    MPI_Comm            m_comm;
    
    virtual packed_iarchive& archive() = 0;
    virtual void deserialize(status& stat) = 0;

    probe_info_base(int source, int tag, const MPI_Comm& comm)
      : m_source(source),
        m_tag(tag),
        m_message(MPI_MESSAGE_NO_PROC),
        m_comm(comm) {}
    virtual ~probe_info_base() {}
  };

  template<class T>
  struct probe_info : public probe_info_base {
    probe_info(int source, int tag, const MPI_Comm& comm, T* values, int nb)
      : probe_info_base(source, tag, comm),
        m_archive(comm),
        m_values(values),
        m_nb(nb) {}

    packed_iarchive& archive() { return m_archive;}
    void deserialize(status& stat) {
      for(int i = 0; i < m_nb; ++i) {
        m_archive >> m_values[i]; 
      }
      stat.m_count = m_nb;
    }
    packed_iarchive m_archive;
    T*              m_values;
    int             m_nb;
  };

  struct probe_info_iarchive : public probe_info_base {
    probe_info_iarchive(int source, int tag, const MPI_Comm& comm, packed_iarchive& ar)
      : probe_info_base(source, tag, comm),
        m_archive(ar) {}
    packed_iarchive& archive() { return m_archive;}
    void deserialize(status& stat) {}
    
    packed_iarchive& m_archive;
  };

  template<class T> struct probe_info_const_skeleton_proxy;
  template<class T> struct probe_info_skeleton_proxy;
 public: // while debuging
  shared_ptr<MPI_Request>      m_request;
  shared_ptr<probe_info_base>  m_probe_info;
  shared_ptr<void>             m_data;

  friend class communicator;
};


} } // end namespace boost::mpi

#endif // BOOST_MPI_REQUEST_HPP
