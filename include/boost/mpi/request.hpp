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
#include <boost/mpi/status.hpp>
#include <boost/optional.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/mpi/packed_iarchive.hpp>

namespace boost { namespace mpi {

class status;
class communicator;

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
   *  Constructs a NULL request.
   */
  request();

  /**
   * Construct request for primitive data of statically known size.
   */
  static request make_trivial();
  /**
   * Construct request for simple data of unknown size.
   */
  static request make_dynamic();
  /**
   *  Constructs request for serialized data.
   */
  template<typename T>
  static request make_serialized(communicator const& comm, int source, int tag, T& value);
  /**
   *  Constructs request for array of complex data.
   */  
  template<typename T>
  static request make_serialized_array(communicator const& comm, int source, int tag, T* values, int n);
  /**
   *  Constructs request for array of primitive data.
   */
  template<typename T, class A>
  static request make_dynamic_primitive_array(communicator const& comm, int source, int tag, 
                                              std::vector<T,A>& values);

  static bool probe_messages();
  /**
   *  Wait until the communication associated with this request has
   *  completed, then return a @c status object describing the
   *  communication.
   */
  status wait() { return m_handler->wait(); }

  /**
   *  Determine whether the communication associated with this request
   *  has completed successfully. If so, returns the @c status object
   *  describing the communication. Otherwise, returns an empty @c
   *  optional<> to indicate that the communication has not completed
   *  yet. Note that once @c test() returns a @c status object, the
   *  request has completed and @c wait() should not be called.
   */
  optional<status> test() { return m_handler->test(); }

  /**
   *  Cancel a pending communication, assuming it has not already been
   *  completed.
   */
  void cancel() { m_handler->cancel(); }
  
  /**
   * The trivial MPI requet implenting this request, provided it's trivial.
   * Probably irrelevant to most users.
   */
  optional<MPI_Request&> trivial() { return m_handler->trivial(); }

  /**
   * For two steps requests, that need to first send the size, then the payload,
   * access to the size request.
   * Probably irrelevant to most users.
   */
  MPI_Request& size_request() { return m_handler->size_request(); }

  /**
   * For two steps requests, that need to first send the size, then the payload,
   * access to the size request.
   * Probably irrelevant to most users.
   */
  MPI_Request& payload_request() { return m_handler->payload_request(); }

  /**
   * Is this request potentialy pending ?
   */
  bool active() const { return m_handler->active(); }

  template<class T>  boost::shared_ptr<T>    data() { return boost::static_pointer_cast<T>(m_handler->data()); }
  template<class T>  void                    set_data(boost::shared_ptr<T> d) { m_handler->set_data(d); }

  struct handler {
    virtual ~handler() = 0;
    virtual status wait() = 0;
    virtual optional<status> test() = 0;
    virtual void cancel() = 0;
    
    virtual bool active() const = 0;
    virtual optional<MPI_Request&> trivial() = 0;
    
    virtual MPI_Request& size_request() = 0;
    virtual MPI_Request& payload_request() = 0;
    
    virtual boost::shared_ptr<void> data() = 0;
    virtual void                    set_data(boost::shared_ptr<void> d) = 0;
  };
  
 private:
  
  request(handler *h) : m_handler(h) {};

  // specific implementations
  struct legacy_handler;
  struct probe_handler;
  struct trivial_handler;  
  struct dynamic_handler;
  struct legacy_handler;
  template<typename T> struct legacy_serialized_handler;
  template<typename T> struct legacy_serialized_array_handler;
  template<typename T, class A> struct legacy_dynamic_primitive_array_handler;
  template<class A> struct dynamic_primitive_array_handler;
  
 private:
  shared_ptr<handler> m_handler;
};

inline
bool
request::probe_messages() {
#ifdef BOOST_MPI_NO_IMPROBE
  return false;
#else
  return true;
#endif
}
} } // end namespace boost::mpi

#endif // BOOST_MPI_REQUEST_HPP
