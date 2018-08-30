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
   *  Constructs request for complex data.
   */
  template<typename T> request(communicator const& comm, int source, int tag, T& value)
    : m_handler(new legacy_handler(comm, source, tag, value)) {}
  /**
   *  Constructs request for array of complex data.
   */  
  template<typename T> request(communicator const& comm, int source, int tag, T* values, int n)
    : m_handler(new legacy_handler(comm, source, tag, values, n)) {}
  /**
   *  Constructs request for array of primitive data.
   */
  template<typename T, class A> request(communicator const& comm, int source, int tag, std::vector<T,A>& values, mpl::true_ primitive)
    : m_handler(new legacy_handler(comm, source, tag, values, primitive)) {}

  static request make_trivial() { return request(new trivial_handler()); }
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

  enum request_action { ra_wait, ra_test, ra_cancel };

  struct legacy_handler : public handler {
    legacy_handler()
      : m_data(), m_handler(0) {
      m_requests[0] = MPI_REQUEST_NULL;
      m_requests[1] = MPI_REQUEST_NULL;
    }
    template<typename T> legacy_handler(communicator const& comm, int source, int tag, T& value);
    template<typename T> legacy_handler(communicator const& comm, int source, int tag, T* value, int n);
    template<typename T, class A> legacy_handler(communicator const& comm, int source, int tag, std::vector<T,A>& values, mpl::true_ primitive);

    status wait();
    optional<status> test();
    void cancel();

    bool active() const;
    optional<MPI_Request&> trivial();
    
    MPI_Request& size_request() { return m_requests[0]; }
    MPI_Request& payload_request() { return m_requests[1]; }

    boost::shared_ptr<void> data() { return m_data; }
    void                    set_data(boost::shared_ptr<void> d) { m_data = d; }

    template<class T>  boost::shared_ptr<T>    data() { return boost::static_pointer_cast<T>(m_data); }
    template<class T>  void                    set_data(boost::shared_ptr<T> d) { m_data = d; }
    
    typedef optional<status> (*handler_type)(request::legacy_handler* self, 
					     request_action action);
    template<typename T>
    static optional<status> 
    handle_serialized_irecv(legacy_handler* self, request_action action);
    
    template<typename T>
    static optional<status> 
    handle_serialized_array_irecv(legacy_handler* self, request_action action);
    
    template<typename T, class A>
    static optional<status> 
    handle_dynamic_primitive_array_irecv(legacy_handler* self, request_action action);
    
    MPI_Request      m_requests[2];
    shared_ptr<void> m_data;
    handler_type     m_handler;  
  };

  struct trivial_handler : public handler {
    trivial_handler();
    
    status wait();
    optional<status> test();
    void cancel();

    bool active() const;
    optional<MPI_Request&> trivial();
    
    MPI_Request& size_request();
    MPI_Request& payload_request();
    
    boost::shared_ptr<void> data();
    void                    set_data(boost::shared_ptr<void> d);
    
    MPI_Request      m_request;
    shared_ptr<void> m_data;
  };
  
 private:
  shared_ptr<handler> m_handler;
};

} } // end namespace boost::mpi

#endif // BOOST_MPI_REQUEST_HPP
