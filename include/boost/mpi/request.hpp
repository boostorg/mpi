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
#include <boost/mpi/status.hpp>

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
  class handler;
  /**
   *  Constructs a trivial request.
   */
  request();

  /**
   *  Constructs a less trivial request.
   */
  request(handler* h);

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

  bool trivial() const;
  
 private:
  enum request_action { ra_wait, ra_test, ra_cancel };
  typedef optional<status> (*handler_type)(request::handler* self, 
                                           request_action action);

  /**
   * INTERNAL ONLY
   *
   * Handles the non-blocking receive of a serialized value.
   */
  template<typename T>
  static optional<status> 
    handle_serialized_irecv(request::handler* self, request_action action);

  /**
   * INTERNAL ONLY
   *
   * Handles the non-blocking receive of an array of  serialized values.
   */
  template<typename T>
  static optional<status> 
    handle_serialized_array_irecv(request::handler* self, request_action action);

 public: // template friends are not portable
  shared_ptr<handler> m_handler;
  friend class communicator;
};

class request::handler {
public:
  handler();
  virtual ~handler();
  friend class communicator;
  friend class request;

  virtual status wait();
  virtual optional<status> test();
  virtual void cancel();
  
  bool trivial() const { return !m_handler && m_requests[1] == MPI_REQUEST_NULL; }
  
  MPI_Request m_requests[2];
  handler_type m_handler;
  shared_ptr<void> m_data;
};

inline bool
request::trivial() const { 
  return m_handler->trivial();
}

inline status
request::wait() 
{
  return m_handler->wait();
}

inline optional<status> 
request::test() 
{
  return m_handler->test();  
}

inline void 
request::cancel()
{
  m_handler->cancel();
}

} } // end namespace boost::mpi

#endif // BOOST_MPI_REQUEST_HPP
