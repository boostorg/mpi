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
  class simple_handler;
  class archive_handler;
  
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

  bool trivial()       const;
  bool null_requests() const;

 public:
  shared_ptr<handler> m_handler;
  friend class communicator;
};

class request::handler {
protected:
  handler();
public:
  virtual ~handler();
  friend class communicator;
  friend class request;

  virtual status wait() = 0;
  virtual optional<status> test() = 0;
  virtual void cancel();

  virtual bool null_requests() const;
  virtual MPI_Request* requests() = 0;
  MPI_Request const* requests() const;
  MPI_Request& request(int i);
  virtual int  nb_requests() const { return 1; }
  virtual bool trivial() const = 0;
};

class request::simple_handler
  : public handler
{
public:
  simple_handler();
  virtual ~simple_handler();
  
  friend class communicator;
  friend class request;

  virtual status wait();
  virtual optional<status> test();
  // virtual void cancel() = inherit

  virtual bool         null_requests() const;
  virtual MPI_Request* requests()    { return &m_request; }
  virtual int          nb_requests() const { return 1; }
  virtual bool         trivial() const { return true; }
  
private:
  MPI_Request m_request;
};

class request::archive_handler 
  : public request::handler
{
public:
  template<class Archive>
  archive_handler(Archive& archive, 
                  MPI_Request* size_and_worlload)
    : handler(), m_archive(shared_ptr<Archive>(&archive)) {
    std::copy(size_and_worlload, size_and_worlload + 2, m_requests);
  }
  
  virtual status wait();
  virtual optional<status> test();
  virtual void cancel();

  virtual MPI_Request* requests()    { return m_requests; }
  virtual int          nb_requests() const { return 2; }
  virtual bool trivial() const { return false; }

private:
  MPI_Request      m_requests[2];
  shared_ptr<void> m_archive;
};

inline bool
request::trivial() const { 
  return m_handler->trivial();
}

inline bool
request::null_requests() const {
  return m_handler->null_requests();
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

inline
MPI_Request const*
request::handler::requests() const { 
  return const_cast<handler*>(this)->requests();
}

inline
MPI_Request&
request::handler::request(int i) {
  assert(i>=0 && i<nb_requests());
  return requests()[i];
}

} } // end namespace boost::mpi

#endif // BOOST_MPI_REQUEST_HPP
