// Copyright (C) 2006 Douglas Gregor <doug.gregor -at- gmail.com>.

// Use, modification and distribution is subject to the Boost Software
// License, Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)

/** @file nonblocking.hpp
 *
 *  This header defines operations for completing non-blocking
 *  communication requests.
 */
#ifndef BOOST_MPI_NONBLOCKING_HPP
#define BOOST_MPI_NONBLOCKING_HPP

#include <boost/mpi/config.hpp>
#include <vector>
#include <memory>
#include <cstdio>
#include <iterator> // for std::iterator_traits
#include <boost/optional.hpp>
#include <utility> // for std::pair
#include <algorithm> // for iter_swap, reverse
#include <boost/static_assert.hpp>
#include <boost/mpi/request.hpp>
#include <boost/mpi/status.hpp>
#include <boost/mpi/exception.hpp>

namespace boost { namespace mpi {

/** 
 *  @brief Wait until any non-blocking request has completed.
 *
 *  This routine takes in a set of requests stored in the iterator
 *  range @c [first,last) and waits until any of these requests has
 *  been completed. It provides functionality equivalent to 
 *  @c MPI_Waitany.
 *
 *  @param first The iterator that denotes the beginning of the
 *  sequence of request objects.
 *
 *  @param last The iterator that denotes the end of the sequence of
 *  request objects. This may not be equal to @c first.
 *
 *  @returns A pair containing the status object that corresponds to
 *  the completed operation and the iterator referencing the completed
 *  request.
 */
template<typename ForwardIterator>
std::pair<status, ForwardIterator> 
wait_any(ForwardIterator first, ForwardIterator last)
{
  BOOST_ASSERT(first != last);
  
  ForwardIterator current = first;
  while (current != last) {
    if (current->active()) {
      status stat = current->wait();
      return std::make_pair(stat, current);
      break;
    } else {
      ++current;
    }
  }
  BOOST_ASSERT(false);
  return std::make_pair(status(), last);
}

/** 
 *  @brief Test whether any non-blocking request has completed.
 *
 *  This routine takes in a set of requests stored in the iterator
 *  range @c [first,last) and tests whether any of these requests has
 *  been completed. This routine is similar to @c wait_any, but will
 *  not block waiting for requests to completed. It provides
 *  functionality equivalent to @c MPI_Testany.
 *
 *  @param first The iterator that denotes the beginning of the
 *  sequence of request objects.
 *
 *  @param last The iterator that denotes the end of the sequence of
 *  request objects. 
 *
 *  @returns If any outstanding requests have completed, a pair
 *  containing the status object that corresponds to the completed
 *  operation and the iterator referencing the completed
 *  request. Otherwise, an empty @c optional<>.
 */
template<typename ForwardIterator>
optional<std::pair<status, ForwardIterator> >
test_any(ForwardIterator first, ForwardIterator last)
{
  while (first != last) {
    optional<status> result = first->test();
    if (result) {
      return std::make_pair(*result, first);
    }
    ++first;
  }
  
  // We found nothing
  return optional<std::pair<status, ForwardIterator> >();
}

/** 
 *  @brief Wait until all non-blocking requests have completed.
 *
 *  This routine takes in a set of requests stored in the iterator
 *  range @c [first,last) and waits until all of these requests have
 *  been completed. It provides functionality equivalent to 
 *  @c MPI_Waitall.
 *
 *  @param first The iterator that denotes the beginning of the
 *  sequence of request objects.
 *
 *  @param last The iterator that denotes the end of the sequence of
 *  request objects. 
 *
 *  @param out If provided, an output iterator through which the
 *  status of each request will be emitted. The @c status objects are
 *  emitted in the same order as the requests are retrieved from 
 *  @c [first,last).
 *
 *  @returns If an @p out parameter was provided, the value @c out
 *  after all of the @c status objects have been emitted.
 */
template<typename ForwardIterator, typename OutputIterator>
OutputIterator 
wait_all(ForwardIterator first, ForwardIterator last, OutputIterator out)
{
  std::vector<request*> requests(std::distance(first, last));
  for(int i = 0; i < requests.size(); ++i) {
    requests[i] = &(*first++);
  }
  std::vector<status>   statuses(requests.size());
  typedef std::vector<request*>::iterator vriter;
  int pending;
  do {
    pending = 0;
    for(int i = 0; i < requests.size(); ++i) {
      if (requests[i]) {
        request& req = *requests[i];
        if (!req.active()) {
          statuses[i] = status::empty_status();
        } else {
          optional<status> stat = req.test();
          if (stat) {
            printf("Proc %i got msg %d\n", communicator().rank(), stat->tag());
            statuses[i] = *stat;
            requests[i] = 0;
          } else {
            ++pending;
          }
        }
      }
    }
  } while(pending>0);
  std::copy(statuses.begin(), statuses.end(), out);
  return out;
}

/**
 * \overload
 */
template<typename ForwardIterator>
void
wait_all(ForwardIterator first, ForwardIterator last)
{
  std::vector<status>   statuses(std::distance(first, last));
  wait_all(first, last, statuses.begin());
}

/** 
 *  @brief Tests whether all non-blocking requests have completed.
 *
 *  This routine takes in a set of requests stored in the iterator
 *  range @c [first,last) and determines whether all of these requests
 *  have been completed. However, due to limitations of the underlying
 *  MPI implementation, if any of the requests refers to a
 *  non-blocking send or receive of a serialized data type, @c
 *  test_all will always return the equivalent of @c false (i.e., the
 *  requests cannot all be finished at this time). This routine
 *  performs the same functionality as @c wait_all, except that this
 *  routine will not block. This routine provides functionality
 *  equivalent to @c MPI_Testall.
 *
 *  @param first The iterator that denotes the beginning of the
 *  sequence of request objects.
 *
 *  @param last The iterator that denotes the end of the sequence of
 *  request objects. 
 *
 *  @param out If provided and all requests have been completed, an
 *  output iterator through which the status of each request will be
 *  emitted. The @c status objects are emitted in the same order as
 *  the requests are retrieved from @c [first,last).
 *
 *  @returns If an @p out parameter was provided, the value @c out
 *  after all of the @c status objects have been emitted (if all
 *  requests were completed) or an empty @c optional<>. If no @p out
 *  parameter was provided, returns @c true if all requests have
 *  completed or @c false otherwise.
 */
template<typename ForwardIterator, typename OutputIterator>
optional<OutputIterator>
test_all(ForwardIterator first, ForwardIterator last, OutputIterator out)
{
  bool completed = true;
  for (ForwardIterator it = first; it != last; ++it) {
    if (it->active()) {
      optional<status> stat = it->test();
      if (stat) {
        *out++ = *stat;
      } else {
        completed = false;
      }
    }
  }
  return completed ? optional<OutputIterator>(out) : optional<OutputIterator>();
}

/**
 *  \overload
 */
template<typename ForwardIterator>
bool
test_all(ForwardIterator first, ForwardIterator last)
{
  bool completed = true;
  for (ForwardIterator it = first; it != last; ++it) {
    if (it->active()) {
      if (!it->test()) {
        completed = false;
      }
    }
  }
  return completed;
}

namespace detail {
template<class InputIterator, class OutputIterator>
OutputIterator
sort_requests(InputIterator first, InputIterator last, 
              std::vector<request>& pending,
              std::vector<request>& completed,
              std::vector<request>& inactive,
              OutputIterator out) 
{
  OutputIterator statuses = out;
  for (InputIterator iter = first; iter != last; ++iter) {
    if (!iter->active()) {
      inactive.push_back(*iter);
    } else {
      optional<status> stat = iter->test();
      if (stat) {
        completed.push_back(*iter);
        *statuses++ = *stat;
      } else {
        pending.push_back(*iter);
      }
    }
  }
  return statuses;
}
}
/** 
 *  @brief Wait until some non-blocking requests have completed.
 *
 *  This routine takes in a set of requests stored in the iterator
 *  range @c [first,last) and waits until at least one of the requests
 *  has completed. It then completes all of the requests it can,
 *  partitioning the input sequence into pending requests followed by
 *  completed requests. If an output iterator is provided, @c status
 *  objects will be emitted for each of the completed requests. This
 *  routine provides functionality equivalent to @c MPI_Waitsome.
 *
 *  @param first The iterator that denotes the beginning of the
 *  sequence of request objects.
 *
 *  @param last The iterator that denotes the end of the sequence of
 *  request objects. This may not be equal to @c first.
 *
 *  @param out If provided, the @c status objects corresponding to
 *  completed requests will be emitted through this output iterator.

 *  @returns If the @p out parameter was provided, a pair containing
 *  the output iterator @p out after all of the @c status objects have
 *  been written through it and an iterator referencing the first
 *  completed request. If no @p out parameter was provided, only the
 *  iterator referencing the first completed request will be emitted.
 */
template<typename BidirectionalIterator, typename OutputIterator>
std::pair<OutputIterator, BidirectionalIterator> 
wait_some(BidirectionalIterator first, BidirectionalIterator last,
          OutputIterator out)
{
  typedef std::vector<request>::iterator viter;
  BOOST_ASSERT(first != last); // nothing to wait for
  std::vector<request> pending, completed, inactive;  
  OutputIterator statuses = detail::sort_requests(first, last, pending, completed, inactive, out);
  
  while (completed.empty()) {
    BOOST_ASSERT(!pending.empty()); // nothing to wait for
    optional<std::pair<status, viter> > found = test_any(pending.begin(), pending.end());
    if (found) {
      *statuses++ = found->first;
      completed.push_back(*found->second);
      pending.erase(found->second);
    }
  }
  BidirectionalIterator ordered = first;
  ordered = std::copy(pending.begin(), pending.end(), ordered);
  BidirectionalIterator first_completed = ordered;
  ordered = std::copy(completed.begin(), completed.end(), ordered);
  ordered = std::copy(inactive.begin(), inactive.end(), ordered);
  return std::make_pair(statuses, first_completed);
}

/**
 *  \overload
 */
template<typename BidirectionalIterator>
BidirectionalIterator
wait_some(BidirectionalIterator first, BidirectionalIterator last)
{
  std::vector<status> statuses;
  return wait_some(first, last, std::back_inserter(statuses)).second;
}

/** 
 *  @brief Test whether some non-blocking requests have completed.
 *
 *  This routine takes in a set of requests stored in the iterator
 *  range @c [first,last) and tests to see if any of the requests has
 *  completed. It completes all of the requests it can, partitioning
 *  the input sequence into pending requests followed by completed
 *  requests. If an output iterator is provided, @c status objects
 *  will be emitted for each of the completed requests. This routine
 *  is similar to @c wait_some, but does not wait until any requests
 *  have completed. This routine provides functionality equivalent to
 *  @c MPI_Testsome.
 *
 *  @param first The iterator that denotes the beginning of the
 *  sequence of request objects.
 *
 *  @param last The iterator that denotes the end of the sequence of
 *  request objects. This may not be equal to @c first.
 *
 *  @param out If provided, the @c status objects corresponding to
 *  completed requests will be emitted through this output iterator.

 *  @returns If the @p out parameter was provided, a pair containing
 *  the output iterator @p out after all of the @c status objects have
 *  been written through it and an iterator referencing the first
 *  completed request. If no @p out parameter was provided, only the
 *  iterator referencing the first completed request will be emitted.
 */
template<typename BidirectionalIterator, typename OutputIterator>
std::pair<OutputIterator, BidirectionalIterator> 
test_some(BidirectionalIterator first, BidirectionalIterator last,
          OutputIterator out)
{
  typedef std::vector<request>::iterator viter;
  BOOST_ASSERT(first != last); // nothing to wait for
  std::vector<request> pending, completed, inactive;  
  OutputIterator statuses = detail::sort_requests(first, last, pending, completed, inactive, out);
  BidirectionalIterator ordered = first;
  ordered = std::copy(pending.begin(), pending.end(), ordered);
  BidirectionalIterator first_completed = ordered;
  ordered = std::copy(completed.begin(), completed.end(), ordered);
  ordered = std::copy(inactive.begin(), inactive.end(), ordered);
  return std::make_pair(statuses, first_completed);
}

/**
 *  \overload
 */
template<typename BidirectionalIterator>
BidirectionalIterator
test_some(BidirectionalIterator first, BidirectionalIterator last)
{
  std::vector<status> statuses;
  return test_some(first, last, std::back_inserter(statuses)).second;
}

} } // end namespace boost::mpi


#endif // BOOST_MPI_NONBLOCKING_HPP
