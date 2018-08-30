// Copyright (C) 2018 Alain Miniussi <alain.miniussi@oca.eu>.

// Use, modification and distribution is subject to the Boost Software
// License, Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)

// Request implementation dtails

// This header should be included only after the communicator and request 
// classes has been defined.
#ifndef BOOST_MPI_REQUEST_HANDLERS_HPP
#define BOOST_MPI_REQUEST_HANDLERS_HPP

namespace boost { namespace mpi {

struct request::legacy_handler : public request::handler {
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

template<typename T>
struct request::legacy_serialized_handler 
  : public request::legacy_handler {
  legacy_serialized_handler(communicator const& comm, int source, int tag, T& value)
    : legacy_handler(comm, source, tag, value) {}
};

template<typename T>
struct request::legacy_serialized_array_handler 
  : public request::legacy_handler {
  legacy_serialized_array_handler(communicator const& comm, int source, int tag, T* values, int n)
    : legacy_handler(comm, source, tag, values, n) {}
};

template<typename T, class A>
struct request::legacy_dynamic_primitive_array_handler 
  : public request::legacy_handler {
  legacy_dynamic_primitive_array_handler(communicator const& comm, int source, int tag, std::vector<T,A>& values)
    : legacy_handler(comm, source, tag, values, mpl::true_()) {}
};

struct request::trivial_handler : public request::handler {
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

struct request::dynamic_handler : public request::handler {
  dynamic_handler();
  
  status wait();
  optional<status> test();
  void cancel();
  
  bool active() const;
  optional<MPI_Request&> trivial();
  
  MPI_Request& size_request();
  MPI_Request& payload_request();
  
  boost::shared_ptr<void> data();
  void                    set_data(boost::shared_ptr<void> d);
  
  MPI_Request      m_requests[2];
  shared_ptr<void> m_data;
};

template<typename T> 
request request::make_serialized(communicator const& comm, int source, int tag, T& value) {
  return request(new legacy_serialized_handler<T>(comm, source, tag, value));
}

template<typename T>
request request::make_serialized_array(communicator const& comm, int source, int tag, T* values, int n) {
  return request(new legacy_serialized_array_handler<T>(comm, source, tag, values, n));
}

template<typename T, class A>
request request::make_dynamic_primitive_array(communicator const& comm, int source, int tag, 
                                              std::vector<T,A>& values) {
  return request(new legacy_dynamic_primitive_array_handler<T,A>(comm, source, tag, values));
}

namespace detail {
  /**
   * Internal data structure that stores everything required to manage
   * the receipt of serialized data via a request object.
   */
  template<typename T>
  struct serialized_irecv_data
  {
    serialized_irecv_data(const communicator& comm, int source, int tag, 
                          T& value)
      : comm(comm), source(source), tag(tag), ia(comm), value(value) 
    { 
    }

    void deserialize(status& stat) 
    { 
      ia >> value; 
      stat.m_count = 1;
    }

    communicator comm;
    int source;
    int tag;
    std::size_t count;
    packed_iarchive ia;
    T& value;
  };

  template<>
  struct serialized_irecv_data<packed_iarchive>
  {
    serialized_irecv_data(const communicator& comm, int source, int tag, 
                          packed_iarchive& ia)
      : comm(comm), source(source), tag(tag), ia(ia) { }

    void deserialize(status&) { /* Do nothing. */ }

    communicator comm;
    int source;
    int tag;
    std::size_t count;
    packed_iarchive& ia;
  };

  /**
   * Internal data structure that stores everything required to manage
   * the receipt of an array of serialized data via a request object.
   */
  template<typename T>
  struct serialized_array_irecv_data
  {
    serialized_array_irecv_data(const communicator& comm, int source, int tag, 
                                T* values, int n)
      : comm(comm), source(source), tag(tag), ia(comm), values(values), n(n)
    { 
    }

    void deserialize(status& stat);

    communicator comm;
    int source;
    int tag;
    std::size_t count;
    packed_iarchive ia;
    T* values;
    int n;
  };

  template<typename T>
  void serialized_array_irecv_data<T>::deserialize(status& stat)
  {
    // Determine how much data we are going to receive
    int count;
    ia >> count;
    
    // Deserialize the data in the message
    boost::serialization::array_wrapper<T> arr(values, count > n? n : count);
    ia >> arr;
    
    if (count > n) {
      boost::throw_exception(
        std::range_error("communicator::recv: message receive overflow"));
    }
    
    stat.m_count = count;
  }

  /**
   * Internal data structure that stores everything required to manage
   * the receipt of an array of primitive data but unknown size.
   * Such an array can have been send with blocking operation and so must
   * be compatible with the (size_t,raw_data[]) format.
   */
  template<typename T, class A>
  struct dynamic_array_irecv_data
  {
    BOOST_STATIC_ASSERT_MSG(is_mpi_datatype<T>::value, "Can only be specialized for MPI datatypes.");

    dynamic_array_irecv_data(const communicator& comm, int source, int tag, 
                             std::vector<T,A>& values)
      : comm(comm), source(source), tag(tag), count(-1), values(values)
    { 
    }

    communicator comm;
    int source;
    int tag;
    std::size_t count;
    std::vector<T,A>& values;
  };

}

template<typename T>
optional<status> 
request::legacy_handler::handle_serialized_irecv(legacy_handler* self, request_action action)
{
  typedef detail::serialized_irecv_data<T> data_t;
  shared_ptr<data_t> data = static_pointer_cast<data_t>(self->m_data);

  if (action == ra_wait) {
    status stat;
    if (self->m_requests[1] == MPI_REQUEST_NULL) {
      // Wait for the count message to complete
      BOOST_MPI_CHECK_RESULT(MPI_Wait,
                             (self->m_requests, &stat.m_status));
      // Resize our buffer and get ready to receive its data
      data->ia.resize(data->count);
      BOOST_MPI_CHECK_RESULT(MPI_Irecv,
                             (data->ia.address(), data->ia.size(), MPI_PACKED,
                              stat.source(), stat.tag(), 
                              MPI_Comm(data->comm), self->m_requests + 1));
    }

    // Wait until we have received the entire message
    BOOST_MPI_CHECK_RESULT(MPI_Wait,
                           (self->m_requests + 1, &stat.m_status));

    data->deserialize(stat);
    return stat;
  } else if (action == ra_test) {
    status stat;
    int flag = 0;

    if (self->m_requests[1] == MPI_REQUEST_NULL) {
      // Check if the count message has completed
      BOOST_MPI_CHECK_RESULT(MPI_Test,
                             (self->m_requests, &flag, &stat.m_status));
      if (flag) {
        // Resize our buffer and get ready to receive its data
        data->ia.resize(data->count);
        BOOST_MPI_CHECK_RESULT(MPI_Irecv,
                               (data->ia.address(), data->ia.size(),MPI_PACKED,
                                stat.source(), stat.tag(), 
                                MPI_Comm(data->comm), self->m_requests + 1));
      } else
        return optional<status>(); // We have not finished yet
    } 

    // Check if we have received the message data
    BOOST_MPI_CHECK_RESULT(MPI_Test,
                           (self->m_requests + 1, &flag, &stat.m_status));
    if (flag) {
      data->deserialize(stat);
      return stat;
    } else 
      return optional<status>();
  } else {
    return optional<status>();
  }
}

template<typename T>
optional<status> 
request::legacy_handler::handle_serialized_array_irecv(legacy_handler* self, request_action action)
{
  typedef detail::serialized_array_irecv_data<T> data_t;
  shared_ptr<data_t> data = static_pointer_cast<data_t>(self->m_data);

  if (action == ra_wait) {
    status stat;
    if (self->m_requests[1] == MPI_REQUEST_NULL) {
      // Wait for the count message to complete
      BOOST_MPI_CHECK_RESULT(MPI_Wait,
                             (self->m_requests, &stat.m_status));
      // Resize our buffer and get ready to receive its data
      data->ia.resize(data->count);
      BOOST_MPI_CHECK_RESULT(MPI_Irecv,
                             (data->ia.address(), data->ia.size(), MPI_PACKED,
                              stat.source(), stat.tag(), 
                              MPI_Comm(data->comm), self->m_requests + 1));
    }

    // Wait until we have received the entire message
    BOOST_MPI_CHECK_RESULT(MPI_Wait,
                           (self->m_requests + 1, &stat.m_status));

    data->deserialize(stat);
    return stat;
  } else if (action == ra_test) {
    status stat;
    int flag = 0;

    if (self->m_requests[1] == MPI_REQUEST_NULL) {
      // Check if the count message has completed
      BOOST_MPI_CHECK_RESULT(MPI_Test,
                             (self->m_requests, &flag, &stat.m_status));
      if (flag) {
        // Resize our buffer and get ready to receive its data
        data->ia.resize(data->count);
        BOOST_MPI_CHECK_RESULT(MPI_Irecv,
                               (data->ia.address(), data->ia.size(),MPI_PACKED,
                                stat.source(), stat.tag(), 
                                MPI_Comm(data->comm), self->m_requests + 1));
      } else
        return optional<status>(); // We have not finished yet
    } 

    // Check if we have received the message data
    BOOST_MPI_CHECK_RESULT(MPI_Test,
                           (self->m_requests + 1, &flag, &stat.m_status));
    if (flag) {
      data->deserialize(stat);
      return stat;
    } else 
      return optional<status>();
  } else {
    return optional<status>();
  }
}

template<typename T, class A>
optional<status> 
request::legacy_handler::handle_dynamic_primitive_array_irecv(legacy_handler* self, request_action action)
{
  typedef detail::dynamic_array_irecv_data<T,A> data_t;
  shared_ptr<data_t> data = static_pointer_cast<data_t>(self->m_data);

  if (action == ra_wait) {
    status stat;
    if (self->m_requests[1] == MPI_REQUEST_NULL) {
      // Wait for the count message to complete
      BOOST_MPI_CHECK_RESULT(MPI_Wait,
                             (self->m_requests, &stat.m_status));
      // Resize our buffer and get ready to receive its data
      data->values.resize(data->count);
      BOOST_MPI_CHECK_RESULT(MPI_Irecv,
                             (&(data->values[0]), data->values.size(), get_mpi_datatype<T>(),
                              stat.source(), stat.tag(), 
                              MPI_Comm(data->comm), self->m_requests + 1));
    }
    // Wait until we have received the entire message
    BOOST_MPI_CHECK_RESULT(MPI_Wait,
                           (self->m_requests + 1, &stat.m_status));
    return stat;
  } else if (action == ra_test) {
    status stat;
    int flag = 0;

    if (self->m_requests[1] == MPI_REQUEST_NULL) {
      // Check if the count message has completed
      BOOST_MPI_CHECK_RESULT(MPI_Test,
                             (self->m_requests, &flag, &stat.m_status));
      if (flag) {
        // Resize our buffer and get ready to receive its data
        data->values.resize(data->count);
        BOOST_MPI_CHECK_RESULT(MPI_Irecv,
                               (&(data->values[0]), data->values.size(),MPI_PACKED,
                                stat.source(), stat.tag(), 
                                MPI_Comm(data->comm), self->m_requests + 1));
      } else
        return optional<status>(); // We have not finished yet
    } 

    // Check if we have received the message data
    BOOST_MPI_CHECK_RESULT(MPI_Test,
                           (self->m_requests + 1, &flag, &stat.m_status));
    if (flag) {
      return stat;
    } else 
      return optional<status>();
  } else {
    return optional<status>();
  }
}

template<typename T>
request::legacy_handler::legacy_handler(communicator const& comm, int source, int tag, T& value)
: m_data(new detail::serialized_irecv_data<T>(comm, source, tag, value)),
  m_handler(handle_serialized_irecv<T>) {
  m_requests[0] = MPI_REQUEST_NULL;
  m_requests[1] = MPI_REQUEST_NULL;
  std::size_t& count = data<detail::serialized_irecv_data<T> >()->count;
  BOOST_MPI_CHECK_RESULT(MPI_Irecv,
                         (&count, 1, 
                          get_mpi_datatype(count),
                          source, tag, comm, &size_request()));

}

template<typename T>
request::legacy_handler::legacy_handler(communicator const& comm, int source, int tag, T* values, int n)
  : m_data(new detail::serialized_array_irecv_data<T>(comm, source, tag, values, n)),
    m_handler(handle_serialized_array_irecv<T>) {
  m_requests[0] = MPI_REQUEST_NULL;
  m_requests[1] = MPI_REQUEST_NULL;
  std::size_t& count = data<detail::serialized_array_irecv_data<T> >()->count;
  BOOST_MPI_CHECK_RESULT(MPI_Irecv,
			 (&count, 1, 
                          get_mpi_datatype(count),
                          source, tag, comm, &size_request()));
}

template<typename T, class A>
request::legacy_handler::legacy_handler(communicator const& comm, int source, int tag, std::vector<T,A>& values, mpl::true_ primitive)
  : m_data(new detail::dynamic_array_irecv_data<T,A>(comm, source, tag, values)),
    m_handler(handle_dynamic_primitive_array_irecv<T,A>)
{
  m_requests[0] = MPI_REQUEST_NULL;
  m_requests[1] = MPI_REQUEST_NULL;
  std::size_t& count = data<detail::dynamic_array_irecv_data<T,A> >()->count;
  BOOST_MPI_CHECK_RESULT(MPI_Irecv,
                         (&count, 1, 
                          get_mpi_datatype(count),
                          source, tag, comm, &size_request()));
}

}}

#endif // BOOST_MPI_REQUEST_HANDLERS_HPP
