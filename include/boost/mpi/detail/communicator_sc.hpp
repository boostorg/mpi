// Copyright (C) 2006 Douglas Gregor <doug.gregor -at- gmail.com>.

// Use, modification and distribution is subject to the Boost Software
// License, Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)

// Skeleton and content support for communicators

// This header should be included only after both communicator.hpp and
// skeleton_and_content.hpp have been included.
#ifndef BOOST_MPI_COMMUNICATOR_SC_HPP
#define BOOST_MPI_COMMUNICATOR_SC_HPP

namespace boost { namespace mpi {

template<typename T>
void
communicator::send(int dest, int tag, const skeleton_proxy<T>& proxy) const
{
  packed_skeleton_oarchive ar(*this);
  ar << proxy.object;
  send(dest, tag, ar);
}

template<typename T>
status
communicator::recv(int source, int tag, const skeleton_proxy<T>& proxy) const
{
  packed_skeleton_iarchive ar(*this);
  status result = recv(source, tag, ar);
  ar >> proxy.object;
  return result;
}

template<typename T>
status communicator::recv(int source, int tag, skeleton_proxy<T>& proxy) const
{
  packed_skeleton_iarchive ar(*this);
  status result = recv(source, tag, ar);
  ar >> proxy.object;
  return result;
}

template<typename T>
request
communicator::isend(int dest, int tag, const skeleton_proxy<T>& proxy) const
{
  shared_ptr<packed_skeleton_oarchive> archive(new packed_skeleton_oarchive(*this));

  *archive << proxy.object;
  request result = isend(dest, tag, *archive);
  result.m_data = archive;
  return result;
}

template<class T>
struct request::probe_info_const_skeleton_proxy : public request::probe_info_base {
  probe_info_const_skeleton_proxy(int source, int tag, const MPI_Comm& comm, const skeleton_proxy<T>& proxy)
    : probe_info_base(source, tag, comm),
      m_isa(comm),
      m_archive(m_isa.get_skeleton()),
      m_proxy(proxy) {}
  
  packed_iarchive& archive() { return m_archive;}
  void deserialize(status& stat) {
    m_isa >> m_proxy.object;
    stat.m_count = 1;
  }
  
  packed_skeleton_iarchive m_isa;
  packed_iarchive&         m_archive;
  skeleton_proxy<T>        m_proxy;
};

template<class T>
struct request::probe_info_skeleton_proxy : public request::probe_info_const_skeleton_proxy<T> {
  typedef probe_info_const_skeleton_proxy<T> super;
  probe_info_skeleton_proxy(int source, int tag, const MPI_Comm& comm, const skeleton_proxy<T>& proxy)
    : super(source, tag, comm, proxy) {}
};

template<class T>
request::request(int source, int tag, MPI_Comm const& comm, const skeleton_proxy<T>& proxy) 
  : m_request(), 
    m_probe_info(new probe_info_const_skeleton_proxy<T>(source, tag, comm, proxy)),
    m_data() {}

template<class T>
request::request(int source, int tag, MPI_Comm const& comm, skeleton_proxy<T>& proxy) 
  : m_request(), 
    m_probe_info(new probe_info_skeleton_proxy<T>(source, tag, comm, proxy)),
    m_data() {}

// namespace detail {
//   template<typename T>
//   struct serialized_irecv_data<const skeleton_proxy<T> >
//   {
//     serialized_irecv_data(const communicator& comm, int source, int tag, 
//                           skeleton_proxy<T> proxy)
//       : comm(comm), source(source), tag(tag), isa(comm), 
//         ia(isa.get_skeleton()), proxy(proxy) { }

//     void deserialize(status& stat) 
//     { 
//       isa >> proxy.object;
//       stat.m_count = 1;
//     }

//     communicator comm;
//     int source;
//     int tag;
//     std::size_t count;
//     packed_skeleton_iarchive isa;
//     packed_iarchive& ia;
//     skeleton_proxy<T> proxy;
//   };

//   template<typename T>
//   struct serialized_irecv_data<skeleton_proxy<T> >
//     : public serialized_irecv_data<const skeleton_proxy<T> >
//   {
//     typedef serialized_irecv_data<const skeleton_proxy<T> > inherited;

//     serialized_irecv_data(const communicator& comm, int source, int tag, 
//                           const skeleton_proxy<T>& proxy)
//       : inherited(comm, source, tag, proxy) { }
//   };
// }

} } // end namespace boost::mpi

#endif // BOOST_MPI_COMMUNICATOR_SC_HPP

