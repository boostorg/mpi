
//          Copyright Alain Miniussi 2014.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE_1_0.txt or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

// Authors: Alain Miniussi

/** @file cartesian_communicator.hpp
 *
 *  This header defines facilities to support MPI communicators with
 *  cartesian topologies.
 *  If known at compiled time, the dimension of the implied grid 
 *  can be statically enforced, through the templatized communicator 
 *  class. Otherwise, a non template, dynamic, base class is provided.
 *  
 */
#ifndef BOOST_MPI_CARTESIAN_COMMUNICATOR_HPP
#define BOOST_MPI_CARTESIAN_COMMUNICATOR_HPP

#include <boost/mpi/communicator.hpp>

#include <vector>
#include <utility>
#include <iostream>

// Headers required to implement cartesian topologies
#include <boost/shared_array.hpp>
#include <boost/assert.hpp>

namespace boost { namespace mpi {

struct cartesian_dimension {

  int size;
  bool periodic;
  cartesian_dimension(int sz = 0, bool p = false) : size(sz), periodic(p) {}
private:
  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive & ar, const unsigned int version)
  {
    ar & size & periodic;
  }

};

template <>
struct is_mpi_datatype<cartesian_dimension> : mpl::true_ { };

bool
operator==(cartesian_dimension const& d1, cartesian_dimension const& d2) {
  return d1.size == d2.size && d1.periodic == d2.periodic;
}

bool
operator!=(cartesian_dimension const& d1, cartesian_dimension const& d2) {
  return !(d1 == d2);
}

std::ostream& operator<<(std::ostream& out, cartesian_dimension const& d);

class BOOST_MPI_DECL cartesian_topology 
  : private std::vector<cartesian_dimension> {
  friend class cartesian_communicator;
  typedef std::vector<cartesian_dimension> super;
 public:
  using super::operator[];
  using super::size;
  using super::begin;
  using super::end;
  using super::swap;

  cartesian_topology(int sz) 
    : super(sz) {}

  template<int NDIM>
  explicit cartesian_topology(array<cartesian_dimension, NDIM> const& dims)
    : super(NDIM) {
    std::copy(dims.begin(), dims.end(), begin());
  }
  
  template<class DimIter, class PerIter>
  cartesian_topology(DimIter dim_iter, PerIter period_iter, int ndim) 
    : super(ndim) {
    for(int i = 0; i < ndim; ++i) {
      (*this)[i] = cartesian_dimension(*dim_iter++, *period_iter++);
    }
  }
  
  void split(std::vector<int>& dims, std::vector<bool>& periodics) const;
};

/**
 * @brief An MPI communicator with a cartesian topology.
 *
 * A @c cartesian_communicator is a communicator whose topology is
 * expressed as a grid. Cartesian communicators have the same
 * functionality as (intra)communicators, but also allow one to query
 * the relationships among processes and the properties of the grid.
 */
class BOOST_MPI_DECL cartesian_communicator : public communicator
{
  friend class communicator;

  /**
   * INTERNAL ONLY
   *
   * Construct a cartesian communicator given a shared pointer to the
   * underlying MPI_Comm (which must have a cartesian topology).
   * This operation is used for "casting" from a communicator to 
   * a cartesian communicator.
   */
  explicit cartesian_communicator(const shared_ptr<MPI_Comm>& comm_ptr)
    : communicator()
  {
    this->comm_ptr = comm_ptr;
    BOOST_ASSERT(has_cartesian_topology());    
  }

public:
  /**
   * Build a new Boost.MPI cartesian communicator based on the MPI
   * communicator @p comm with cartesian topology.
   *
   * @p comm may be any valid MPI communicator. If @p comm is
   * MPI_COMM_NULL, an empty communicator (that cannot be used for
   * communication) is created and the @p kind parameter is
   * ignored. Otherwise, the @p kind parameter determines how the
   * Boost.MPI communicator will be related to @p comm:
   *
   *   - If @p kind is @c comm_duplicate, duplicate @c comm to create
   *   a new communicator. This new communicator will be freed when
   *   the Boost.MPI communicator (and all copies of it) is
   *   destroyed. This option is only permitted if the underlying MPI
   *   implementation supports MPI 2.0; duplication of
   *   intercommunicators is not available in MPI 1.x.
   *
   *   - If @p kind is @c comm_take_ownership, take ownership of @c
   *   comm. It will be freed automatically when all of the Boost.MPI
   *   communicators go out of scope.
   *
   *   - If @p kind is @c comm_attach, this Boost.MPI communicator
   *   will reference the existing MPI communicator @p comm but will
   *   not free @p comm when the Boost.MPI communicator goes out of
   *   scope. This option should only be used when the communicator is
   *   managed by the user.
   */
  cartesian_communicator(const MPI_Comm& comm, comm_create_kind kind)
    : communicator(comm, kind)
  { 
    BOOST_ASSERT(has_cartesian_topology());
  }

  /**
   *  Create a new communicator whose topology is described by the
   *  given cartesian. The indices of the vertices in the cartesian will be
   *  assumed to be the ranks of the processes within the
   *  communicator. There may be fewer vertices in the cartesian than
   *  there are processes in the communicator; in this case, the
   *  resulting communicator will be a NULL communicator.
   *
   *  @param comm The communicator that the new, cartesian communicator
   *  will be based on. 
   * 
   *  @param dims the cartesian dimension of the new communicator. The size indicate 
   *  the number of dimension. Some dimensions be set to zero, in which case
   *  the corresponding dimension value is left to the system.
   *  
   *  @param reorder Whether MPI is permitted to re-order the process
   *  ranks within the returned communicator, to better optimize
   *  communication. If false, the ranks of each process in the
   *  returned process will match precisely the rank of that process
   *  within the original communicator.
   */
  cartesian_communicator(const communicator&       comm,
                         const cartesian_topology& dims,
                         bool                      reorder = false);
  
  /**
   * Create a new cartesian communicator whose topology is a subset of
   * an existing cartesian cimmunicator.
   * @param comm the original communicator.
   * @param keep and array containiing the dimension to keep from the existing 
   * communicator.
   */
  cartesian_communicator(const cartesian_communicator& comm,
                         const std::vector<int>&       keep );
    
  using communicator::rank;

  /** 
   * Retrive the number of dimension of the underlying toppology.
   */
  int ndims() const;
  
  /**
   * Return the rank of the process at the given coordinates.
   * @param coords the coordinates. the size must match the communicator's topology.
   */
  int rank(const std::vector<int>& coords) const;
  /**
   * Provides the coordinates of a process with the given rank.
   * @param rk the rank in this communicator.
   * @param cbuf a buffer were to store the coordinates.
   * @returns a reference to cbuf.
   */
  std::vector<int>& coords(int rk, std::vector<int>& cbuf) const;
  /**
   * Provides the coordinates of the process with the given rank.
   * @param rk the ranks in this communicator.
   * @returns the coordinates.
   */
  std::vector<int> coords(int rk) const;
  /**
   * Retrieve the topology.
   *
   */
  void topology( cartesian_topology&  dims, std::vector<int>& coords ) const;
};

/**
 * Given en number of processes, and a partially filled sequence 
 * of dimension, try to complete the dimension sequence.
 * @param nb_proc the numer of mpi processes.fill a sequence of dimension.
 * @param dims a sequence of positive or null dimensions. Non zero dimension 
 *  will be left untouched.
 */
std::vector<int>& cartesian_dimensions(int nb_proc, std::vector<int>&  dims);

/**
 * Given en communicator and a partially filled sequence 
 * of dimension, try to complete the dimension sequence to produce an acceptable
 * cartesian topology.
 * @param comm the prospective parent communicator.
 * @param dims a sequence of positive or null dimensions. Non zero dimension 
 *  will be left untouched.
 */
inline
std::vector<int>& cartesian_dimensions(const communicator& comm, std::vector<int>&  dims) {
  return cartesian_dimensions(comm.size(), dims);
}

} } // end namespace boost::mpi

#endif // BOOST_MPI_CARTESIAN_COMMUNICATOR_HPP
