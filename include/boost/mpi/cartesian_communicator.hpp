// Copyright (C) 2007 Trustees of Indiana University

// Authors: Alain Miniussi

// Use, modification and distribution is subject to the Boost Software
// License, Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)

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

// Headers required to implement cartesian topologies
#include <boost/shared_array.hpp>
#include <boost/assert.hpp>

namespace boost { namespace mpi {

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
   *  @param cartesian Any type that meets the requirements of the
   *  Incidence Cartesian and Vertex List Cartesian concepts from the Boost Cartesian
   *  Library. This structure of this cartesian will become the topology
   *  of the communicator that is returned.
   *
   *  @param reorder Whether MPI is permitted to re-order the process
   *  ranks within the returned communicator, to better optimize
   *  communication. If false, the ranks of each process in the
   *  returned process will match precisely the rank of that process
   *  within the original communicator.
   */
  cartesian_communicator(const communicator&      comm,
                         const std::vector<int>&  dims,
                         const std::vector<bool>& periodic,
                         bool                     reorder = false);
};

} } // end namespace boost::mpi



#endif // BOOST_MPI_CARTESIAN_COMMUNICATOR_HPP
