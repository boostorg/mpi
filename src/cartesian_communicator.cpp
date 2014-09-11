
//          Copyright Alain Miniussi 2014.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE_1_0.txt or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

// Authors: Alain Miniussi

#include <algorithm>

#include <boost/mpi/cartesian_communicator.hpp>

namespace boost { namespace mpi {

cartesian_communicator::cartesian_communicator(const communicator&      comm,
                                               const std::vector<int>&  dims,
                                               const std::vector<bool>& periodic,
                                               bool                     reorder )
  : communicator() 
{
  BOOST_ASSERT(dims.size() == periodic.size());
  MPI_Comm newcomm;
  std::vector<int> local_dims = dims;
  // Fill the gaps, if any
  if (std::count(local_dims.begin(), local_dims.end(), 0) > 0) {
    cartesian_dimensions(comm, local_dims);
  }
  std::vector<int> p(periodic.begin(), periodic.end());
  BOOST_MPI_CHECK_RESULT(MPI_Cart_create, 
                         ((MPI_Comm)comm, dims.size(), 
                          local_dims.data(), p.data(), 
                          int(reorder), &newcomm));
  if(newcomm != MPI_COMM_NULL) {
    comm_ptr.reset(new MPI_Comm(newcomm), comm_free());
  }
}

cartesian_communicator::cartesian_communicator(const cartesian_communicator& comm,
                                               const std::vector<int>&       keep ) 
  : communicator() 
{
  int max_dims = comm.ndims();
  BOOST_ASSERT(keep.size() <= max_dims);
  std::vector<int> bitset(max_dims, int(false));
  for(int i = 0; i < keep.size(); ++i) {
    BOOST_ASSERT(keep[i] < max_dims);
    bitset[keep[i]] = true;
  }
  
  MPI_Comm newcomm;
  BOOST_MPI_CHECK_RESULT(MPI_Cart_sub, 
                         ((MPI_Comm)comm, bitset.data(), &newcomm));
  if(newcomm != MPI_COMM_NULL) {
    comm_ptr.reset(new MPI_Comm(newcomm), comm_free());
  }
}

int
cartesian_communicator::ndims() const {
  int n = -1;
  BOOST_MPI_CHECK_RESULT(MPI_Cartdim_get, 
                         (MPI_Comm(*this), &n));
  return n;
}

int
cartesian_communicator::rank(std::vector<int> const& coords ) const {
  int r = -1;
  BOOST_ASSERT(coords.size() == ndims());
  BOOST_MPI_CHECK_RESULT(MPI_Cart_rank, 
                         (MPI_Comm(*this), const_cast<std::vector<int>&>(coords).data(), 
                          &r));
  return r;
}

std::vector<int>&
cartesian_communicator::coords(int rk, std::vector<int>& cbuf) const {
  cbuf.resize(ndims());
  BOOST_MPI_CHECK_RESULT(MPI_Cart_coords, 
                         (MPI_Comm(*this), rk, cbuf.size(), cbuf.data() ));
  return cbuf;
}

std::vector<int>
cartesian_communicator::coords(int rk) const {
  std::vector<int> coords;
  this->coords(rk, coords);
  return coords;
}

void
cartesian_communicator::topology( std::vector<int>&  dims,
                                  std::vector<bool>& periodic,
                                  std::vector<int>&  coords ) const {
  int ndims = this->ndims();
  dims.resize(ndims);
  periodic.resize(ndims);
  coords.resize(ndims);
  std::vector<int> cperiods(ndims);
  BOOST_MPI_CHECK_RESULT(MPI_Cart_get,
                         (MPI_Comm(*this), ndims, dims.data(), cperiods.data(), coords.data()));
  std::copy(cperiods.begin(), cperiods.end(), periodic.begin());
}

std::vector<int>&
cartesian_dimensions(int sz, std::vector<int>&  dims) {
  BOOST_MPI_CHECK_RESULT(MPI_Dims_create,
                         (sz, dims.size(), dims.data()));
  return dims;
}

} } // end namespace boost::mpi
