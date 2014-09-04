// Copyright (C) 2014 Alain Miniussi.

// Use, modification and distribution is subject to the Boost Software
// License, Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
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
  std::vector<int> p(periodic.begin(), periodic.end());
  BOOST_MPI_CHECK_RESULT(MPI_Cart_create, 
                         ((MPI_Comm)comm, dims.size(), 
                          const_cast<int*>(dims.data()), p.data(), 
                          int(reorder), &newcomm));
  if(newcomm != MPI_COMM_NULL) {
    comm_ptr.reset(new MPI_Comm(newcomm), comm_free());
  }
}
} } // end namespace boost::mpi
