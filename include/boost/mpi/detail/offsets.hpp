//          Copyright Alain Miniussi 2014.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE_1_0.txt or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

// Authors: Alain Miniussi
#ifndef BOOST_MPI_OFFSETS_HPP
#define BOOST_MPI_OFFSETS_HPP

namespace boost { namespace mpi {
namespace detail {

// Convert a sequence of sizes [S0..Sn] to a sequence displacement 
// [O0..On] where O[0] = 0 and O[k+1] = O[k]+S[k].
template<class Alloc1, class Alloc2>
void
sizes2offsets(std::vector<int, Alloc1> const& sizes, std::vector<int, Alloc2>& offsets) 
{
  offsets.resize(sizes.size());
  offsets[0] = 0;
  for(int i = 0; i < sizes.size()-1; ++i) {
    offsets[i+1] = offsets[i] + sizes[i];
  }
}


}
}}
#endif // BOOST_MPI_OFFSETS_HPP
