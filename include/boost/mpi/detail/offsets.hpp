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
void
sizes2offsets(int const* sizes, int* offsets, int n) 
{
  offsets[0] = 0;
  for(int i = 1; i < n; ++i) {
    offsets[i] = offsets[i-1] + sizes[i-1];
  }
}

// Convert a sequence of sizes [S0..Sn] to a sequence displacement 
// [O0..On] where O[0] = 0 and O[k+1] = O[k]+S[k].
void
sizes2offsets(std::vector<int> const& sizes, std::vector<int>& offsets) 
{
  int sz = sizes.size();
  offsets.resize(sz);
  sizes2offsets(sizes.data(), offsets.data(), sz);
}

// Given a sequence of sizes (typically the number of records dispatched
// to each process in a scater) and a sequence of displacements (typically the
// slot index at with those record starts), convert the later to a number 
// of skipped slots.
void
offsets2skipped(int const* sizes, int const* offsets, int* skipped, int n) 
{
  skipped[0] = 0;
  for(int i = 1; i < n; ++i) {
    skipped[i] -= offsets[i-1] + sizes[i-1];
  }
}


}
}}
#endif // BOOST_MPI_OFFSETS_HPP
