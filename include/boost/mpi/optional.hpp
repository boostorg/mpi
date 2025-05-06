// Copyright (C) 2025 Alain Miniussi <alain.miniussi -at- oca.eu>.

// Use, modification and distribution is subject to the Boost Software
// License, Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_MPI_OPTIONAL_HPP
#define BOOST_MPI_OPTIONAL_HPP

#include <boost/mpi/config.hpp>

#include <boost/optional.hpp>

namespace boost { namespace mpi {
template <typename T> using optional = boost::optional<T>;
}}

#endif // BOOST_MPI_OPTIONAL_HPP
