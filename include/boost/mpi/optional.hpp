// Copyright (C) 2025 Alain Miniussi <alain.miniussi -at- oca.eu>.

// Use, modification and distribution is subject to the Boost Software
// License, Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_MPI_OPTIONAL_HPP
#define BOOST_MPI_OPTIONAL_HPP

#include <boost/mpi/config.hpp>

#if defined(BOOST_NO_CXX17_HDR_OPTIONAL) || defined(BOOST_MPI_FORCE_BOOST_OPTIONAL)
#define BOOST_MPI_USE_BOOST_OPTIONAL
#endif

#if defined(BOOST_MPI_USE_BOOST_OPTIONAL)
#include <boost/optional.hpp>
#else
#include <optional>
#endif

namespace boost { namespace mpi {
#if defined(BOOST_MPI_USE_BOOST_OPTIONAL)
template <typename T> using optional = boost::optional<T>;
#else
template <typename T> using optional = std::optional<T>;
#endif
}}

#endif // BOOST_MPI_OPTIONAL_HPP
