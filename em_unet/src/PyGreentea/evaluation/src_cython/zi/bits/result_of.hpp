//
// Copyright (C) 2010  Aleksandar Zlateski <zlateski@mit.edu>
// ----------------------------------------------------------
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.
//

#ifndef ZI_BITS_RESULT_OF_HPP
#define ZI_BITS_RESULT_OF_HPP 1

#include <zi/config/config.hpp>

#ifdef __GXX_EXPERIMENTAL_CXX0X__
#  include <functional>
#  define ZI_RESULT_OF_NAMESPACE ::std
#else
#  if defined( ZI_USE_TR1 ) || defined( ZI_NO_BOOST )
#    include <tr1/functional>
#    define ZI_RESULT_OF_NAMESPACE ::std::tr1
#  else
#    include <boost/utility/result_of.hpp>
#    define ZI_RESULT_OF_NAMESPACE ::boost
#  endif
#endif

namespace zi {

using ZI_RESULT_OF_NAMESPACE::result_of;

} // namespace zi

#undef ZI_RESULT_OF_NAMESPACE
#endif
