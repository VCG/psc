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

#ifndef ZI_GL_DETAIL_CONFIG_HPP
#define ZI_GL_DETAIL_CONFIG_HPP 1

#include <zi/config/config.hpp>

#if defined( ZI_OS_WINDOWS )
#  // damn, windows.h defines min and mac macros
#  // which make numerical_limits< ... >::min() / max() unusable!
#  ifndef NOMINMAX
#    define NOMINMAX 1
#  endif
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#    include <windows.h>
#    undef  WIN32_LEAN_AND_MEAN
#  else
#    include <windows.h>
#  endif
#  undef NOMINMAX
#
#endif

#ifndef APIENTRY
#  define APIENTRY /* */
#endif

#endif
