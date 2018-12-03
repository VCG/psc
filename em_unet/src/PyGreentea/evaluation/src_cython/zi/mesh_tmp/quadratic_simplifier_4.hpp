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

#ifndef ZI_MESH_QUADRATIC_SIMPLIFIER_HPP
#define ZI_MESH_QUADRATIC_SIMPLIFIER_HPP 1

#include <zi/bits/shared_ptr.hpp>
#include <zi/bits/unordered_set.hpp>
#include <zi/utility/static_assert.hpp>

#include <zi/heap/binary_heap.hpp>
#include <zi/mesh/tri_list.hpp>
#include <zi/mesh/tri_mesh.hpp>
#include <zi/mesh/detail/quadratic.hpp>
#include <zi/mesh/detail/qmetric.hpp>

#include <zi/vl/vec.hpp>
#include <zi/vl/quat.hpp>

#include <vector>
#include <iostream>
#include <functional>

namespace zi {
namespace mesh {

template< class Float >
class simplifier: non_copyable
{
private:
    ZI_STATIC_ASSERT( is_floating_point< Float >::value, non_floating_point_mesh_simplifier );

    typedef vl::vec< Float, 3 >       coord_t     ;
    typedef qmetric< Float, 6 >       qmetric_t   ;

    std::size_t                 size_    ;
    mesh::tri_mesh              mesh_cnt_   ;
    std::vector< coord_t >      points_cnt_ ;
    std::vector< coord_t >      normals_cnt_;

    mesh::tri_mesh&             mesh_      ;
    std::vector< coord_t >&     points_    ;
    std::vector< coord_t >&     normals_   ;

    std::vector< qmetric_t >    quadratic_;

    unordered_set< uint64_t >   invalid_ ;

    struct heap_entry
    {
        uint64_t                  edge_   ;
        Float                     value_  ;
        const vl::vec< Float, 3 > optimal_;
        const vl::vec< Float, 3 > optimal_normal_;

        Float value() const
        {
            return value_;
        }

        heap_entry()
        {
        }

        heap_entry( const uint64_t e, const Float v,
                    const vl::vec< Float, 3 >& p,
                    const vl::vec< Float, 3 >& q )
            : edge_( e ), value_( v ), optimal_( p ), optimal_normal_( q )
        {
        }

        uint32_t v0() const
        {
            return detail::edge_source( edge_ );
        }

        uint32_t v1() const
        {
            return detail::edge_sink( edge_ );
        }

        uint32_t source() const
        {
            return detail::edge_source( edge_ );
        }

        uint32_t sink() const
        {
            return detail::edge_sink( edge_ );
        }

    };

    friend struct heap_entry;

    typedef binary_heap<
        heap_entry,
        zi::heap::hashed_index<
            zi::heap::member_variable<
                heap_entry,
                uint64_t,
                &heap_entry::edge_
            >
        >,

        zi::heap::value<
            zi::heap::member_variable<
                heap_entry,
                Float,
                &heap_entry::value_
            >,
            std::less< Float >
        >
    > heap_type;

    heap_type heap_;

private:
    inline bool check_valid_edge( const uint64_t e ) const
    {
        return e && mesh_.valid_edge( e );
    }

    inline bool check_compactness( const uint64_t e, const vl::vec< Float, 3 >& p ) const
    {
        const Float min_compactness = 0.002;

        const uint32_t v0 = detail::edge_source( e );
        const uint32_t v1 = detail::edge_sink( e );

        const uint64_t einv = detail::make_edge( v1, v0 );

        const uint32_t tr = mesh_.across_edge( e );
        const uint32_t bl = mesh_.across_edge( einv );

        for ( uint32_t v = tr; v != bl; )
        {
            const uint32_t vn = mesh_.across_edge( v0, v );
            Float r = 1; //vl::triangle::compactness( p,
                         //                        points_[ v ],
                         //                        points_[ vn ] );
            if ( r < min_compactness )
            {
                return false;
            }
            v = vn;
        }

        for ( uint32_t v = bl; v != tr; )
        {
            const uint32_t vn = mesh_.across_edge( v1, v );
            Float r = 1; //vl::triangle::compactness( p,
            //                      points_[ v ],
            //                                   points_[ vn ] );
            if ( r < min_compactness )
            {
                return false;
            }
            v = vn;
        }

        return true;
    }

    inline bool check_inversion( const uint64_t e, const vl::vec< Float, 3 >& p )
    {
        //if ( invalid_.count( e ) )
        //{
        //return false;
        //}

        const uint32_t max_degree = 24;
        const Float    min_angle  = 0.01;

        const uint32_t v0 = detail::edge_source( e );
        const uint32_t v1 = detail::edge_sink( e );

        const uint64_t einv = detail::make_edge( v1, v0 );

        const uint32_t tr = mesh_.across_edge( e );
        const uint32_t bl = mesh_.across_edge( einv );

        uint32_t degree = 0;

        for ( uint32_t v = tr; v != bl; )
        {
            const uint32_t vn = mesh_.across_edge( v0, v );
            vl::vec< Float, 3 > a = points_[ vn ] - points_[ v ];

            if ( dot( cross( a, points_[ v0 ] - points_[ v ] ),
                      cross( a, p - points_[ v ] )) < min_angle )
            {
                return false;
            }

            v = vn;

            ++degree;
        }

        for ( uint32_t v = bl; v != tr; )
        {
            const uint32_t vn = mesh_.across_edge( v1, v );

            vl::vec< Float, 3 > a = points_[ vn ] - points_[ v ];

            if ( dot( cross( a, points_[ v1 ] - points_[ v ] ),
                      cross( a, p - points_[ v ] )) < min_angle )
            {
                return false;
            }

            v = vn;
            ++degree;
        }

        return degree < max_degree;
    }

    inline bool check_topology( const uint64_t e )
    {

        if ( invalid_.count( e ) )
        {
            return false;
        }

        const uint32_t v0 = detail::edge_source( e );
        const uint32_t v1 = detail::edge_sink( e );

        const uint32_t tr = mesh_.across_edge( e );
        const uint32_t bl = mesh_.across_edge( v1, v0 );

        if ( bl == tr )
        {
            return false;
        }

        for ( uint32_t v = mesh_.across_edge( v0, tr );
              v != bl;
              v = mesh_.across_edge( v0, v ) )
        {
            if ( mesh_.has_edge( v1, v ) )
            {
                invalid_.insert( e );
                return false;
            }
        }

        return true;
    }

public:

    simplifier()
        : size_( 0 ),
          mesh_cnt_( 0 ),
          points_cnt_( 0 ),
          normals_cnt_( 0 ),
          mesh_( mesh_cnt_ ),
          points_( points_cnt_ ),
          normals_( normals_cnt_ ),
          quadratic_( 0 ),
          invalid_(),
          heap_()
    {
    }

    explicit simplifier( std::size_t s )
        : size_( s ),
          mesh_cnt_( s ),
          points_cnt_( s ),
          normals_cnt_( s ),
          mesh_( mesh_cnt_ ),
          points_( points_cnt_ ),
          normals_( normals_cnt_ ),
          quadratic_( s ),
          invalid_(),
          heap_()
    {
    }

    explicit simplifier( mesh::tri_mesh& m )
        : size_( m.size() ),
          mesh_cnt_(),
          points_cnt_( size_ ),
          normals_cnt_( size_ ),
          mesh_( m ),
          points_( points_cnt_ ),
          normals_( normals_cnt_ ),
          quadratic_( size_ ),
          invalid_(),
          heap_()
    {
    }

    explicit simplifier( mesh::tri_mesh& m, std::vector< coord_t >& v )
        : size_( m.size() ),
          mesh_cnt_(),
          points_cnt_(),
          normals_cnt_( size_ ),
          mesh_( m ),
          points_( v ),
          normals_( normals_cnt_ ),
          quadratic_( size_ ),
          invalid_(),
          heap_()
    {
        v.resize( m.size() );
    }

    explicit simplifier( mesh::tri_mesh& m,
                         std::vector< coord_t >& v,
                         std::vector< coord_t >& n )
        : size_( m.size() ),
          mesh_cnt_(),
          points_cnt_(),
          normals_cnt_(),
          mesh_( m ),
          points_( v ),
          normals_( n ),
          quadratic_( size_ ),
          invalid_(),
          heap_()
    {
        v.resize( m.size() );
        n.resize( m.size() );
    }


    vl::vec< Float, 3 >& point( std::size_t idx )
    {
        ZI_ASSERT( idx < size_ );
        return points_[ idx ];
    }

    const vl::vec< Float, 3 >& point( std::size_t idx ) const
    {
        ZI_ASSERT( idx < size_ );
        return points_[ idx ];
    }

    qmetric< Float, 6 >& quadratic( std::size_t idx )
    {
        ZI_ASSERT( idx < size_ );
        return quadratic_[ idx ];
    }

    const detail::quadratic< Float >& quadratic( std::size_t idx ) const
    {
        ZI_ASSERT( idx < size_ );
        return quadratic_[ idx ];
    }

    vl::vec< Float, 3 >& normal( std::size_t idx )
    {
        ZI_ASSERT( idx < size_ );
        return normals_[ idx ];
    }

    const vl::vec< Float, 3 >& normal( std::size_t idx ) const
    {
        ZI_ASSERT( idx < size_ );
        return normals_[ idx ];
    }

    void resize( std::size_t s )
    {
        size_ = s;
        heap_.clear();
        invalid_.clear();

        mesh_.resize( s );
        points_.resize( s );
        normals_.resize( s );
        quadratic_.resize( s );
    }

    void clear( std::size_t s = 0 )
    {
        if ( s != 0 )
        {
            size_ = s;
        }
        resize( size_ );
    }

    uint32_t add_face( const uint32_t x, const uint32_t y, const uint32_t z )
    {
        return mesh_.add_face( x, y, z );
    }

    inline void prepare()
    {
        mesh_.check_rep();
        generate_normals();
        generate_quadratic();
        init_heap();
    }

    inline std::size_t heap_size() const
    {
        return heap_.size();
    }

    inline std::size_t round()
    {
        iterate();
        return heap_.size();
    }

    inline std::size_t optimize( std::size_t target_faces, Float max_error )
    {

        std::size_t bad = 0;
        while ( mesh_.face_count() > target_faces )
        {
            if ( ( heap_.size() == 0 ) || ( heap_.top().value_ > max_error ) )
            {
                break;
            }
            if ( !iterate() )
            {
                ++bad;
            }
        }

        //generate_normals();

        invalid_.clear();
        std::cout << "BAD >>>>>>> " << bad << "\n\n";
        return mesh_.face_count();
    }

    inline std::size_t face_count() const
    {
        return mesh_.face_count();
    }

    inline std::size_t edge_count() const
    {
        return mesh_.edge_count();
    }

    inline std::size_t vertex_count() const
    {
        return size_;
    }

    inline Float min_error() const
    {
        if ( heap_.size() )
        {
            return heap_.top().value_;
        }

        return 0;
    }

    inline detail::tri_mesh_face_container& faces()
    {
        return mesh_.faces;
    }

private:

    inline bool check_valid( const uint64_t e, const vl::vec< Float, 3 >& p ) const
    {
        // todo: better inverion check
        //return ( check_topology( e, p ) && ( check_inversion( e, p ) < 0.1 ) );
        return false;
    }

    inline bool iterate()
    {
        ZI_ASSERT( heap_.size() );

        heap_entry e( heap_.top() );
        heap_.pop();

        const uint32_t v0 = detail::edge_source( e.edge_ );
        const uint32_t v1 = detail::edge_sink  ( e.edge_ );

        if ( !check_valid_edge( e.edge_ ) )
        {
            return false;
        }

        if ( !check_topology( e.edge_ ) )
        {
            return false;
        }

        if ( !check_inversion( e.edge_, e.optimal_ ) ) // todo: better
        {
            return false;
        }

        //if ( !check_compactness( e.edge_, e.optimal_ ) )
        //{
        //return false;
        //}

        // erase old ones
        for ( uint32_t v = mesh_.across_edge( v0, v1 );
              v != v1;
              v = mesh_.across_edge( v0, v ) )
        {
            uint64_t eind = ( v0 < v ) ?
                detail::make_edge( v0, v ) :
                detail::make_edge( v, v0 );
            heap_.erase_key( eind );
        }

        for ( uint32_t v = mesh_.across_edge( v1, v0 );
              v != v0;
              v = mesh_.across_edge( v1, v ) )
        {
            uint64_t eind = ( v1 < v ) ?
                detail::make_edge( v1, v ) :
                detail::make_edge( v, v1 );
            heap_.erase_key( eind );
        }

        uint32_t v = mesh_.collapse_edge( v0, v1 );
        points_[ v ]  = e.optimal_;
        normals_[ v ] = e.optimal_normal_;

        quadratic_[ v ] += ( v == v0 ) ? quadratic_[ v1 ] : quadratic_[ v0 ];

        ZI_ASSERT( mesh_.valid_vertex( v ) );

        uint32_t vlast = detail::edge_sink( mesh_.vertex_edge( v ) );

        uint32_t vind = vlast;
        do {
            if ( v < vind )
            {
                add_to_heap( v, vind );
            }
            else
            {
                add_to_heap( vind, v );
            }
            vind = mesh_.across_edge( v, vind );
        } while ( vind != vlast );

        return true;

    }

    void generate_quadratic()
    {
        FOR_EACH( it, quadratic_ )
        {
            it->clear();
        }

        FOR_EACH( it, mesh_.faces )
        {
            vl::vec< Float, 3 > &v0 = points_[ it->v0() ];
            vl::vec< Float, 3 > &v1 = points_[ it->v1() ];
            vl::vec< Float, 3 > &v2 = points_[ it->v2() ];
            //vl::vec< Float, 3 > &n0 = normals_[ it->v0() ];
            //vl::vec< Float, 3 > &n1 = normals_[ it->v1() ];
            //vl::vec< Float, 3 > &n2 = normals_[ it->v2() ];

            vl::vec< Float, 3 > a = cross( v1 - v0, v2 - v0 );
            Float area = normalize( a );

            //vl::vec< Float, 6 > p0( points_[ it->v0() ], a );
            //vl::vec< Float, 6 > p1( points_[ it->v1() ], a );
            //vl::vec< Float, 6 > p2( points_[ it->v2() ], a );

            //qmetric_t q( ( v0, v0 ), ( v1, v1 ), ( v2, v2 ) );

            //a *= ( static_cast< Float >( 1 ) / sqrlen( a ) );
            qmetric_t q( ( v0, a ), ( v1 , a ), ( v2 , a ) );

            q *= area;

            quadratic_[ it->v0() ] += q;
            quadratic_[ it->v1() ] += q;
            quadratic_[ it->v2() ] += q;
        }

        //FOR_EACH( it, d_.vd_ )
        //{
            //std::cout << it->quadratic_ << "\n";
        //}


    }

    void generate_normals()
    {
        std::vector< int > counts( size_ );
        std::fill_n( counts.begin(), size_, 0 );

        FOR_EACH( it, normals_ )
        {
            (*it) = vl::vec< Float, 3 >::zero;
        }

        FOR_EACH( it, mesh_.faces )
        {
            vl::vec< Float, 3 > &v0 = points_[ it->v0() ];
            vl::vec< Float, 3 > &v1 = points_[ it->v1() ];
            vl::vec< Float, 3 > &v2 = points_[ it->v2() ];

            vl::vec< Float, 3 > n( cross( v1 - v0, v2 - v0 ) );
            //n *= ( static_cast< Float >( 1 ) / len( n ) );
            n = norm( n );
            normals_[ it->v0() ] += n;
            normals_[ it->v1() ] += n;
            normals_[ it->v2() ] += n;

            ++counts[ it->v0() ];
            ++counts[ it->v1() ];
            ++counts[ it->v2() ];
        }

        for ( std::size_t i = 0; i < size_; ++i )
        {
            if ( counts[ i ] > 0 )
            {
                //normals_[ i ] /= static_cast< Float >( counts[ i ] );
                normalize( normals_[ i ] );
            }
        }
    }

    inline void add_to_heap( uint32_t v0, uint32_t v1 )
    {
        const uint64_t e = detail::make_edge( v0, v1 );

        ZI_ASSERT_0( heap_.key_count( e ) );

        if ( !check_valid_edge( e ) )
        {
            return;
        }

        const Float third = static_cast< Float >( 1 ) / 3;

        qmetric< Float, 6 > q( quadratic_[ v0 ] + quadratic_[ v1 ] );

        vl::vec< Float, 6 > pos( 0 );

        if ( !q.optimize( pos ) )
        {
            //if ( !q.optimize( pos, points_[ v0 ], points_[ v1 ] ) )
            //{
            //std::cout << "YEA\n";
            pos  = ( points_[ v0 ], normals_[ v0 ] );
            pos += ( points_[ v1 ], normals_[ v1 ] );
            pos *= 0.5;
            //}
        }

        //vl::vec< Float, 3 > va( pos[ 0 ], pos[ 1 ], pos[ 2 ] );
        //vl::vec< Float, 3 > vb( pos[ 3 ], pos[ 4 ], pos[ 5 ] );

        //std::cout << "ADDING TO HEAP: " << points_[ v0 ]
        //<< ", " << points_[ v1 ] << " ::: " << pos << "\n\n";

        //if ( check_inversion( e, pos ) < 0.01 ) // todo: better
        //{
        //return;
        //}

/*        std::ostringstream oss;

        oss << "SUM = " << q.evaluate( pos )
            << " ?= " << quadratic_[ v0 ].evaluate( pos ) + quadratic_[ v1 ].evaluate( pos )
            << " == " << quadratic_[ v0 ].evaluate( pos )
            << " + "  << quadratic_[ v1 ].evaluate( pos )
            << " ::: " << std::numeric_limits< double >::epsilon()
            << " ::: " << std::numeric_limits< double >::round_error()
            << "\n\n";

        std::cout << oss.str() << std::flush ;
*/

        //std::cout << pos << "\n";

        heap_.insert( heap_entry( e, q.evaluate( pos ),
                                  pos.template subvector< 3, 0 >(),
                                  pos.template subvector< 3, 3 >() ));
    }

    void init_heap()
    {
        FOR_EACH( it, mesh_.faces )
        {
            if ( it->v0() < it->v1() )
            {
                add_to_heap( it->v0(), it->v1() );
            }

            if ( it->v1() < it->v2() )
            {
                add_to_heap( it->v1(), it->v2() );
            }

            if ( it->v2() < it->v0() )
            {
                add_to_heap( it->v2(), it->v0() );
            }
        }
    }

};

} // namespace mesh
} // namespace zi

#endif

