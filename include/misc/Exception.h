#pragma once

#include <stdexcept>
#include <sstream>
#include <psdr/macros.h>

#define PSDR_ASSERT( cond )                                                    \
    do                                                                         \
    {                                                                          \
        if( !(cond) )                                                          \
        {                                                                      \
            std::stringstream ss;                                              \
            ss << __FILE__ << " (" << __LINE__ << "): " << #cond;              \
            throw psdr_jit::Exception( ss.str().c_str() );                         \
        }                                                                      \
    } while( 0 )


#define PSDR_ASSERT_MSG( cond, msg )                                           \
    do                                                                         \
    {                                                                          \
        if( !(cond) )                                                          \
        {                                                                      \
            std::stringstream ss;                                              \
            ss << "\n  File \"" << __FILE__ << "\", line " << __LINE__;        \
            throw psdr_jit::Exception( ( std::string(msg) + ss.str() ).c_str() );  \
        }                                                                      \
    } while( 0 )



NAMESPACE_BEGIN(psdr_jit)

class Exception : public std::runtime_error
{
 public:
     Exception( const char* msg )
         : std::runtime_error( msg )
     { }
};

NAMESPACE_END(psdr_jit)
