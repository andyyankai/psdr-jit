#pragma once

#include <psdr/psdr.h>
#include <misc/Exception.h>


namespace psdr
{

template <int channels>
struct Bitmap {
    static_assert(channels == 1 || channels == 3);
    using ScalarValue = typename std::conditional<channels == 1, float, const ScalarVector3f&>::type;

    template <bool ad>
    using Value = typename std::conditional<channels == 1, Float<ad>, Vector3f<ad>>::type;

    using ValueC = Value<false>;
    using ValueD = Value<true>;

    Bitmap();
    Bitmap(const char *file_name);
    Bitmap(ScalarValue value);
    Bitmap(int width, int height, const ValueD &data);

    void load_openexr(const char *file_name);

    inline void set_transform(const Matrix3fD &mat, bool set_left = true) {
        if ( set_left ) {
            m_to_world_left = mat;
        } else {
            m_to_world_right = mat;
        }
    }

    inline void append_transform(const Matrix3fD &mat, bool append_left = true) {
        if ( append_left ) {
            m_to_world_left = mat*m_to_world_left;
        } else {
            m_to_world_right *= mat;
        }
    }

    inline void fill(ScalarValue value) {
        m_resolution = ScalarVector2i(1, 1);
        m_data = ValueD(value);
    }

    template <bool ad> Value<ad> eval(Vector2f<ad> uv, bool flip_v = true) const;

    ScalarVector2i m_resolution;
    ValueD m_data;

    Matrix3fD               m_to_world_raw   = identity<Matrix3fD>(),
                            m_to_world_left  = identity<Matrix3fD>(),
                            m_to_world_right = identity<Matrix3fD>();
};

using Bitmap1fD = Bitmap<1>;
using Bitmap3fD = Bitmap<3>;

} // namespace psdr
