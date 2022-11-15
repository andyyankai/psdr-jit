#include <misc/Exception.h>
#include <psdr/core/bitmap_loader.h>
#include <psdr/core/bitmap.h>
#include <psdr/core/transform.h>

NAMESPACE_BEGIN(psdr_jit)

template <int channels>
Bitmap<channels>::Bitmap()
    : m_resolution(1, 1), m_data(zeros<ValueD>()) {drjit::make_opaque(m_to_world_raw);}


template <int channels>
Bitmap<channels>::Bitmap(const char *file_name) {
    load_openexr(file_name);
    drjit::make_opaque(m_to_world_raw);
}


template <int channels>
Bitmap<channels>::Bitmap(ScalarValue value) : m_resolution(1, 1), m_data(value) {drjit::make_opaque(m_to_world_raw);}


template <int channels>
Bitmap<channels>::Bitmap(int width, int height, const ValueD &data) : m_resolution(width, height), m_data(data) {
    PSDR_ASSERT(width*height == static_cast<int>(data.size()));
    drjit::make_opaque(m_to_world_raw);
}


template <int channels>
void Bitmap<channels>::load_openexr(const char *file_name) {
    Vector4fC data;
    std::tie(data, m_resolution) = BitmapLoader::load_openexr_rgba(file_name);
    if constexpr ( channels == 1 ) {
        m_data = data[0];
    } else {
        m_data = Vector3fD(data[0], data[1], data[2]);
    }
}


template <int channels>
template <bool ad>
typename Bitmap<channels>::template Value<ad> Bitmap<channels>::eval(Vector2f<ad> uv, bool flip_v) const {
    const int width = m_resolution.x(), height = m_resolution.y();

    if ( static_cast<int>(slices<ValueD>(m_data)) != width*height )
        throw Exception("Bitmap: invalid data size!");

    if ( width == 1 && height == 1 ) {
        if constexpr ( ad )
            return m_data;
        else
            return detach(m_data);
    } else {
        if ( width < 2 || height < 2 )
            throw Exception("Bitmap: invalid resolution!");

        if ( flip_v ) {
            // flip the v coordinates to match common practices
            uv.y() = -uv.y();
        }        

        // BUG
        if constexpr ( ad ) {
            uv = transform2d_pos(m_to_world_raw, uv);
        } else {
            const Matrix3fC &to_worldC = detach(m_to_world_raw);
            uv = transform2d_pos(to_worldC, uv);
        }
        uv -= floor(uv);


        uv *= Vector2f<ad>(m_resolution - 1);


        // std::cout << uv+0.1f << std::endl;

        Vector2i<ad> pos = floor2int<Vector2i<ad>, Vector2f<ad>>(uv);
        Vector2f<ad> w1 = uv - Vector2f<ad>(pos), w0 = 1.0f - w1;
        pos = minimum(pos, m_resolution - 2);

        //Int<ad> idx = pos.y()*width + pos.x();
        Int<ad> idx = fmadd(pos.y(), width, pos.x());


        Value<ad> v00, v10, v01, v11;
        if constexpr ( ad ) {
            v00 = gather<ValueD>(m_data, idx);
            v10 = gather<ValueD>(m_data, idx + 1);
            v01 = gather<ValueD>(m_data, idx + width);
            v11 = gather<ValueD>(m_data, idx + width + 1);
        } else {
            const ValueC &data = detach(m_data);
            v00 = gather<ValueC>(data, idx);
            v10 = gather<ValueC>(data, idx + 1);
            v01 = gather<ValueC>(data, idx + width);
            v11 = gather<ValueC>(data, idx + width + 1);
        }

        // Bilinear interpolation
        Value<ad> v0 = fmadd(w0.x(), v00, w1.x()*v10),
                  v1 = fmadd(w0.x(), v01, w1.x()*v11);
        return fmadd(w0.y(), v0, w1.y()*v1);
    }
}


// Explicit instantiations
template struct Bitmap<1>;
template struct Bitmap<3>;

template FloatC Bitmap<1>::eval<false>(Vector2fC, bool) const;
template FloatD Bitmap<1>::eval<true>(Vector2fD, bool) const;

template Vector3fC Bitmap<3>::eval<false>(Vector2fC, bool) const;
template Vector3fD Bitmap<3>::eval<true>(Vector2fD, bool) const;

NAMESPACE_END(psdr_jit)
