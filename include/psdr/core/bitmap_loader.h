#pragma once

#include <psdr/psdr.h>

NAMESPACE_BEGIN(psdr_jit)


struct BitmapLoader {
    static std::pair<Vector4fC, ScalarVector2i> load_openexr_rgba(const char *file_name);
};


NAMESPACE_END(psdr_jit)
