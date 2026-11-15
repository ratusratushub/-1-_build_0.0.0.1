#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>
namespace space {
// These layouts are shared verbatim with GLSL std430 (one 16-byte vector each).
struct alignas(16) Vec4 { float x{}, y{}, z{}, w{}; };
static_assert(sizeof(Vec4) == 16);
struct GridConfig {
    std::uint32_t side = 10;
    float halfExtent = 500.f;
    std::size_t count() const { return std::size_t(side) * side * side; }
};
struct alignas(16) GravityCell {
    Vec4 nearCOM; // xyz center of mass, w mass
    Vec4 farCOM;  // all cells outside the 3x3x3 neighborhood
};
static_assert(sizeof(GravityCell) == 32);
struct MassiveBody { // small CPU-owned top-level system, separate from particles
    double x{}, y{}, z{}, mass{}, vx{}, vy{}, vz{};
    float radius=12.f;
    bool fixed=false;
};
}
