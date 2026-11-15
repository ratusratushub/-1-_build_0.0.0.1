#pragma once
#include <space/Types.hpp>
#include <space/ThreadPool.hpp>
#include <span>
namespace space {
class SpatialGrid {
    GridConfig config_;
    std::vector<GravityCell> gravity_;
public:
    explicit SpatialGrid(GridConfig config);
    // Snapshot entries are first moments (sum(m*x), sum(m*y), sum(m*z), sum(m)).
    void update(std::span<const Vec4> moments, ThreadPool& pool);
    const std::vector<GravityCell>& gravity() const { return gravity_; }
    const GridConfig& config() const { return config_; }
};
void advanceMassiveBodies(std::vector<MassiveBody>& bodies, double dt, double G, double softening);
}
