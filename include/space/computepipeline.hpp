#pragma once
#include <space/EntityManager.hpp>
#include <space/SpatialGrid.hpp>
#include <space/World.hpp>
#include <array>
#include <deque>
#include <filesystem>
namespace space {
class ComputePipeline {
    struct Snapshot { GLuint buffer=0;GLsync fence=nullptr;std::uint64_t sequence=0; };
    EntityManager& entities_;
    GridConfig grid_;
    GLuint erase_=0;
    GLuint neighbors_=0,neighborCounts_=0;
    std::uint32_t neighborCapacity_=0;
    GLuint force_=0,integrate_=0,index_=0,bodies_=0,counts_=0,offsets_=0,sorted_=0;
    std::uint32_t sortedCapacity_=0,bodyCount_=0,particleGravity_=0;
    float sourceMass_=0;
    std::array<Snapshot,3> snapshots_{};
    std::uint64_t sequence_=0,consumed_=0;
    GLint maxGroups_=0;
    std::deque<GLsync> frames_;
    std::array<GLuint,4> timers_{};
    std::array<bool,4> timerPending_{};
    double lastGPUms_=-1;
    void dispatch(GLuint program);
    void release() noexcept;
public:
    ComputePipeline(EntityManager& entities,GridConfig grid,const std::filesystem::path& shaders);
    ~ComputePipeline();
    ComputePipeline(const ComputePipeline&)=delete;
    ComputePipeline& operator=(const ComputePipeline&)=delete;
    void beginFrame();
    bool pollStatistics(std::vector<std::uint32_t>& counts);
    void uploadBodies(std::span<const MassiveBody> bodies,unsigned particleGravity=0);
    void step(float dt,float G,float softening);
    void reindex();
    void capture();
    void resetStatistics();
    std::uint32_t clearChunk(std::uint32_t chunk);
    double gpuMilliseconds() const { return lastGPUms_; }
    // Radius-filtered GPU broad/narrow phase; result is a uint count per particle.
    GLuint countNeighbors(float radius);
    GLuint sortedBuffer() const { return sorted_; }
    GLuint offsetBuffer() const { return offsets_; }
    // Blocking diagnostic path used only by GPU validation.
    void readIndex(std::vector<std::uint32_t>& membership,std::vector<std::uint32_t>& offsets,std::vector<std::uint32_t>& sorted) const;
};
}
