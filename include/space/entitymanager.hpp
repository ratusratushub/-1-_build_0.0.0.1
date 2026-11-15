#pragma once
#include <glad/gl.h>
#include <space/Config.hpp>
#include <span>
#include <optional>
#include <space/config.hpp>
namespace space {
class EntityManager {
    GLuint positions_=0,velocities_=0,accelerations_=0,membership_=0,identities_=0;
    std::uint32_t count_=0,capacity_=0,limit_=0;
    double massUpperBound_=0;
    std::uint64_t nextIdentity_=0;
    friend class ComputePipeline;
    // Internal compaction kernel uses a validated program and device dispatch limit.
    std::uint32_t eraseChunk(std::uint32_t chunk,GLuint program,GLint maxGroups);
public:
    explicit EntityManager(std::uint32_t count=0);
    ~EntityManager();
    EntityManager(const EntityManager&)=delete;
    EntityManager& operator=(const EntityManager&)=delete;
    void reserve(std::uint32_t required);
    void initialize(const SimulationConfig& config);
    void append(std::uint32_t count,const SimulationConfig& config,std::optional<std::uint32_t> chunk=std::nullopt);
    void clear() { count_=0;massUpperBound_=0;nextIdentity_=0; }
    // Explicit diagnostic/maintenance path; never used by rendering or frame loop.
    void setState(std::span<const Vec4> positions,std::span<const Vec4> velocities);
    void readState(std::vector<Vec4>& positions,std::vector<Vec4>& velocities) const;
    void bind() const;
    GLuint positionBuffer() const { return positions_; }
    GLuint velocityBuffer() const { return velocities_; }
    GLuint identityBuffer() const { return identities_; }
    GLuint accelerationBuffer() const { return accelerations_; }
    GLuint membershipBuffer() const { return membership_; }
    std::uint32_t count() const { return count_; }
    std::uint32_t capacity() const { return capacity_; }
    std::uint32_t limit() const { return limit_; }
    // Exact at initialization; conservative after deletions.
    double massUpperBound() const { return massUpperBound_; }
};
}
