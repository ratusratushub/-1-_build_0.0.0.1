#pragma once
#include <space/Types.hpp>
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
namespace space {
struct SimulationConfig {
    GridConfig world{};
    std::uint32_t initialCount=1000, seed=42;
    float minMass=0.5f,maxMass=2.f,minVelocity=-1.f,maxVelocity=1.f;
    float G=0.1f,softening=5.f,fixedStep=1.f/240.f;
    float blackHoleMass=100000.f,blackHoleRadius=12.f;
    // 0: external bodies only; 1: exact particle gravity (validation/small N only).
    unsigned particleGravity=0;
    void validate() const {
        if(!world.side || world.side>128 || !std::isfinite(world.halfExtent) || world.halfExtent<=0 || world.halfExtent>1e9f ||
           initialCount>10000000 || !std::isfinite(minMass) || !std::isfinite(maxMass) ||
           minMass<=0 || maxMass<minMass || maxMass>1e12f || !std::isfinite(minVelocity) || !std::isfinite(maxVelocity) ||
           maxVelocity<minVelocity || std::abs(minVelocity)>1e6f || std::abs(maxVelocity)>1e6f || !std::isfinite(G) || G<=0 || G>1e4f || !std::isfinite(softening) || softening<0.001f || softening>1e6f ||
           !std::isfinite(fixedStep) || fixedStep<=0 || fixedStep>1.f/60.f ||
           !std::isfinite(blackHoleMass) || blackHoleMass<=0 || blackHoleMass>1e12f || !std::isfinite(blackHoleRadius) || blackHoleRadius<=0 ||
           particleGravity>1 || (particleGravity && initialCount>4096))
            throw std::invalid_argument("Invalid simulation configuration (exact gravity is limited to 4096 particles)");
    }
};
// Stateless generation gives reproducible append batches without a CPU particle mirror.
inline float randomUnit(std::uint32_t value) {
    value^=value>>16; value*=0x7feb352du; value^=value>>15;
    value*=0x846ca68bu; value^=value>>16;
    return float(value>>8)*(1.f/16777216.f);
}
inline Vec4 initialPosition(std::uint32_t id,const SimulationConfig& c) {
    const auto key=id*8u+c.seed*0x9e3779b9u;
    const float h=c.world.halfExtent;
    return {(2*randomUnit(key)-1)*h,(2*randomUnit(key+1)-1)*h,(2*randomUnit(key+2)-1)*h,
            c.minMass+(c.maxMass-c.minMass)*randomUnit(key+3)};
}
inline Vec4 initialVelocity(std::uint32_t id,const SimulationConfig& c) {
    const auto key=id*8u+c.seed*0x9e3779b9u;
    auto v=[&](unsigned offset){return c.minVelocity+(c.maxVelocity-c.minVelocity)*randomUnit(key+offset);};
    return {v(4),v(5),v(6),0};
}
// Geometric capacity growth, bounded by both device and application limits.
inline std::uint32_t grownCapacity(std::uint32_t current,std::uint32_t required,std::uint32_t limit) {
    if(required>limit) throw std::length_error("Requested particle capacity exceeds GPU/application limit");
    if(required<=current) return current;
    return std::uint32_t(std::min<std::uint64_t>(limit,std::max<std::uint64_t>(required,std::max<std::uint64_t>(16,std::uint64_t(current)*3/2))));
}
}
