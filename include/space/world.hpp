#pragma once
#include <space/Types.hpp>
#include <algorithm>
#include <cmath>
#include <limits>
#include <span>
#include <stdexcept>
#include <unordered_map>
namespace space {
inline constexpr std::uint32_t outsideChunk=std::numeric_limits<std::uint32_t>::max();
struct ChunkCoord { int x{},y{},z{}; bool operator==(const ChunkCoord&) const = default; };
struct Chunk {
    ChunkCoord coord;
    Vec4 minimum,maximum;
    std::uint32_t snapshotOffset=0,count=0; // historical statistics; current ranges use GPU offsets[id]
    bool active() const { return count!=0; }
    void clear() { snapshotOffset=count=0; }
};
class World {
    GridConfig config_;
    int coordinate(float value) const {
        const float h=config_.halfExtent,size=chunkSize();
        int cell=std::clamp(int(std::floor((value+h)/size)),0,int(config_.side)-1);
        while(cell>0 && value<std::fma(float(cell),size,-h)) --cell;
        while(cell<int(config_.side)-1 && value>=std::fma(float(cell+1),size,-h)) ++cell;
        return cell;
    }
public:
    explicit World(GridConfig config={}) : config_(config) {
        if(!config.side || config.side>128 || !std::isfinite(config.halfExtent) || config.halfExtent<=0)
            throw std::invalid_argument("Invalid world");
    }
    const GridConfig& config() const { return config_; }
    float chunkSize() const { return 2*config_.halfExtent/float(config_.side); }
    std::uint32_t index(ChunkCoord c) const {
        if(c.x<0 || c.y<0 || c.z<0 || c.x>=int(config_.side) || c.y>=int(config_.side) || c.z>=int(config_.side)) return outsideChunk;
        return std::uint32_t((c.z*int(config_.side)+c.y)*int(config_.side)+c.x);
    }
    ChunkCoord coordinates(std::uint32_t id) const {
        if(id>=config_.count()) throw std::out_of_range("Invalid chunk id");
        const auto n=config_.side;
        return {int(id%n),int((id/n)%n),int(id/(n*n))};
    }
    std::uint32_t locate(Vec4 p) const {
        const float h=config_.halfExtent;
        if(!std::isfinite(p.x)||!std::isfinite(p.y)||!std::isfinite(p.z)||p.x < -h || p.y < -h || p.z < -h || p.x>h || p.y>h || p.z>h) return outsideChunk;
        // +halfExtent belongs to the final cell; exterior positions are NOT clamped.
        return index({coordinate(p.x),coordinate(p.y),coordinate(p.z)});
    }
    Chunk chunk(std::uint32_t id) const {
        const auto c=coordinates(id); const float size=chunkSize(),h=config_.halfExtent;
        Vec4 lo{std::fma(float(c.x),size,-h),std::fma(float(c.y),size,-h),std::fma(float(c.z),size,-h),0};
        Vec4 hi{std::fma(float(c.x+1),size,-h),std::fma(float(c.y+1),size,-h),std::fma(float(c.z+1),size,-h),0};
        return {c,lo,hi,0,0};
    }
    // Broad phase: interaction radius is independent of chunk dimensions.
    std::vector<std::uint32_t> overlapping(Vec4 p,float radius) const {
        if(!std::isfinite(radius) || radius<0 || !std::isfinite(p.x)||!std::isfinite(p.y)||!std::isfinite(p.z)) throw std::invalid_argument("Invalid neighbor query");
        const double h=config_.halfExtent;
        if(double(p.x)-radius>h||double(p.y)-radius>h||double(p.z)-radius>h||double(p.x)+radius < -h||double(p.y)+radius < -h||double(p.z)+radius < -h) return {};
        const auto low=[&](float v){return coordinate(float(std::clamp(double(v)-radius,-h,h)));};
        const auto high=[&](float v){return coordinate(float(std::clamp(double(v)+radius,-h,h)));};
        std::vector<std::uint32_t> result;
        for(int z=low(p.z);z<=high(p.z);++z) for(int y=low(p.y);y<=high(p.y);++y) for(int x=low(p.x);x<=high(p.x);++x) result.push_back(index({x,y,z}));
        return result;
    }
};
// Sparse CPU directory: only occupied chunks have live records. Particle lists
// reside in compact GPU ranges, not per-coordinate C++ objects or CPU mirrors.
class ChunkManager {
    World world_;
    std::unordered_map<std::uint32_t,Chunk> active_;
    std::uint32_t outside_=0;
public:
    explicit ChunkManager(GridConfig config={}) : world_(config) {}
    void clear() { active_.clear(); outside_=0; }
    void update(std::span<const std::uint32_t> counts) {
        if(counts.size()!=world_.config().count()+1) throw std::invalid_argument("Wrong chunk histogram size");
        clear(); std::uint32_t offset=0;
        for(std::uint32_t id=0;id<world_.config().count();++id) {
            if(counts[id]) { auto c=world_.chunk(id);c.snapshotOffset=offset;c.count=counts[id];active_.emplace(id,c); }
            if(counts[id]>std::numeric_limits<std::uint32_t>::max()-offset) throw std::overflow_error("Chunk counts overflow");
            offset+=counts[id];
        }
        outside_=counts.back();
    }
    const auto& active() const { return active_; }
    std::uint32_t outsideCount() const { return outside_; }
    const World& world() const { return world_; }
};
}
