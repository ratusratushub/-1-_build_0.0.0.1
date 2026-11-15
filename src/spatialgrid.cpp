#include <space/SpatialGrid.hpp>
#include <cmath>
#include <stdexcept>
namespace space {
SpatialGrid::SpatialGrid(GridConfig config) : config_(config) {
    if (!config.side || config.side > 128 || !std::isfinite(config.halfExtent) || config.halfExtent <= 0)
        throw std::invalid_argument("Invalid grid dimensions");
    gravity_.resize(config.count());
}
void SpatialGrid::update(std::span<const Vec4> moments, ThreadPool& pool) {
    if (moments.size() != gravity_.size()) throw std::invalid_argument("Snapshot size mismatch");
    double tx=0, ty=0, tz=0, tm=0;
    for (auto m : moments) {
        if(!std::isfinite(m.x)||!std::isfinite(m.y)||!std::isfinite(m.z)||!std::isfinite(m.w)||m.w<0) throw std::invalid_argument("Invalid chunk moments");
        tx+=m.x; ty+=m.y; tz+=m.z; tm+=m.w;
    }
    int side = int(config_.side);
    pool.parallelFor(moments.size(), 128, [&](std::size_t begin, std::size_t end) {
        for (auto i=begin; i<end; ++i) {
            const auto m = moments[i];
            gravity_[i].nearCOM = m.w > 0 ? Vec4{m.x/m.w,m.y/m.w,m.z/m.w,m.w} : Vec4{};
            int x=int(i)%side, y=(int(i)/side)%side, z=int(i)/(side*side);
            double nx=0, ny=0, nz=0, nm=0;
            for (int dz=-1; dz<=1; ++dz) for (int dy=-1; dy<=1; ++dy) for (int dx=-1; dx<=1; ++dx) {
                int a=x+dx,b=y+dy,c=z+dz;
                if (a<0 || b<0 || c<0 || a>=side || b>=side || c>=side) continue;
                auto n=moments[std::size_t((c*side+b)*side+a)];
                nx+=n.x; ny+=n.y; nz+=n.z; nm+=n.w;
            }
            double mass = tm-nm;
            gravity_[i].farCOM = mass > 0 ? Vec4{float((tx-nx)/mass),float((ty-ny)/mass),float((tz-nz)/mass),float(mass)} : Vec4{};
        }
    });
}
void advanceMassiveBodies(std::vector<MassiveBody>& bodies, double dt, double G, double softening) {
    if(!std::isfinite(dt)||dt<=0||!std::isfinite(G)||G<0||!std::isfinite(softening)||softening<=0) throw std::invalid_argument("Invalid massive-body step");
    for(auto b:bodies) if(!std::isfinite(b.x)||!std::isfinite(b.y)||!std::isfinite(b.z)||!std::isfinite(b.mass)||b.mass<=0||!std::isfinite(b.vx)||!std::isfinite(b.vy)||!std::isfinite(b.vz)) throw std::invalid_argument("Invalid massive body state");
    struct Accel { double x=0,y=0,z=0; };
    const auto accelerations=[&] {
        std::vector<Accel> result(bodies.size());
        for(std::size_t i=0;i<bodies.size();++i) for(std::size_t j=0;j<bodies.size();++j) if(i!=j) {
            double x=bodies[j].x-bodies[i].x,y=bodies[j].y-bodies[i].y,z=bodies[j].z-bodies[i].z;
            double r2=x*x+y*y+z*z+softening*softening;
            double factor=G*bodies[j].mass/(r2*std::sqrt(r2));
            result[i].x+=x*factor;result[i].y+=y*factor;result[i].z+=z*factor;
        }
        return result;
    };
    auto a=accelerations();
    for(std::size_t i=0;i<bodies.size();++i) if(!bodies[i].fixed) {
        auto& b=bodies[i];b.vx+=0.5*dt*a[i].x;b.vy+=0.5*dt*a[i].y;b.vz+=0.5*dt*a[i].z;
        b.x+=dt*b.vx;b.y+=dt*b.vy;b.z+=dt*b.vz;
    }
    a=accelerations();
    for(std::size_t i=0;i<bodies.size();++i) if(!bodies[i].fixed) {
        auto& b=bodies[i];b.vx+=0.5*dt*a[i].x;b.vy+=0.5*dt*a[i].y;b.vz+=0.5*dt*a[i].z;
    }
}
}
