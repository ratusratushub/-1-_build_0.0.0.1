#pragma once
#include <space/Config.hpp>
#include <span>
namespace space {
inline Vec4 referenceAcceleration(Vec4 p,std::span<const Vec4> sources,float G,float softening) {
    if(!std::isfinite(G)||G<0||!std::isfinite(softening)||softening<=0) throw std::invalid_argument("Invalid gravity constants");
    double ax=0,ay=0,az=0;
    for(auto b:sources) {
        double x=double(b.x)-p.x,y=double(b.y)-p.y,z=double(b.z)-p.z;
        double r2=x*x+y*y+z*z+double(softening)*softening;
        double factor=G*double(b.w)/(r2*std::sqrt(r2));
        ax+=x*factor;ay+=y*factor;az+=z*factor;
    }
    return {float(ax),float(ay),float(az),0};
}
inline void referenceStep(std::vector<Vec4>& p,std::vector<Vec4>& v,std::span<const Vec4> bodies,float dt,float G,float eps,bool selfGravity=false) {
    if(p.size()!=v.size()||!std::isfinite(dt)||dt<=0) throw std::invalid_argument("Invalid reference step");
    auto accelerations=[&] {
        std::vector<Vec4> a(p.size());
        for(std::size_t i=0;i<p.size();++i) {
            a[i]=referenceAcceleration(p[i],bodies,G,eps);
            if(selfGravity) for(std::size_t j=0;j<p.size();++j) if(i!=j) {
                auto f=referenceAcceleration(p[i],std::span<const Vec4>(&p[j],1),G,eps);
                a[i].x+=f.x;a[i].y+=f.y;a[i].z+=f.z;
            }
        }
        return a;
    };
    auto a=accelerations();
    for(std::size_t i=0;i<p.size();++i) {
        v[i].x+=0.5f*dt*a[i].x;v[i].y+=0.5f*dt*a[i].y;v[i].z+=0.5f*dt*a[i].z;
        p[i].x+=dt*v[i].x;p[i].y+=dt*v[i].y;p[i].z+=dt*v[i].z;
    }
    a=accelerations();
    for(std::size_t i=0;i<p.size();++i) {
        v[i].x+=0.5f*dt*a[i].x;v[i].y+=0.5f*dt*a[i].y;v[i].z+=0.5f*dt*a[i].z;
    }
}
inline float safeStep(float mass,float G,float eps) {
    // Conservative resolution of the fastest softened central frequency.
    return std::min(1.f/120.f,0.05f/std::sqrt(std::max(G*mass/(eps*eps*eps),1e-12f)));
}
class SimulationController {
    double accumulator_=0;
public:
    bool paused=false;
    float timeScale=1;
    double time=0,droppedTime=0;
    void reset() { accumulator_=0;time=droppedTime=0; }
    template<class Step> void advance(double elapsed,float dt,bool singleStep,Step step) {
        if(!std::isfinite(elapsed)||elapsed<0||!std::isfinite(dt)||dt<=0||!std::isfinite(timeScale)||timeScale<0) throw std::invalid_argument("Invalid clock parameters");
        if(paused) { accumulator_=0; if(singleStep){step(dt);time+=dt;} return; }
        accumulator_+=std::min(elapsed,0.25)*timeScale;
        unsigned steps=0;
        while(accumulator_>=dt && steps<32) { step(dt);accumulator_-=dt;time+=dt;++steps; }
        if(accumulator_>=dt) { auto keep=std::fmod(accumulator_,dt);droppedTime+=accumulator_-keep;accumulator_=keep; }
    }
};
}
