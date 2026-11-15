#include <space/SpatialGrid.hpp>
#include <space/World.hpp>
#include <space/Physics.hpp>
#include <array>
#include <atomic>
#include <iostream>
#include <limits>
#include <string>
namespace {
void require(bool condition,const char* message) {if(!condition) throw std::runtime_error(message);}
template<class F> void rejects(F operation) {bool caught=false;try{operation();}catch(const std::exception&){caught=true;}require(caught,"Expected rejection");}
void poolTests() {
    space::ThreadPool pool(4);std::atomic<unsigned> visited=0;
    pool.parallelFor(10003,128,[&](auto a,auto b){visited.fetch_add(unsigned(b-a));});require(visited==10003,"Task coverage");
    rejects([&]{pool.parallelFor(500,128,[](auto,auto){throw std::runtime_error("task");});});
    rejects([&]{pool.parallelFor(1,0,[](auto,auto){});});
    space::SpatialGrid grid({4,10});std::vector<space::Vec4> moments(64);
    moments[0]={2,4,6,2};moments[63]={30,0,0,3};grid.update(moments,pool);
    require(grid.gravity()[0].nearCOM.x==1&&grid.gravity()[0].farCOM.w==3&&grid.gravity()[0].farCOM.x==10,"Legacy COM/exclusion");
    moments.assign(64,{});grid.update(moments,pool);
    for(auto cell:grid.gravity()) require(cell.nearCOM.w==0&&cell.farCOM.w==0,"Empty COM");
    moments[0].w=-1;rejects([&]{grid.update(moments,pool);});
}
void worldTests() {
    space::World w;
    require(w.config().count()==1000&&w.chunkSize()==100,"World configuration");
    require(w.locate({-500,-500,-500,0})==0,"Negative corner");
    require(w.locate({500,500,500,0})==999,"Positive corner inclusive");
    require(w.locate({0,0,0,0})==555,"Origin index");
    require(w.locate({-0.00001f,0,0,0})==554,"Cancellation at negative origin face");
    require(w.locate({99.99999f,0,0,0})==555,"Cancellation below positive face");
    require(w.locate({-400.01f,-499,-499,0})==0,"Negative coordinate floor");
    require(w.locate({-400,-499,-499,0})==1,"Internal face");
    require(w.locate({500.01f,0,0,0})==space::outsideChunk,"Outside positive");
    require(w.locate({-500.01f,0,0,0})==space::outsideChunk,"Outside negative");
    require(w.locate({std::numeric_limits<float>::quiet_NaN(),0,0,0})==space::outsideChunk,"NaN classification");
    for(unsigned i=0;i<1000;++i) require(w.index(w.coordinates(i))==i,"Chunk index round trip");
    auto chunk=w.chunk(555);require(chunk.minimum.x==0&&chunk.maximum.z==100,"Chunk bounds");
    space::ChunkManager manager;std::vector<std::uint32_t> counts(1001);
    counts[0]=1;manager.update(counts);require(manager.active().size()==1&&manager.active().at(0).count==1,"Sparse fill");
    counts[0]=0;counts[1]=1;manager.update(counts);require(!manager.active().contains(0)&&manager.active().contains(1),"Chunk migration");
    counts[1]=0;counts[1000]=1;manager.update(counts);require(manager.active().empty()&&manager.outsideCount()==1,"Exterior bucket");
    manager.clear();require(manager.outsideCount()==0&&manager.active().empty(),"Chunk clear");
    require(w.overlapping({50,50,50,0},1).size()==1,"Small interaction radius");
    require(w.overlapping({50,50,50,0},101).size()>1,"Independent interaction radius");
    require(w.overlapping({500,500,500,0},0).size()==1,"Inclusive query corner");
    require(w.overlapping({5000,0,0,0},10).empty(),"Exterior query");
}
void generationTests() {
    space::SimulationConfig c;c.validate();std::array<unsigned,8> octants{};double x=0,y=0,z=0;
    for(unsigned id=0;id<100000;++id) {
        auto p=space::initialPosition(id,c),v=space::initialVelocity(id,c);
        require(p.x>=-500&&p.x<=500&&p.y>=-500&&p.y<=500&&p.z>=-500&&p.z<=500,"Generation bounds");
        require(p.w>=c.minMass&&p.w<=c.maxMass&&std::isfinite(v.z),"Generation attributes");
        ++octants[(p.x>=0?1:0)+(p.y>=0?2:0)+(p.z>=0?4:0)];x+=p.x;y+=p.y;z+=p.z;
        require(space::initialPosition(id,c).x==p.x,"Reproducibility");
    }
    for(auto n:octants) require(n>11500&&n<13500,"Volumetric distribution");
    require(std::abs(x/100000)<5&&std::abs(y/100000)<5&&std::abs(z/100000)<5,"Uniform mean");
    auto old=space::initialPosition(0,c);++c.seed;require(space::initialPosition(0,c).x!=old.x,"Seed variation");
    require(space::grownCapacity(16,17,100)==24,"Geometric growth");
    require(space::grownCapacity(90,95,100)==100,"Capacity cap");
    rejects([]{space::grownCapacity(100,101,100);});
    c.softening=0;rejects([&]{c.validate();});
}
void physicsTests() {
    const std::array<space::Vec4,1> body{{{0,0,0,100000}}};
    auto a=space::referenceAcceleration({10,0,0,1},body,0.1f,5);
    require(a.x<0&&a.y==0&&a.z==0,"Gravity direction");
    auto center=space::referenceAcceleration({0,0,0,1},body,0.1f,5);
    require(center.x==0&&std::isfinite(center.y),"Softened origin");
    rejects([&]{space::referenceAcceleration({},body,0.1f,0);});
    std::vector<space::Vec4> p{{50,0,0,1}},v{{0,float(std::sqrt(10000.*2500/std::pow(2525.,1.5))),0,0}};
    const auto energy=[&]{double r2=double(p[0].x)*p[0].x+double(p[0].y)*p[0].y+double(p[0].z)*p[0].z;return 0.5*(double(v[0].x)*v[0].x+double(v[0].y)*v[0].y+double(v[0].z)*v[0].z)-10000/std::sqrt(r2+25);};
    const double initial=energy();
    for(int step=0;step<6000;++step) space::referenceStep(p,v,body,0.002f,0.1f,5);
    require(std::isfinite(p[0].x)&&std::isfinite(v[0].y),"Finite integration");
    require(std::abs((energy()-initial)/initial)<0.001,"Orbital energy drift");
    std::vector<space::MassiveBody> pair{{-1,0,0,1,0,0,0},{1,0,0,1,0,0,0}};
    space::advanceMassiveBodies(pair,0.01,1,0.1);
    require(pair[0].vx>0&&pair[1].vx<0&&std::abs(pair[0].vx+pair[1].vx)<1e-12,"Massive-body symmetry");
    pair[0].fixed=true;auto x=pair[0].x;space::advanceMassiveBodies(pair,0.01,1,0.1);require(pair[0].x==x,"Pinned body");
    space::SimulationController clock;unsigned steps=0;
    clock.paused=true;clock.advance(1,0.001f,false,[&](float){++steps;});require(steps==0,"Pause");
    clock.advance(1,0.001f,true,[&](float){++steps;});require(steps==1,"Single step");
    clock.paused=false;clock.timeScale=8;clock.advance(10,0.001f,false,[&](float){++steps;});require(steps==33&&clock.droppedTime>0,"Bounded catch-up");
    clock.reset();require(clock.time==0&&clock.droppedTime==0,"Restart clock");
}
}
int main(){try{poolTests();worldTests();generationTests();physicsTests();std::cout<<"CPU foundation invariants passed\n";return 0;}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
