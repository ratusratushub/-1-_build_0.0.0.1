#include <glad/gl.h>
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include <space/ComputePipeline.hpp>
#include <space/Physics.hpp>
#include <space/GLUtils.hpp>
#include <algorithm>
#include <array>
#include <iostream>
#include <numeric>
namespace {
void require(bool ok,const char* message){if(!ok) throw std::runtime_error(message);}
void compare(float actual,float expected) {
    if(!std::isfinite(actual)||std::abs(actual-expected)>2e-4f+std::abs(expected)*3e-5f) throw std::runtime_error("GPU/CPU state mismatch");
}
void validate(unsigned mode,std::uint32_t count) {
    space::SimulationConfig config;config.initialCount=count;config.particleGravity=mode;
    space::World world(config.world);space::EntityManager entities;entities.initialize(config);
    std::vector<space::Vec4> positions,velocities;
    entities.readState(positions,velocities);
    for(std::uint32_t i=0;i<count;++i) {
        auto expected=space::initialPosition(i,config);compare(positions[i].x,expected.x);compare(positions[i].w,expected.w);
    }
    if(count>=5) {
        positions[0]={-400.001f,-450,-450,1};velocities[0]={100,0,0,0};
        positions[1]={500,500,500,1};velocities[1]={0,0,0,0};
        positions[2]={501,0,0,1};velocities[2]={0,0,0,0};
        positions[3]={0,0,0,1};velocities[3]={0,0,0,0};
        positions[4]={-500,-500,-500,1};velocities[4]={-100,0,0,0};
        entities.setState(positions,velocities);
    }
    if(count>=7){positions[5]={-0.00001f,0,0,1};positions[6]={99.99999f,0,0,1};entities.setState(positions,velocities);}
    auto capacity=entities.capacity();entities.reserve(capacity+1);
    require(entities.capacity()>capacity,"Dynamic reserve failed");
    std::vector<space::Vec4> copiedP,copiedV;entities.readState(copiedP,copiedV);
    for(std::size_t i=0;i<positions.size();++i){compare(copiedP[i].x,positions[i].x);compare(copiedV[i].x,velocities[i].x);}
    space::ComputePipeline compute(entities,config.world,SPACE_SHADER_DIR);
    std::vector<space::MassiveBody> massive{{0,0,0,100000,0,0,0,12,true},{100,80,-50,1000,0,0,0,4,true}};
    const std::array<space::Vec4,2> sources{{{0,0,0,100000},{100,80,-50,1000}}};
    compute.uploadBodies(massive,mode);compute.reindex();
    const auto verifyIndex=[&] {
        std::vector<std::uint32_t> membership,offsets,sorted;
        compute.readIndex(membership,offsets,sorted);
        require(offsets.front()==0&&offsets.back()==count,"Prefix bounds");
        require(std::is_sorted(offsets.begin(),offsets.end()),"Nonmonotonic prefix");
        auto ids=sorted;std::sort(ids.begin(),ids.end());
        for(std::uint32_t i=0;i<count;++i) require(ids[i]==i,"Scatter uniqueness/coverage");
        for(std::uint32_t c=0;c<=config.world.count();++c) for(auto i=offsets[c];i<offsets[c+1];++i) require(membership[sorted[i]]==c,"Scatter membership");
        for(std::uint32_t i=0;i<count;++i) {
            auto cpu=world.locate(positions[i]);if(cpu==space::outsideChunk) cpu=std::uint32_t(config.world.count());
            require(membership[i]==cpu,"CPU/GPU chunk classification");
        }
    };
    verifyIndex();
    for(int step=0;step<12;++step) {
        compute.beginFrame();compute.step(0.001f,config.G,config.softening);
        space::referenceStep(positions,velocities,sources,0.001f,config.G,config.softening,mode!=0);
    }
    entities.readState(copiedP,copiedV);
    for(std::size_t i=0;i<positions.size();++i) {
        compare(copiedP[i].x,positions[i].x);compare(copiedP[i].y,positions[i].y);compare(copiedP[i].z,positions[i].z);compare(copiedP[i].w,positions[i].w);
        compare(copiedV[i].x,velocities[i].x);compare(copiedV[i].y,velocities[i].y);compare(copiedV[i].z,velocities[i].z);
    }
    std::vector<space::Vec4> acceleration(count);
    glBindBuffer(GL_COPY_READ_BUFFER,entities.accelerationBuffer());
    glGetBufferSubData(GL_COPY_READ_BUFFER,0,GLsizeiptr(acceleration.size()*sizeof(space::Vec4)),acceleration.data());
    for(std::uint32_t i=0;i<count;++i) {
        auto expected=space::referenceAcceleration(copiedP[i],sources,config.G,config.softening);
        if(mode) for(std::uint32_t j=0;j<count;++j) if(i!=j) {
            auto other=space::referenceAcceleration(copiedP[i],std::span<const space::Vec4>(&copiedP[j],1),config.G,config.softening);
            expected.x+=other.x;expected.y+=other.y;expected.z+=other.z;
        }
        compare(acceleration[i].x,expected.x);compare(acceleration[i].y,expected.y);compare(acceleration[i].z,expected.z);
    }
    // Classification compares the actual float positions to avoid misclassifying
    // tiny CPU/GPU rounding differences at a face as a spatial-index defect.
    positions=copiedP;verifyIndex();
    auto neighborBuffer=compute.countNeighbors(100.f);
    std::vector<std::uint32_t> neighbors(count);glBindBuffer(GL_COPY_READ_BUFFER,neighborBuffer);
    glGetBufferSubData(GL_COPY_READ_BUFFER,0,GLsizeiptr(neighbors.size()*sizeof(std::uint32_t)),neighbors.data());
    for(std::uint32_t i=0;i<count;++i) {
        unsigned expected=0;
        for(std::uint32_t j=0;j<count;++j) if(i!=j) {
            float x=positions[i].x-positions[j].x,y=positions[i].y-positions[j].y,z=positions[i].z-positions[j].z;
            if(x*x+y*y+z*z<=10000.f) ++expected;
        }
        require(neighbors[i]==expected,"GPU neighbor query disagrees with exact radius scan");
    }
    if(count>=5) require(world.coordinates(world.locate(positions[0])).x==1,"Particle failed to cross chunk face");
    compute.capture();glFinish();std::vector<std::uint32_t> counts;
    require(compute.pollStatistics(counts),"Missing statistics snapshot");
    require(std::accumulate(counts.begin(),counts.end(),std::uint64_t(0))==count,"Histogram count mismatch");
    if(count) {
        const auto target=world.locate(positions[0]);
        if(target!=space::outsideChunk) {
            unsigned expectedRemoved=0;for(auto p:positions) if(world.locate(p)==target) ++expectedRemoved;
            require(compute.clearChunk(target)==expectedRemoved,"Clear chunk removed wrong count");
            require(entities.count()==count-expectedRemoved,"Compaction live count mismatch");
            std::vector<std::uint32_t> ids(entities.count());
            glBindBuffer(GL_COPY_READ_BUFFER,entities.identityBuffer());glGetBufferSubData(GL_COPY_READ_BUFFER,0,GLsizeiptr(ids.size()*sizeof(std::uint32_t)),ids.data());
            std::vector<space::Vec4> compactP,compactV;entities.readState(compactP,compactV);
            auto unique=ids;std::sort(unique.begin(),unique.end());require(std::adjacent_find(unique.begin(),unique.end())==unique.end(),"Duplicate persistent identity");
            for(std::size_t i=0;i<ids.size();++i) {
                require(ids[i]<count&&world.locate(positions[ids[i]])!=target,"Identity survived its chunk deletion");
                compare(compactP[i].x,positions[ids[i]].x);compare(compactP[i].y,positions[ids[i]].y);compare(compactP[i].z,positions[ids[i]].z);
            }
            const auto oldCount=entities.count();entities.append(7,config,target);compute.reindex();
            entities.readState(compactP,compactV);ids.resize(entities.count());
            glBindBuffer(GL_COPY_READ_BUFFER,entities.identityBuffer());glGetBufferSubData(GL_COPY_READ_BUFFER,0,GLsizeiptr(ids.size()*sizeof(std::uint32_t)),ids.data());
            for(auto i=oldCount;i<entities.count();++i) {
                require(ids[i]>=count,"Persistent identity was reused");require(world.locate(compactP[i])==target,"Dynamic fill escaped target chunk");
            }
        }
    }
    entities.clear();compute.reindex();compute.resetStatistics();compute.capture();glFinish();
    require(compute.pollStatistics(counts),"Missing empty snapshot");
    require(std::accumulate(counts.begin(),counts.end(),std::uint64_t(0))==0,"Empty grid count mismatch");
    bool rejected=false;try{entities.reserve(entities.limit()+1);}catch(const std::length_error&){rejected=true;}require(rejected,"Overflow was accepted");
    rejected=false;try{compute.step(1,config.G,config.softening);}catch(const std::invalid_argument&){rejected=true;}require(rejected,"Unsafe timestep accepted");
    space::checkGL("GPU validation");
}
}
int main() {
    if(!glfwInit()){std::cerr<<"SKIP: no display/GLFW\n";return 77;}
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR,4);glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR,3);
    glfwWindowHint(GLFW_OPENGL_PROFILE,GLFW_OPENGL_CORE_PROFILE);glfwWindowHint(GLFW_VISIBLE,GLFW_FALSE);
    auto* window=glfwCreateWindow(64,64,"Validation",nullptr,nullptr);
    if(!window){glfwTerminate();std::cerr<<"SKIP: no OpenGL 4.3 context\n";return 77;}
    glfwMakeContextCurrent(window);
    if(!gladLoadGL(glfwGetProcAddress)){glfwDestroyWindow(window);glfwTerminate();std::cerr<<"GL loader failed\n";return 1;}
    int result=0;
    try{validate(0,0);validate(0,1);validate(0,257);validate(1,257);std::cout<<"CPU/GPU gravity, indexing and capacity checks passed\n";}
    catch(const std::exception& e){std::cerr<<e.what()<<'\n';result=1;}
    glfwDestroyWindow(window);glfwTerminate();return result;
}
