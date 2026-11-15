#include <space/EntityManager.hpp>
#include <space/World.hpp>
#include <space/GLUtils.hpp>
#include <array>
#include <algorithm>
#include <stdexcept>
namespace space {
EntityManager::EntityManager(std::uint32_t count) {
    GLint64 bytes=0;glGetInteger64v(GL_MAX_SHADER_STORAGE_BLOCK_SIZE,&bytes);
    limit_=std::uint32_t(std::min<GLint64>(10000000,bytes/sizeof(Vec4)));
    if(!limit_) throw std::runtime_error("No usable SSBO capacity");
    try {
        reserve(std::max(1u,count));
        if(count) { SimulationConfig config;config.initialCount=count;initialize(config); }
    } catch(...) {
        for(auto buffer:{positions_,velocities_,accelerations_,membership_,identities_}) glDeleteBuffers(1,&buffer);
        throw;
    }
}
EntityManager::~EntityManager() {
    for(auto b:{positions_,velocities_,accelerations_,membership_,identities_}) glDeleteBuffers(1,&b);
}
void EntityManager::reserve(std::uint32_t required) {
    auto next=grownCapacity(capacity_,required,limit_);
    if(next==capacity_) return;
    std::array<GLuint,5> fresh{};
    const std::array<GLuint,5> previous{positions_,velocities_,accelerations_,membership_,identities_};
    const std::array<std::size_t,5> stride{sizeof(Vec4),sizeof(Vec4),sizeof(Vec4),sizeof(std::uint32_t),sizeof(std::uint32_t)};
    glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);
    glGenBuffers(5,fresh.data());
    try {
        for(std::size_t i=0;i<fresh.size();++i) {
            glBindBuffer(GL_COPY_WRITE_BUFFER,fresh[i]);
            glBufferData(GL_COPY_WRITE_BUFFER,GLsizeiptr(next)*stride[i],nullptr,GL_DYNAMIC_COPY);
            if(glGetError()!=GL_NO_ERROR) throw std::runtime_error("GPU allocation failed; previous buffers preserved");
            if(previous[i] && count_) {
                glBindBuffer(GL_COPY_READ_BUFFER,previous[i]);
                glCopyBufferSubData(GL_COPY_READ_BUFFER,GL_COPY_WRITE_BUFFER,0,0,GLsizeiptr(count_)*stride[i]);
                if(glGetError()!=GL_NO_ERROR) throw std::runtime_error("GPU resize copy failed");
            }
        }
    } catch(...) {glDeleteBuffers(5,fresh.data());throw;}
    glDeleteBuffers(5,previous.data());
    positions_=fresh[0];velocities_=fresh[1];accelerations_=fresh[2];membership_=fresh[3];identities_=fresh[4];capacity_=next;
}
void EntityManager::initialize(const SimulationConfig& config) { config.validate();clear();append(config.initialCount,config); }
void EntityManager::append(std::uint32_t count,const SimulationConfig& config,std::optional<std::uint32_t> chunk) {
    config.validate();
    if(count>limit_-count_) throw std::length_error("Particle append exceeds device limit");
    if(nextIdentity_+count>std::uint64_t(std::numeric_limits<std::uint32_t>::max())+1) throw std::overflow_error("Particle ID space exhausted; start a new simulation");
    World world(config.world);std::optional<Chunk> region;
    if(chunk) region=world.chunk(*chunk);
    reserve(count_+count);
    double appendedMass=0;
    const auto batch=std::min(65536u,count);
    std::vector<Vec4> p(batch),v(batch);std::vector<std::uint32_t> ids(batch);
    for(std::uint32_t start=0;start<count;) {
        const auto n=std::min(batch,count-start);
        for(std::uint32_t j=0;j<n;++j) {
            const auto id=std::uint32_t(nextIdentity_+start+j);ids[j]=id;
            p[j]=initialPosition(id,config);v[j]=initialVelocity(id,config);appendedMass+=p[j].w;
            if(region) {
                auto remap=[&](float value,float lo,float hi){return std::min(std::nextafter(hi,lo),lo+(value+config.world.halfExtent)/(2*config.world.halfExtent)*(hi-lo));};
                p[j].x=remap(p[j].x,region->minimum.x,region->maximum.x);
                p[j].y=remap(p[j].y,region->minimum.y,region->maximum.y);
                p[j].z=remap(p[j].z,region->minimum.z,region->maximum.z);
            }
        }
        glBindBuffer(GL_SHADER_STORAGE_BUFFER,positions_);
        glBufferSubData(GL_SHADER_STORAGE_BUFFER,GLintptr(count_+start)*sizeof(Vec4),GLsizeiptr(n)*sizeof(Vec4),p.data());
        glBindBuffer(GL_SHADER_STORAGE_BUFFER,velocities_);
        glBufferSubData(GL_SHADER_STORAGE_BUFFER,GLintptr(count_+start)*sizeof(Vec4),GLsizeiptr(n)*sizeof(Vec4),v.data());
        glBindBuffer(GL_SHADER_STORAGE_BUFFER,identities_);
        glBufferSubData(GL_SHADER_STORAGE_BUFFER,GLintptr(count_+start)*sizeof(std::uint32_t),GLsizeiptr(n)*sizeof(std::uint32_t),ids.data());
        start+=n;
    }
    if(glGetError()!=GL_NO_ERROR) throw std::runtime_error("Particle initialization upload failed");
    count_+=count;massUpperBound_+=appendedMass;nextIdentity_+=count;
}
void EntityManager::setState(std::span<const Vec4> p,std::span<const Vec4> v) {
    if(p.size()!=v.size()||p.size()>limit_) throw std::invalid_argument("Invalid state size");
    for(std::size_t i=0;i<p.size();++i)
        if(!std::isfinite(p[i].x)||!std::isfinite(p[i].y)||!std::isfinite(p[i].z)||!std::isfinite(p[i].w)||p[i].w<=0||p[i].w>1e20f||std::abs(p[i].x)>1e12f||std::abs(p[i].y)>1e12f||std::abs(p[i].z)>1e12f||!std::isfinite(v[i].x)||!std::isfinite(v[i].y)||!std::isfinite(v[i].z)||std::abs(v[i].x)>1e12f||std::abs(v[i].y)>1e12f||std::abs(v[i].z)>1e12f) throw std::invalid_argument("Nonfinite state or nonpositive mass");
    reserve(std::uint32_t(p.size()));
    glBindBuffer(GL_SHADER_STORAGE_BUFFER,positions_);glBufferSubData(GL_SHADER_STORAGE_BUFFER,0,GLsizeiptr(p.size_bytes()),p.data());
    glBindBuffer(GL_SHADER_STORAGE_BUFFER,velocities_);glBufferSubData(GL_SHADER_STORAGE_BUFFER,0,GLsizeiptr(v.size_bytes()),v.data());
    if(glGetError()!=GL_NO_ERROR) throw std::runtime_error("State upload failed");
    glBindBuffer(GL_SHADER_STORAGE_BUFFER,identities_);
    std::vector<std::uint32_t> ids(std::min<std::size_t>(p.size(),65536));
    for(std::size_t start=0;start<p.size();start+=ids.size()) {
        auto n=std::min(ids.size(),p.size()-start);
        for(std::size_t i=0;i<n;++i) ids[i]=std::uint32_t(start+i);
        glBufferSubData(GL_SHADER_STORAGE_BUFFER,GLintptr(start*sizeof(std::uint32_t)),GLsizeiptr(n*sizeof(std::uint32_t)),ids.data());
    }
    checkGL("Upload particle identities");
    count_=std::uint32_t(p.size());nextIdentity_=count_;massUpperBound_=0;for(auto source:p) massUpperBound_+=source.w;
}
void EntityManager::readState(std::vector<Vec4>& p,std::vector<Vec4>& v) const {
    p.resize(count_);v.resize(count_);glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);
    glBindBuffer(GL_COPY_READ_BUFFER,positions_);glGetBufferSubData(GL_COPY_READ_BUFFER,0,GLsizeiptr(p.size()*sizeof(Vec4)),p.data());
    glBindBuffer(GL_COPY_READ_BUFFER,velocities_);glGetBufferSubData(GL_COPY_READ_BUFFER,0,GLsizeiptr(v.size()*sizeof(Vec4)),v.data());
}
std::uint32_t EntityManager::eraseChunk(std::uint32_t chunk,GLuint program,GLint maxGroups) {
    if(!count_) return 0;
    std::array<GLuint,4> fresh{};
    try {
        glGenBuffers(4,fresh.data());
        const std::array<std::size_t,4> sizes{std::size_t(capacity_)*sizeof(Vec4),std::size_t(capacity_)*sizeof(Vec4),std::size_t(capacity_)*sizeof(std::uint32_t),sizeof(std::uint32_t)};
        for(std::size_t i=0;i<fresh.size();++i) {
            glBindBuffer(GL_SHADER_STORAGE_BUFFER,fresh[i]);glBufferData(GL_SHADER_STORAGE_BUFFER,GLsizeiptr(sizes[i]),nullptr,GL_DYNAMIC_COPY);
        }
        const GLuint zero=0;glBindBuffer(GL_SHADER_STORAGE_BUFFER,fresh[3]);
        glBufferSubData(GL_SHADER_STORAGE_BUFFER,0,sizeof(zero),&zero);checkGL("Allocate chunk compaction");
        const std::array<GLuint,8> bindings{positions_,velocities_,identities_,fresh[3],membership_,fresh[0],fresh[1],fresh[2]};
        for(GLuint i=0;i<bindings.size();++i) glBindBufferBase(GL_SHADER_STORAGE_BUFFER,i,bindings[i]);
        glUseProgram(program);glUniform1ui(glGetUniformLocation(program,"particleCount"),count_);glUniform1ui(glGetUniformLocation(program,"targetChunk"),chunk);
        const auto batch=std::uint64_t(maxGroups)*256;
        for(std::uint64_t first=0;first<count_;first+=batch) {
            glUniform1ui(glGetUniformLocation(program,"baseParticle"),GLuint(first));
            glDispatchCompute(GLuint((std::min(batch,std::uint64_t(count_)-first)+255)/256),1,1);
        }
        glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT|GL_SHADER_STORAGE_BARRIER_BIT|GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT);
        GLuint survivors=0;glBindBuffer(GL_COPY_READ_BUFFER,fresh[3]);glGetBufferSubData(GL_COPY_READ_BUFFER,0,sizeof(survivors),&survivors);
        checkGL("Compact chunk particles");
        if(survivors>count_) throw std::runtime_error("Invalid survivor count");
        const auto removed=count_-survivors;
        for(auto b:{positions_,velocities_,identities_}) glDeleteBuffers(1,&b);
        positions_=fresh[0];velocities_=fresh[1];identities_=fresh[2];count_=survivors;
        // massUpperBound_ remains conservative until restart/setState.
        glDeleteBuffers(1,&fresh[3]);return removed;
    } catch(...) {glDeleteBuffers(4,fresh.data());throw;}
}
void EntityManager::bind() const {
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER,0,positions_);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER,1,velocities_);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER,2,accelerations_);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER,4,membership_);
}
}
