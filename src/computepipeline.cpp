#include <space/ComputePipeline.hpp>
#include <space/GLUtils.hpp>
#include <space/Physics.hpp>
#include <algorithm>
#include <limits>
namespace space {
namespace {
void allocate(GLuint& buffer,std::size_t size,GLenum usage) {
    glGenBuffers(1,&buffer);glBindBuffer(GL_SHADER_STORAGE_BUFFER,buffer);
    glBufferData(GL_SHADER_STORAGE_BUFFER,GLsizeiptr(size),nullptr,usage);
    try{checkGL("Allocate metadata");}catch(...){glDeleteBuffers(1,&buffer);buffer=0;throw;}
}
void scalar(GLuint p,const char* name,float value) {glUniform1f(glGetUniformLocation(p,name),value);}
void integer(GLuint p,const char* name,GLuint value) {glUniform1ui(glGetUniformLocation(p,name),value);}
GLuint kernel(const std::filesystem::path& path) {
    const std::array<std::pair<GLenum,std::filesystem::path>,1> files{{{GL_COMPUTE_SHADER,path}}};return makeProgram(files);
}
}
ComputePipeline::ComputePipeline(EntityManager& e,GridConfig grid,const std::filesystem::path& path) : entities_(e),grid_(grid) {
    World validated(grid);
    try {
        GLint bindings=0,size=0,invocations=0;
        glGetIntegerv(GL_MAX_SHADER_STORAGE_BUFFER_BINDINGS,&bindings);
        glGetIntegerv(GL_MAX_COMPUTE_WORK_GROUP_INVOCATIONS,&invocations);
        glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_SIZE,0,&size);
        glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_COUNT,0,&maxGroups_);
        if(bindings<8||invocations<256||size<256||maxGroups_<=0) throw std::runtime_error("Insufficient compute capabilities");
        erase_=kernel(path/"erase.comp");neighbors_=kernel(path/"neighbors.comp");force_=kernel(path/"force.comp");integrate_=kernel(path/"integrate.comp");index_=kernel(path/"aggregate.comp");
        const auto buckets=grid.count()+1;
        allocate(bodies_,256*sizeof(Vec4),GL_DYNAMIC_DRAW);
        allocate(counts_,2*buckets*sizeof(std::uint32_t),GL_DYNAMIC_COPY);
        allocate(offsets_,(buckets+1)*sizeof(std::uint32_t),GL_DYNAMIC_COPY);
        for(auto& s:snapshots_) allocate(s.buffer,buckets*sizeof(std::uint32_t),GL_STREAM_READ);
        glGenQueries(GLsizei(timers_.size()),timers_.data());checkGL("Initialize compute");
    } catch(...) {release();throw;}
}
void ComputePipeline::release() noexcept {
    for(auto f:frames_) glDeleteSync(f);
    for(auto& s:snapshots_) {if(s.fence) glDeleteSync(s.fence);glDeleteBuffers(1,&s.buffer);}
    for(auto b:{bodies_,counts_,offsets_,sorted_,neighborCounts_}) glDeleteBuffers(1,&b);
    for(auto p:{force_,integrate_,index_,neighbors_,erase_}) glDeleteProgram(p);
    glDeleteQueries(GLsizei(timers_.size()),timers_.data());
}
ComputePipeline::~ComputePipeline(){release();}
void ComputePipeline::dispatch(GLuint program) {
    glUseProgram(program);entities_.bind();
    integer(program,"particleCount",entities_.count());integer(program,"gridSide",grid_.side);
    scalar(program,"halfExtent",grid_.halfExtent);
    const std::uint64_t batch=std::uint64_t(maxGroups_)*256;
    for(std::uint64_t first=0;first<entities_.count();first+=batch) {
        integer(program,"baseParticle",GLuint(first));
        auto n=std::min(batch,std::uint64_t(entities_.count())-first);
        glDispatchCompute(GLuint((n+255)/256),1,1);
    }
}
void ComputePipeline::beginFrame() {
    if(frames_.size()>=3) {
        for(;;) {
            const auto result=glClientWaitSync(frames_.front(),GL_SYNC_FLUSH_COMMANDS_BIT,1000000);
            if(result==GL_WAIT_FAILED) throw std::runtime_error("Frame fence failed");
            if(result!=GL_TIMEOUT_EXPIRED) break;
        }
        glDeleteSync(frames_.front());frames_.pop_front();
    }
    for(std::size_t i=0;i<timers_.size();++i) if(timerPending_[i]) {
        GLint ready=0;glGetQueryObjectiv(timers_[i],GL_QUERY_RESULT_AVAILABLE,&ready);
        if(ready) {GLuint64 ns=0;glGetQueryObjectui64v(timers_[i],GL_QUERY_RESULT,&ns);lastGPUms_=double(ns)/1e6;timerPending_[i]=false;}
    }
}
bool ComputePipeline::pollStatistics(std::vector<std::uint32_t>& result) {
    Snapshot* latest=nullptr;
    for(auto& s:snapshots_) if(s.fence) {
        auto status=glClientWaitSync(s.fence,0,0);
        if(status==GL_WAIT_FAILED) throw std::runtime_error("Snapshot fence failed");
        if((status==GL_ALREADY_SIGNALED||status==GL_CONDITION_SATISFIED)&&s.sequence>consumed_&&(!latest||s.sequence>latest->sequence)) latest=&s;
    }
    if(!latest) return false;
    result.resize(grid_.count()+1);glBindBuffer(GL_COPY_READ_BUFFER,latest->buffer);
    glGetBufferSubData(GL_COPY_READ_BUFFER,0,GLsizeiptr(result.size()*sizeof(std::uint32_t)),result.data());
    consumed_=latest->sequence;
    for(auto& s:snapshots_) if(s.fence&&s.sequence<=consumed_) {glDeleteSync(s.fence);s.fence=nullptr;}
    checkGL("Read chunk statistics");return true;
}
void ComputePipeline::uploadBodies(std::span<const MassiveBody> bodies,unsigned mode) {
    if(bodies.size()>256||mode>1||(mode&&entities_.count()>4096)) throw std::invalid_argument("Body count or exact gravity limit exceeded");
    std::vector<Vec4> packed;double total=0;
    for(auto b:bodies) {
        if(!std::isfinite(b.x)||!std::isfinite(b.y)||!std::isfinite(b.z)||!std::isfinite(b.mass)||b.mass<=0||std::abs(b.x)>1e12||std::abs(b.y)>1e12||std::abs(b.z)>1e12||b.mass>1e20||!std::isfinite(b.radius)||b.radius<=0||b.radius>1e9f) throw std::invalid_argument("Invalid massive body");
        packed.push_back({float(b.x),float(b.y),float(b.z),float(b.mass)});total+=b.mass;
    }
    sourceMass_=float(total);bodyCount_=GLuint(bodies.size());particleGravity_=mode;
    glBindBuffer(GL_SHADER_STORAGE_BUFFER,bodies_);glBufferData(GL_SHADER_STORAGE_BUFFER,256*sizeof(Vec4),nullptr,GL_DYNAMIC_DRAW);
    if(!packed.empty()) glBufferSubData(GL_SHADER_STORAGE_BUFFER,0,GLsizeiptr(packed.size()*sizeof(Vec4)),packed.data());
    checkGL("Upload bodies");
}
void ComputePipeline::step(float dt, float G, float eps) {
    if (!std::isfinite(dt) || !std::isfinite(G) || !std::isfinite(eps) || 
        G <= 0 || G > 1e4f || eps < 0.001f || eps > 1e6f || dt <= 0 ||
        dt > safeStep(sourceMass_, G, eps) * 1.001f) {
        throw std::invalid_argument("Unsafe physics step; use smaller fixed substeps");
    }
    
    // УБРАЛИ ограничение: if(particleGravity_ && entities_.count() > 4096) throw...
    
    const auto freeTimer = std::find(timerPending_.begin(), timerPending_.end(), false);
    std::size_t timer = std::size_t(freeTimer - timerPending_.begin());
    
    if (timer < timers_.size()) glBeginQuery(GL_TIME_ELAPSED, timers_[timer]);
    
    // Привязываем все буферы
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, positions_);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, velocities_);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, accelerations_);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, bodies_);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, membership_);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, offsets_);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, index_);
    
    glUseProgram(force_);
    integer(force_, "bodyCount", bodyCount_);
    scalar(force_, "gravityConstant", G);
    scalar(force_, "softening", eps);
    
    glUseProgram(integrate_);
    scalar(integrate_, "dt", dt);
    
    // Kick-drift-kick: force passes read immutable positions
    dispatch(force_);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    
    glUseProgram(integrate_);
    integer(integrate_, "phase", 0);
    dispatch(integrate_);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    
    dispatch(force_);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    
    glUseProgram(integrate_);
    integer(integrate_, "phase", 1);
    dispatch(integrate_);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT | GL_BUFFER_UPDATE_BARRIER_BIT);
    
    reindex();
    
    if (timer < timers_.size()) {
        glEndQuery(GL_TIME_ELAPSED);
        timerPending_[timer] = true;
    }
    
    capture();
    auto fence = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
    if (!fence) throw std::runtime_error("Frame fence failed");
    frames_.push_back(fence);
    glFlush();
    checkGL("Simulation step");
}
void ComputePipeline::reindex() {
    if(sortedCapacity_<entities_.capacity()) {
        GLuint fresh=0;
        allocate(fresh,std::size_t(entities_.capacity())*sizeof(std::uint32_t),GL_DYNAMIC_COPY);
        glDeleteBuffers(1,&sorted_);sorted_=fresh;sortedCapacity_=entities_.capacity();
    }
    glBindBuffer(GL_SHADER_STORAGE_BUFFER,counts_);const GLuint zero=0;
    glClearBufferData(GL_SHADER_STORAGE_BUFFER,GL_R32UI,GL_RED_INTEGER,GL_UNSIGNED_INT,&zero);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER,5,counts_);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER,6,offsets_);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER,7,sorted_);
    glUseProgram(index_);integer(index_,"mode",0);dispatch(index_);glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    integer(index_,"mode",1);integer(index_,"gridSide",grid_.side);glDispatchCompute(1,1,1);glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    integer(index_,"mode",2);dispatch(index_);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT|GL_BUFFER_UPDATE_BARRIER_BIT);
    checkGL("Build particle chunk index");
}
void ComputePipeline::capture() {
    auto s=std::find_if(snapshots_.begin(),snapshots_.end(),[](auto& value){return !value.fence;});
    if(s==snapshots_.end()) return;
    glBindBuffer(GL_COPY_READ_BUFFER,counts_);glBindBuffer(GL_COPY_WRITE_BUFFER,s->buffer);
    glCopyBufferSubData(GL_COPY_READ_BUFFER,GL_COPY_WRITE_BUFFER,0,0,GLsizeiptr((grid_.count()+1)*sizeof(std::uint32_t)));
    s->fence=glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE,0);if(!s->fence) throw std::runtime_error("Statistics fence failed");
    s->sequence=++sequence_;glFlush();
}
void ComputePipeline::resetStatistics() {
    for(auto& s:snapshots_) if(s.fence) {glDeleteSync(s.fence);s.fence=nullptr;}
    consumed_=sequence_;lastGPUms_=-1;
}
std::uint32_t ComputePipeline::clearChunk(std::uint32_t chunk) {
    if(chunk>grid_.count()) throw std::out_of_range("Chunk id outside directory");
    reindex();GLuint occupants=0;glBindBuffer(GL_COPY_READ_BUFFER,counts_);
    glGetBufferSubData(GL_COPY_READ_BUFFER,GLintptr(chunk)*sizeof(GLuint),sizeof(occupants),&occupants);
    checkGL("Read chunk occupancy");if(!occupants) return 0;
    const auto removed=entities_.eraseChunk(chunk,erase_,maxGroups_);
    resetStatistics();reindex();capture();return removed;
}
GLuint ComputePipeline::countNeighbors(float radius) {
    if(!std::isfinite(radius)||radius<0||radius>1e9f) throw std::invalid_argument("Invalid interaction radius");
    if(neighborCapacity_<entities_.capacity()) {
        GLuint fresh=0;allocate(fresh,std::size_t(entities_.capacity())*sizeof(std::uint32_t),GL_DYNAMIC_COPY);
        glDeleteBuffers(1,&neighborCounts_);neighborCounts_=fresh;neighborCapacity_=entities_.capacity();
    }
    reindex();glUseProgram(neighbors_);scalar(neighbors_,"radius",radius);
    // dispatch() binds acceleration at 2; override it before each dispatch here.
    glUseProgram(neighbors_);entities_.bind();
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER,2,neighborCounts_);
    integer(neighbors_,"particleCount",entities_.count());integer(neighbors_,"gridSide",grid_.side);scalar(neighbors_,"halfExtent",grid_.halfExtent);
    const std::uint64_t batch=std::uint64_t(maxGroups_)*256;
    for(std::uint64_t first=0;first<entities_.count();first+=batch) {
        integer(neighbors_,"baseParticle",GLuint(first));
        auto n=std::min(batch,std::uint64_t(entities_.count())-first);glDispatchCompute(GLuint((n+255)/256),1,1);
    }
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT|GL_BUFFER_UPDATE_BARRIER_BIT);checkGL("Neighbor query");return neighborCounts_;
}
void ComputePipeline::readIndex(std::vector<std::uint32_t>& membership,std::vector<std::uint32_t>& offsets,std::vector<std::uint32_t>& sorted) const {
    glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);
    membership.resize(entities_.count());offsets.resize(grid_.count()+2);sorted.resize(entities_.count());
    const auto read=[](GLuint b,auto& v){glBindBuffer(GL_COPY_READ_BUFFER,b);glGetBufferSubData(GL_COPY_READ_BUFFER,0,GLsizeiptr(v.size()*sizeof(std::uint32_t)),v.data());};
    read(entities_.membershipBuffer(),membership);read(offsets_,offsets);read(sorted_,sorted);checkGL("Read validation index");
}
}
