#include <space/Renderer.hpp>
#include <space/GLUtils.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <array>
#include <cstddef>
#include <numbers>
namespace space {
namespace {
struct Vertex {glm::vec3 position,color;};
GLuint graphics(const std::filesystem::path& root,const char* vert,const char* frag) {
    const std::array<std::pair<GLenum,std::filesystem::path>,2> files{{{GL_VERTEX_SHADER,root/vert},{GL_FRAGMENT_SHADER,root/frag}}};return makeProgram(files);
}
void matrix(GLuint program,const char* name,const glm::mat4& m){glUniformMatrix4fv(glGetUniformLocation(program,name),1,GL_FALSE,glm::value_ptr(m));}
}
Renderer::Renderer(const World& world,const std::filesystem::path& shaders) {
    try {
        lineProgram_=graphics(shaders,"lines.vert","lines.frag");
        pointProgram_=graphics(shaders,"particles.vert","particles.frag");
        sphereProgram_=graphics(shaders,"sphere.vert","sphere.frag");
        std::vector<Vertex> lines;
        const auto line=[&](glm::vec3 a,glm::vec3 b,glm::vec3 color){lines.push_back({a,color});lines.push_back({b,color});};
        const float h=world.config().halfExtent,size=world.chunkSize();
        for(unsigned a=0;a<=world.config().side;++a) for(unsigned b=0;b<=world.config().side;++b) {
            float x=-h+a*size,y=-h+b*size;
            glm::vec3 color{0.14f,0.23f,0.28f};
            line({-h,x,y},{h,x,y},color);line({x,-h,y},{x,h,y},color);line({x,y,-h},{x,y,h},color);
        }
        // The origin lies on the shared corner of eight chunks for an even side.
        for(std::uint32_t id=0;id<world.config().count();++id) {
            auto c=world.chunk(id);
            if(c.minimum.x>0||c.minimum.y>0||c.minimum.z>0||c.maximum.x<0||c.maximum.y<0||c.maximum.z<0) continue;
            glm::vec3 lo{c.minimum.x,c.minimum.y,c.minimum.z},hi{c.maximum.x,c.maximum.y,c.maximum.z};
            for(int axis=0;axis<3;++axis) for(int a=0;a<2;++a) for(int b=0;b<2;++b) {
                glm::vec3 start=lo,end=lo;int u=(axis+1)%3,v=(axis+2)%3;
                start[u]=end[u]=a?hi[u]:lo[u];start[v]=end[v]=b?hi[v]:lo[v];end[axis]=hi[axis];
                line(start,end,{0.38f,0.49f,0.47f});
            }
        }
        // Always draw axes after the optional lattice.
        const float axis=h*1.12f;
        line({-axis,0,0},{axis,0,0},{0.85f,0.32f,0.26f});
        line({0,-axis,0},{0,axis,0},{0.34f,0.72f,0.46f});
        line({0,0,-axis},{0,0,axis},{0.33f,0.56f,0.88f});
        lineCount_=GLsizei(lines.size());
        glGenVertexArrays(1,&linesVAO_);glGenBuffers(1,&linesVBO_);
        glBindVertexArray(linesVAO_);glBindBuffer(GL_ARRAY_BUFFER,linesVBO_);
        glBufferData(GL_ARRAY_BUFFER,GLsizeiptr(lines.size()*sizeof(Vertex)),lines.data(),GL_STATIC_DRAW);
        glEnableVertexAttribArray(0);glVertexAttribPointer(0,3,GL_FLOAT,GL_FALSE,sizeof(Vertex),reinterpret_cast<void*>(offsetof(Vertex,position)));
        glEnableVertexAttribArray(1);glVertexAttribPointer(1,3,GL_FLOAT,GL_FALSE,sizeof(Vertex),reinterpret_cast<void*>(offsetof(Vertex,color)));
        glGenVertexArrays(1,&pointsVAO_);
        std::vector<glm::vec3> sphere;
        const auto vertex=[](int lat,int lon){float p=std::numbers::pi_v<float>*float(lat)/24,t=2*std::numbers::pi_v<float>*float(lon)/48;return glm::vec3(std::sin(p)*std::cos(t),std::cos(p),std::sin(p)*std::sin(t));};
        for(int lat=0;lat<24;++lat) for(int lon=0;lon<48;++lon) {
            auto a=vertex(lat,lon),b=vertex(lat+1,lon),c=vertex(lat+1,lon+1),d=vertex(lat,lon+1);
            sphere.insert(sphere.end(),{a,b,c,a,c,d});
        }
        sphereCount_=GLsizei(sphere.size());glGenVertexArrays(1,&sphereVAO_);glGenBuffers(1,&sphereVBO_);
        glBindVertexArray(sphereVAO_);glBindBuffer(GL_ARRAY_BUFFER,sphereVBO_);
        glBufferData(GL_ARRAY_BUFFER,GLsizeiptr(sphere.size()*sizeof(glm::vec3)),sphere.data(),GL_STATIC_DRAW);
        glEnableVertexAttribArray(0);glVertexAttribPointer(0,3,GL_FLOAT,GL_FALSE,sizeof(glm::vec3),nullptr);
        glBindVertexArray(0);checkGL("Initialize renderer");
    } catch(...) {release();throw;}
}
void Renderer::release() noexcept {
    for(auto p:{lineProgram_,pointProgram_,sphereProgram_}) glDeleteProgram(p);
    for(auto b:{linesVBO_,sphereVBO_}) glDeleteBuffers(1,&b);
    for(auto a:{linesVAO_,pointsVAO_,sphereVAO_}) glDeleteVertexArrays(1,&a);
}
Renderer::~Renderer(){release();}
void Renderer::draw(const Camera& camera,const EntityManager& entities,std::span<const MassiveBody> bodies,int width,int height,bool showGrid) {
    if(width<=0||height<=0) return;
    glViewport(0,0,width,height);glEnable(GL_DEPTH_TEST);glDepthMask(GL_TRUE);
    glClearColor(0.018f,0.028f,0.043f,1);glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);
    const auto vp=camera.viewProjection(float(width)/float(height));
    glDisable(GL_BLEND);glUseProgram(sphereProgram_);matrix(sphereProgram_,"viewProjection",vp);
    glUniform3fv(glGetUniformLocation(sphereProgram_,"eye"),1,glm::value_ptr(camera.position));
    glBindVertexArray(sphereVAO_);
    for(auto b:bodies) {
        glUniform4f(glGetUniformLocation(sphereProgram_,"body"),float(b.x),float(b.y),float(b.z),b.radius);
        glDrawArrays(GL_TRIANGLES,0,sphereCount_);
    }
    glUseProgram(lineProgram_);matrix(lineProgram_,"viewProjection",vp);glBindVertexArray(linesVAO_);
    if(showGrid) glDrawArrays(GL_LINES,0,lineCount_);
    else glDrawArrays(GL_LINES,lineCount_-6,6);
    glEnable(GL_PROGRAM_POINT_SIZE);glEnable(GL_BLEND);glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA);
    glDepthMask(GL_FALSE);glUseProgram(pointProgram_);matrix(pointProgram_,"viewProjection",vp);
    glUniform1f(glGetUniformLocation(pointProgram_,"viewportHeight"),float(height));
    glBindVertexArray(pointsVAO_);glBindBuffer(GL_ARRAY_BUFFER,entities.positionBuffer());
    glEnableVertexAttribArray(0);glVertexAttribPointer(0,4,GL_FLOAT,GL_FALSE,sizeof(Vec4),nullptr);
    glDrawArrays(GL_POINTS,0,GLsizei(entities.count()));
    glDepthMask(GL_TRUE);glDisable(GL_BLEND);glBindVertexArray(0);checkGL("Render scene");
}
}
