#pragma once
#include <space/Camera.hpp>
#include <space/EntityManager.hpp>
#include <space/World.hpp>
#include <filesystem>
namespace space {
class Renderer {
    GLuint lineProgram_=0,pointProgram_=0,sphereProgram_=0;
    GLuint linesVAO_=0,linesVBO_=0,pointsVAO_=0,sphereVAO_=0,sphereVBO_=0;
    GLsizei lineCount_=0,sphereCount_=0;
    void release() noexcept;
public:
    Renderer(const World& world,const std::filesystem::path& shaders);
    ~Renderer();
    Renderer(const Renderer&)=delete;
    Renderer& operator=(const Renderer&)=delete;
    void draw(const Camera& camera,const EntityManager& entities,std::span<const MassiveBody> bodies,int width,int height,bool showGrid=true);
};
}
