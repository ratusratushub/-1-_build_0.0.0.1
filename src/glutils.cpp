#include <space/GLUtils.hpp>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>
#include <sstream>
namespace space {
namespace {
std::string shaderSource(const std::filesystem::path& path,unsigned depth=0) {
    if(depth>8) throw std::runtime_error("Shader include depth exceeded: "+path.string());
    std::ifstream in(path);if(!in) throw std::runtime_error("Cannot open shader: "+path.string());
    std::ostringstream result;std::string line;unsigned number=0;
    while(std::getline(in,line)) {
        ++number;
        if(line.starts_with("#include \"")) {
            const auto end=line.find('"',10);
            if(end==std::string::npos) throw std::runtime_error("Malformed shader include: "+path.string());
            const auto include=path.parent_path()/line.substr(10,end-10);
            result<<"#line 1\n"<<shaderSource(include,depth+1)<<"\n#line "<<number+1<<'\n';
        } else result<<line<<'\n';
    }
    return result.str();
}
}
GLuint makeProgram(std::span<const std::pair<GLenum,std::filesystem::path>> files) {
    GLuint program=glCreateProgram();std::vector<GLuint> shaders;
    try {
        for(const auto& [type,path]:files) {
            std::string text=shaderSource(path);const char* ptr=text.c_str();
            GLuint shader=glCreateShader(type);shaders.push_back(shader);
            glShaderSource(shader,1,&ptr,nullptr);glCompileShader(shader);
            GLint ok=0;glGetShaderiv(shader,GL_COMPILE_STATUS,&ok);
            if(!ok) { char log[8192]{};glGetShaderInfoLog(shader,sizeof(log),nullptr,log);throw std::runtime_error(path.string()+": "+log); }
            glAttachShader(program,shader);
        }
        glLinkProgram(program);GLint ok=0;glGetProgramiv(program,GL_LINK_STATUS,&ok);
        if(!ok) {char log[8192]{};glGetProgramInfoLog(program,sizeof(log),nullptr,log);throw std::runtime_error(std::string("Program link: ")+log);}
    } catch(...) { for(auto s:shaders) glDeleteShader(s);glDeleteProgram(program);throw; }
    for(auto s:shaders) {glDetachShader(program,s);glDeleteShader(s);}return program;
}
void checkGL(const char* operation) {
    GLenum first=glGetError();if(first==GL_NO_ERROR) return;
    while(glGetError()!=GL_NO_ERROR) {}
    throw std::runtime_error(std::string(operation)+" (OpenGL error "+std::to_string(first)+")");
}
}
