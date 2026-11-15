#pragma once
#include <glad/gl.h>
#include <filesystem>
#include <span>
#include <utility>
namespace space {
GLuint makeProgram(std::span<const std::pair<GLenum,std::filesystem::path>> shaders);
void checkGL(const char* operation);
}
