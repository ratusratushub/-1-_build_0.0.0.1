#pragma once
struct GLFWwindow;
namespace space {
class Overlay {
    bool glfwReady_=false,glReady_=false;
public:
    explicit Overlay(GLFWwindow* window);
    ~Overlay();
    Overlay(const Overlay&)=delete;
    Overlay& operator=(const Overlay&)=delete;
    void begin();
    void end();
};
}
