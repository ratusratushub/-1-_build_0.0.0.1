#include <space/Overlay.hpp>
#include <imgui.h>
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_opengl3.h>
#include <stdexcept>
namespace space {
Overlay::Overlay(GLFWwindow* window) {
    IMGUI_CHECKVERSION();ImGui::CreateContext();ImGui::GetIO().IniFilename=nullptr;
    ImGui::StyleColorsDark();auto& style=ImGui::GetStyle();style.WindowRounding=5;style.FrameRounding=3;
    style.Colors[ImGuiCol_WindowBg]=ImVec4(0.035f,0.055f,0.07f,0.94f);
    style.Colors[ImGuiCol_Button]=ImVec4(0.17f,0.28f,0.31f,1);
    glfwReady_=ImGui_ImplGlfw_InitForOpenGL(window,true);
    if(glfwReady_) glReady_=ImGui_ImplOpenGL3_Init("#version 430 core");
    if(!glReady_) {if(glfwReady_) ImGui_ImplGlfw_Shutdown();ImGui::DestroyContext();throw std::runtime_error("Overlay initialization failed");}
}
Overlay::~Overlay(){if(glReady_) ImGui_ImplOpenGL3_Shutdown();if(glfwReady_) ImGui_ImplGlfw_Shutdown();ImGui::DestroyContext();}
void Overlay::begin(){ImGui_ImplOpenGL3_NewFrame();ImGui_ImplGlfw_NewFrame();ImGui::NewFrame();}
void Overlay::end(){ImGui::Render();ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());}
}
