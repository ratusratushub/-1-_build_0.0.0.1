#include <glad/gl.h>
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <space/ComputePipeline.hpp>
#include <space/Physics.hpp>
#include <space/Renderer.hpp>
#include <space/Overlay.hpp>
#include <array>
#include <iostream>
#include <string>
#include <thread>
namespace {
struct Options {
    space::SimulationConfig config;
    unsigned frames=0;
    bool hidden=false,vsync=true;
};
Options parse(int argc,char** argv) {
    Options o;
    auto uintValue=[](const std::string& s) {
        std::size_t end=0;if(s.empty()||s[0]=='-') throw std::invalid_argument("Expected unsigned integer");
        auto value=std::stoull(s,&end);
        if(end!=s.size()||value>0xffffffffull) throw std::invalid_argument("Integer out of range");
        return std::uint32_t(value);
    };
    auto floatValue=[](const std::string& s) {std::size_t end=0;auto v=std::stof(s,&end);if(end!=s.size()||!std::isfinite(v)) throw std::invalid_argument("Expected finite number");return v;};
    for(int i=1;i<argc;++i) {
        std::string key=argv[i];
        if(key=="--hidden") {o.hidden=true;continue;}
        if(key=="--no-vsync") {o.vsync=false;continue;}
        if(key=="--exact") {o.config.particleGravity=1;continue;}
        if(i+1==argc) throw std::invalid_argument("Missing value for "+key);
        std::string value=argv[++i];
        if(key=="--count") o.config.initialCount=uintValue(value);
        else if(key=="--seed") o.config.seed=uintValue(value);
        else if(key=="--mass") o.config.blackHoleMass=floatValue(value);
        else if(key=="--radius") o.config.blackHoleRadius=floatValue(value);
        else if(key=="--min-mass") o.config.minMass=floatValue(value);
        else if(key=="--max-mass") o.config.maxMass=floatValue(value);
        else if(key=="--min-velocity") o.config.minVelocity=floatValue(value);
        else if(key=="--max-velocity") o.config.maxVelocity=floatValue(value);
        else if(key=="--step") o.config.fixedStep=floatValue(value);
        else if(key=="--softening") o.config.softening=floatValue(value);
        else if(key=="--frames") o.frames=uintValue(value);
        else throw std::invalid_argument("Unknown option: "+key);
    }
    o.config.validate();return o;
}
void GLAD_API_PTR debugMessage(GLenum,GLenum type,GLuint,GLenum severity,GLsizei,const GLchar* message,const void*) {
    if(type==GL_DEBUG_TYPE_ERROR||severity==GL_DEBUG_SEVERITY_HIGH) std::cerr<<"OpenGL: "<<message<<'\n';
}
void run(GLFWwindow* window,const Options& options) {
    auto config=options.config;
    space::World world(config.world);space::ChunkManager chunks(config.world);
    space::ThreadPool workers(std::min(8u,std::max(1u,std::thread::hardware_concurrency())));
    space::EntityManager entities;entities.initialize(config);
    space::ComputePipeline compute(entities,config.world,SPACE_SHADER_DIR);
    space::Renderer renderer(world,SPACE_SHADER_DIR);space::Camera camera;
    space::Overlay overlay(window);space::SimulationController clock;
    std::vector<space::MassiveBody> bodies{{0,0,0,config.blackHoleMass,0,0,0,config.blackHoleRadius,true}};
    compute.uploadBodies(bodies,config.particleGravity);compute.reindex();compute.capture();
    bool showGrid=true,captured=false;
    int targetChunk[3]{5,5,5},fillCount=100;std::string maintenanceError;
    std::array<bool,GLFW_KEY_LAST+1> previous{};
    auto pressed=[&](int key){bool now=glfwGetKey(window,key)==GLFW_PRESS;bool edge=now&&!previous[std::size_t(key)];previous[std::size_t(key)]=now;return edge;};
    double last=glfwGetTime(),lastMouseX=0,lastMouseY=0;
    unsigned frames=0;
    std::vector<std::uint32_t> histogram;
    while(!glfwWindowShouldClose(window) && (!options.frames||frames<options.frames)) {
        glfwPollEvents();const double now=glfwGetTime(),elapsed=std::clamp(now-last,0.,0.25);last=now;
        int width=0,height=0;glfwGetFramebufferSize(window,&width,&height);
        if(width==0||height==0) {glfwWaitEventsTimeout(0.02);continue;}
        compute.beginFrame();
        if(compute.pollStatistics(histogram)) {
            // Sparse record preparation stays off the GL thread.
            workers.submit([&]{chunks.update(histogram);}).get();
        }
        overlay.begin();
        auto& io=ImGui::GetIO();bool singleStep=false,restart=false,fillChunk=false,clearChunk=false;
        if(!io.WantCaptureKeyboard) {
            if(pressed(GLFW_KEY_P)) clock.paused=!clock.paused;
            if(pressed(GLFW_KEY_N)) {clock.paused=true;singleStep=true;}
            if(pressed(GLFW_KEY_R)) camera.reset();
            if(pressed(GLFW_KEY_F)) camera.focus();
            if(pressed(GLFW_KEY_HOME)) restart=true;
        }
        const bool rotate=glfwGetMouseButton(window,GLFW_MOUSE_BUTTON_RIGHT)==GLFW_PRESS && (captured||!io.WantCaptureMouse);
        double mx=0,my=0;glfwGetCursorPos(window,&mx,&my);
        if(rotate!=captured) {
            captured=rotate;glfwSetInputMode(window,GLFW_CURSOR,captured?GLFW_CURSOR_DISABLED:GLFW_CURSOR_NORMAL);
            glfwGetCursorPos(window,&mx,&my);lastMouseX=mx;lastMouseY=my;
        }
        if(captured) camera.rotate(float(mx-lastMouseX),float(my-lastMouseY));
        lastMouseX=mx;lastMouseY=my;
        if(!io.WantCaptureMouse) camera.scroll(io.MouseWheel);
        if(!io.WantCaptureKeyboard) {
            auto down=[&](int key){return glfwGetKey(window,key)==GLFW_PRESS?1.f:0.f;};
            camera.move(down(GLFW_KEY_W)-down(GLFW_KEY_S),down(GLFW_KEY_D)-down(GLFW_KEY_A),down(GLFW_KEY_SPACE)-down(GLFW_KEY_LEFT_CONTROL),float(elapsed),down(GLFW_KEY_LEFT_SHIFT)>0);
        }
        if(glfwGetKey(window,GLFW_KEY_ESCAPE)==GLFW_PRESS) glfwSetWindowShouldClose(window,GLFW_TRUE);
        ImGui::SetNextWindowPos(ImVec2(18,18),ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(326,0),ImGuiCond_FirstUseEver);
        ImGui::Begin("SPACE / FOUNDATION",nullptr,ImGuiWindowFlags_AlwaysAutoResize);
        ImGui::TextUnformatted("WORLD  /  XYZ");ImGui::Separator();
        ImGui::Text("Particles  %u / capacity %u",entities.count(),entities.capacity());
        ImGui::Text("GPU limit  %u particles",entities.limit());
        ImGui::Text("Chunks*    %zu active / %zu",chunks.active().size(),config.world.count());
        ImGui::Text("Outside    %u",chunks.outsideCount());
        ImGui::Text("FPS        %.1f",io.Framerate);
        if(compute.gpuMilliseconds()>=0) ImGui::Text("GPU step   %.3f ms",compute.gpuMilliseconds());
        else ImGui::TextUnformatted("GPU step   waiting for query");
        ImGui::Text("Time       %.3f s",clock.time);
        ImGui::Text("Camera     %.1f  %.1f  %.1f",camera.position.x,camera.position.y,camera.position.z);
        ImGui::TextUnformatted("Black hole 0.0  0.0  0.0");
        ImGui::Separator();
        if(ImGui::Button(clock.paused?"Resume [P]":"Pause [P]")) clock.paused=!clock.paused;
        ImGui::SameLine();if(ImGui::Button("Step [N]")){clock.paused=true;singleStep=true;}
        ImGui::SameLine();if(ImGui::Button("Restart")) restart=true;
        ImGui::SliderFloat("Time scale",&clock.timeScale,0,8,"%.2fx");
        ImGui::SliderFloat("BH mass",&config.blackHoleMass,1000,1000000,"%.0f",ImGuiSliderFlags_Logarithmic);
        ImGui::Checkbox("Chunk boundaries",&showGrid);
        if(clock.droppedTime>0) ImGui::Text("Catch-up discarded: %.3f s",clock.droppedTime);
        ImGui::TextUnformatted("X red / Y green / Z blue");
        ImGui::TextDisabled("* asynchronous chunk statistics");
        if(ImGui::CollapsingHeader("Chunk operations")) {
            ImGui::InputInt3("Chunk XYZ",targetChunk);
            for(auto& axis:targetChunk) axis=std::clamp(axis,0,int(config.world.side)-1);
            ImGui::InputInt("Add particles",&fillCount);fillCount=std::clamp(fillCount,1,1000000);
            if(ImGui::Button("Fill chunk")) fillChunk=true;
            ImGui::SameLine();if(ImGui::Button("Clear chunk")) clearChunk=true;
            ImGui::TextWrapped("Maintenance may wait for GPU completion. Existing particle IDs remain stable.");
        }
        if(!maintenanceError.empty()) ImGui::TextWrapped("Operation failed: %s",maintenanceError.c_str());
        if(ImGui::CollapsingHeader("Controls & model")) {
            ImGui::TextUnformatted("WASD / Space / Ctrl: move\nRMB + mouse: look; wheel: speed\nShift: boost; R: reset; F: center\nHome: restart; Esc: exit");
            ImGui::TextUnformatted(config.particleGravity?"Gravity: bodies + exact particles":"Gravity: external bodies only");
            ImGui::TextWrapped("Softened Newtonian potential. Visible radius is illustrative; no absorption or relativistic effects.");
        }
        ImGui::End();
        if(restart) {
            entities.initialize(config);clock.reset();chunks.clear();compute.resetStatistics();maintenanceError.clear();
            compute.reindex();compute.capture();
        }
        if(!restart&&(fillChunk||clearChunk)) {
            try {
                const auto id=world.index({targetChunk[0],targetChunk[1],targetChunk[2]});
                if(clearChunk) compute.clearChunk(id);
                if(fillChunk) {
                    if(config.particleGravity && std::uint64_t(entities.count())+unsigned(fillCount)>4096) throw std::length_error("Exact gravity is limited to 4096 particles");
                    entities.append(unsigned(fillCount),config,id);compute.resetStatistics();compute.reindex();compute.capture();
                }
                chunks.clear();maintenanceError.clear();
            } catch(const std::exception& error) {clock.paused=true;maintenanceError=error.what();}
        }
        bodies[0].mass=config.blackHoleMass;
        compute.uploadBodies(bodies,config.particleGravity);
        const float totalMass=config.blackHoleMass+float(config.particleGravity?entities.massUpperBound():0);
        const float dt=std::min(config.fixedStep,space::safeStep(totalMass,config.G,config.softening));
        clock.advance(restart?0:elapsed,dt,singleStep,[&](float step){compute.beginFrame();compute.step(step,config.G,config.softening);});
        renderer.draw(camera,entities,bodies,width,height,showGrid);overlay.end();glfwSwapBuffers(window);++frames;
    }
    glFinish();
}
}
int main(int argc,char** argv) {
    GLFWwindow* window=nullptr;
    try {
        if(argc==2&&std::string(argv[1])=="--help") {
            std::cout<<"space_sim [--count N] [--seed N] [--mass M] [--radius R] [--softening E]\n"
                       "          [--min-mass M] [--max-mass M] [--min-velocity V] [--max-velocity V]\n"
                       "          [--step SECONDS] [--exact] [--frames N] [--hidden] [--no-vsync]\n";
            return 0;
        }
        const auto options=parse(argc,argv);
        if(!glfwInit()) throw std::runtime_error("GLFW initialization failed: a display and OpenGL driver are required");
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR,4);glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR,3);
        glfwWindowHint(GLFW_OPENGL_PROFILE,GLFW_OPENGL_CORE_PROFILE);glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT,GLFW_TRUE);
        glfwWindowHint(GLFW_VISIBLE,options.hidden?GLFW_FALSE:GLFW_TRUE);
        window=glfwCreateWindow(1440,900,"Space Simulation Foundation",nullptr,nullptr);
        if(!window) throw std::runtime_error("OpenGL 4.3 context unavailable");
        glfwMakeContextCurrent(window);
        if(!gladLoadGL(glfwGetProcAddress)) throw std::runtime_error("Cannot load OpenGL 4.3");
        glEnable(GL_DEBUG_OUTPUT);glDebugMessageCallback(debugMessage,nullptr);
        glfwSwapInterval(options.vsync?1:0);
        run(window,options);
        glfwDestroyWindow(window);glfwTerminate();return 0;
    } catch(const std::exception& e) {
        std::cerr<<e.what()<<'\n';if(window) glfwDestroyWindow(window);glfwTerminate();return 1;
    }
}
