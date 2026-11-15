#pragma once
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <algorithm>
#include <cmath>
namespace space {
class Camera {
    float yaw_=0,pitch_=0;
public:
    glm::vec3 position{1050,700,1250};
    float speed=350;
    Camera(){focus();}
    glm::vec3 direction() const {
        float y=glm::radians(yaw_),p=glm::radians(pitch_);
        return {std::cos(y)*std::cos(p),std::sin(p),std::sin(y)*std::cos(p)};
    }
    void focus() {
        if(glm::length(position)<0.001f) position={0,0,50};
        auto d=glm::normalize(-position);pitch_=glm::degrees(std::asin(d.y));yaw_=glm::degrees(std::atan2(d.z,d.x));
    }
    void reset(){position={1050,700,1250};speed=350;focus();}
    void rotate(float dx,float dy){yaw_+=dx*0.12f;pitch_=std::clamp(pitch_-dy*0.12f,-89.f,89.f);}
    void scroll(float delta){speed=std::clamp(speed*std::exp(delta*0.15f),1.f,5000.f);}
    void move(float forward,float sideways,float vertical,float elapsed,bool fast) {
        const auto front=glm::normalize(glm::vec3(direction().x,0,direction().z));
        const auto right=glm::normalize(glm::cross(front,glm::vec3(0,1,0)));
        auto d=front*forward+right*sideways+glm::vec3(0,vertical,0);
        if(glm::length(d)>0) position+=glm::normalize(d)*speed*elapsed*(fast?4.f:1.f);
    }
    glm::mat4 viewProjection(float aspect) const {
        return glm::perspective(glm::radians(55.f),aspect,0.2f,20000.f)*glm::lookAt(position,position+direction(),glm::vec3(0,1,0));
    }
};
}
