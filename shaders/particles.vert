#version 430 core
layout(location=0) in vec4 particle;
uniform mat4 viewProjection;
uniform float viewportHeight;
void main(){
    gl_Position=viewProjection*vec4(particle.xyz,1);
    gl_PointSize=clamp(3.0*viewportHeight/max(gl_Position.w,0.01),2.5,9.0);
}
