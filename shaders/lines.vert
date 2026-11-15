#version 430 core
layout(location=0) in vec3 position;
layout(location=1) in vec3 color;
uniform mat4 viewProjection;
out vec3 tint;
void main(){gl_Position=viewProjection*vec4(position,1);tint=color;}
