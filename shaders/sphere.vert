#version 430 core
layout(location=0) in vec3 position;
uniform vec4 body;
uniform mat4 viewProjection;
out vec3 normal,worldPosition;
void main(){normal=position;worldPosition=body.xyz+position*body.w;gl_Position=viewProjection*vec4(worldPosition,1);}
