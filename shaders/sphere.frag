#version 430 core
in vec3 normal,worldPosition;
uniform vec3 eye;
out vec4 color;
void main(){
    float rim=pow(1.0-abs(dot(normalize(normal),normalize(eye-worldPosition))),3.0);
    color=vec4(mix(vec3(0.006,0.009,0.014),vec3(0.81,0.53,0.25),rim),1);
}
