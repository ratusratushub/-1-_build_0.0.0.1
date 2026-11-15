#version 430 core
out vec4 color;
void main(){
    float r=length(gl_PointCoord*2.0-1.0);
    if(r>1.0) discard;
    float alpha=exp(-3.0*r*r)*(1.0-smoothstep(0.7,1.0,r));
    color=vec4(mix(vec3(0.48,0.71,0.79),vec3(0.94,0.96,0.87),exp(-10.0*r*r)),alpha);
}
