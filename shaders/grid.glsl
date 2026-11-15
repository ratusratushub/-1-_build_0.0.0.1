// Shared by index construction and radius queries. The correction compares
// against actual cell faces, preserving negative coordinates very near zero.
int chunkAxis(float value) {
    float size=2.0*halfExtent/float(gridSide);
    int cell=clamp(int(floor((value+halfExtent)/size)),0,int(gridSide)-1);
    while(cell>0 && value<fma(float(cell),size,-halfExtent)) --cell;
    while(cell<int(gridSide)-1 && value>=fma(float(cell+1),size,-halfExtent)) ++cell;
    return cell;
}
ivec3 chunkCoordinates(vec3 p) {return ivec3(chunkAxis(p.x),chunkAxis(p.y),chunkAxis(p.z));}
