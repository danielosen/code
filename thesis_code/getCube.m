%Generating the Unit Cube Mesh using small cubes
%with 8 corner nodes + 1 center node.
%N = 8^n, n the number of such cubes for mesh (i.e. N=8,64,512,..)
%The code assumes the cube is placed with a lower corner in 0,0,0
%then centers it on the origin afterwards.

%Author: Daniel Osen
function [p tri edge] = getCube(N),
    %grab grid distances
    m = round(log(N)/log(8)); 
    
    %build grid spacing
    vec = zeros(m+1,1);
    for i=1:(m+1)
        vec(i) = (i-1)*1/m;
    end
    
    %construct mesh without centerpoints, no duplicates from this
    [x,y,z] = meshgrid(vec,vec,vec);
    X = [x(:),y(:),z(:)];
    
    %build gridspacing for center points
    vec = zeros(m,1);
    for i=1:m
        vec(i) = 1/(2*m) + (i-1)*1/m;
    end
    %construct mesh of centerpoints
    [x,y,z] = meshgrid(vec,vec,vec);
    X_c = [x(:),y(:),z(:)];
    
    %add meshes together (unique)
    p = [X;X_c];
    %center mesh in origin
    %p = p-1/2;
    %grab triangulation
    tri = delaunay(p);
    topology = TriRep(tri,p);
    edge = freeBoundary(topology);
    %use tetramesh to look
end

