function [p tri edge] = getLshape(N),
%With the code for the cube complete, this is relatively simple,
%if we lazily sacrifice some control over specifying number of nodes.
%Let therefore N be the number of smaller cubes in one unit cube,
%such that we just combine three unit cubes to form an L-shape
%first output of getCube returns points
p0 = getCube(N);
[m,n] = size(p0);
%build transform matrices
Ix = zeros(m,n);
Iy = Ix;
Ix(:,1) = Ix(:,1)+1;
Iy(:,2) = Iy(:,2)+1;
%construct L-mesh from unit cubes
p = [p0+Ix;p0;p0+Iy];
%remove duplicate points on intersection of unit cubes
%this also sorts p
p = unique(p,'rows');
%triangulate new mesh
tri = delaunay(p);
triremove = [];
[mtri,ntri] = size(tri);

%remove some unwanted elements going across the L: L\
%triremove is the list of unwanted elements
for i=1:mtri
    u = p(tri(i,:),:);
    ucenter = (u(1,1:2)+u(2,1:2)+u(3,1:2)+u(4,1:2))*1/4;
    if ucenter(1)>1
        if ucenter(2)>1
            %check if xy-center of tetrahedron is outside the interior
            %of the actual Lshape
            triremove = [triremove;i];
            
        end
    end
end
%remove the rows corresponding to unwanted elements
tri = removerows(tri,'ind',triremove);
topology = TriRep(tri,p);
%topology = triangulation(tri,p(:,1),p(:,2),p(:,3));
edge = freeBoundary(topology);
end

