function [p tri edge] = getCylinder()
%GETCYLINDER Creates 3D Cylinder mesh
%   Detailed explanation goes here
data = importdata('myMatlab_nodes.m');
p = unique(data(:,2:4),'rows');
tri  = delaunay(p(:,1), p(:,2), p(:,3));
edge     = freeBoundary(TriRep(tri, p));
end

