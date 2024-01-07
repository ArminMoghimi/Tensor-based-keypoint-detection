function [S_DN,R_DN ]=extraction_DN(ptsScene1,ptsObj1,imgObj,imgScene) 
%  The code wriiten by Armin Moghimi (2022)
% Reference:
% Armin Moghimi, Turgay Celik, Ali Mohammadzadeh (2022)
% "Tensor-based keypoint detection and switching regression model for
% relative radiometric normalization of bitemporal multispectral images,"
% International Journal of Remote Sensing, 43:11, 3927-3956,
% DOI: 10.1080/01431161.2022.2102951
for i=1:max(size(ptsObj1))
S_DN(i,1)=imgObj(ptsObj1(i,2),ptsObj1(i,1),1);
end       
 for i=1:max(size(ptsScene1))
R_DN(i,1)=imgScene(ptsScene1(i,2),ptsScene1(i,1),1);
end   