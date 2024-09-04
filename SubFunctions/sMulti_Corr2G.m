function [lambda1,lambda2]= sMulti_Corr2G(in, r)

% sMulti_Corr2G: local multispectral  to Gray Conversion by Correlation for
% multispectral images developed based on Corr2G developed by Hossein Ziaei
% Nafchi (% Color to Gray Conversion by Correlation", IEEE Signal
% Processing Letters, vol. 24, no. 11, pp. 1651-1655, 2017.)
% in: Input Multispectral image
% r: downsampling parameter is set to 256 by default (use 512, 128, 64, ..., smaller --> faster)
%% The main code is first writted by Hossein Ziaei Nafchi (hossein.zi@synchromedia.ca) for RGB images and is modified for multispectral remote sensing images by Armin Moghimi
% Please refer to the following paper:
% Hossein Ziaei Nafchi, Atena Shahkolaei, Rachid Hedjam and Mohamed Cheriet, "CorrC2G:
% Color to Gray Conversion by Correlation", IEEE Signal Processing Letters, vol. 24, no. 11, pp. 1651-1655, 2017.
%This code modeified with Armin Moghimi then to extend this for multispectral images keypoint-detection 

% If you utilize this modified code, please cite the original code and the
% following papers that present the modifications:
% Hossein Ziaei Nafchi, Atena Shahkolaei, Rachid Hedjam and Mohamed Cheriet, "CorrC2G:
% Color to Gray Conversion by Correlation", IEEE Signal Processing Letters, vol. 24, no. 11, pp. 1651-1655, 2017.

% Armin Moghimi, Turgay Celik, Ali Mohammadzadeh (2022)
% "Tensor-based keypoint detection and switching regression model for
% relative radiometric normalization of bitemporal multispectral images,"
% International Journal of Remote Sensing, 43:11, 3927-3956,
% DOI: 10.1080/01431161.2022.2102951


if ~exist('r', 'var')
    r = 256;
end
[n, m, band] = size(in);
f = r / min(n, m);
f = min(f, 1);

if f < 1
    in01 = im2double( imresize(in, f, 'nearest') );
else
    in01 = im2double( in );
end
for i=1:band
   in01(:,:,i)=imadjust(in01(:,:,i)); 
end

Mu = mean(in01, 3); % Mean image
% 
x = bsxfun(@minus, in01, Mu);
stim=std([zeros(1,band),1]);
Sigma = sqrt( sum(abs(x) .^ 2, 3) ./ band ) ./ stim; % standard deviation image for multispectral images 
Q = Mu .* Sigma; % First contrast map
% inline code to compute Pearson's correlations Rho1 and Rho2
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
m0 = mean2(Q);
for i=1:band
    M(i)=mean2(in01(:, :, i));
end
d1 = Q(:) - m0;
for i=1:band
    D(:,:,i)=in01(:, :, i) - M(i);
end
sumd1 = sum(d1 .^ 2);
D=reshape(D,size(D,1)*size(D,2),band);
for i=1:band
    Sum(i)=sum(D(:,i) .^ 2);
end
for i=1:band
Rho1(i)=sum(d1 .* D(:,i)) ./ ( sumd1 * Sum(i)) ^ 0.5;
end
Q = Mu .* (1 - Sigma); % Second contrast map
m0 = mean2(Q);
d1 = Q(:) - m0;
sumd1 = sum(d1 .^ 2);
for i=1:band
Rho2(i)=sum(d1 .* D(:,i)) ./ ( sumd1 * Sum(i)) ^ 0.5;
end
%% method #1
Gamma1 = ( Rho1 - min(Rho1) ) ./ ( max(Rho1) - min(Rho1) ) - 0.5;
beta1 = abs(Rho1);
beta1 = beta1 ./ sum(beta1);
lambda1 = beta1 + min(beta1, Gamma1);
lambda1 = abs(lambda1);
lambda1 = lambda1 ./ sum(lambda1);

Gamma2 = ( Rho2 - min(Rho2) ) ./ ( max(Rho2) - min(Rho2) ) - 0.5;
beta2 = abs(Rho2);
beta2 = beta2 ./ sum(beta2);
lambda2 = beta2 + min(beta2, Gamma2);
lambda2 = abs(lambda2);
lambda2 = lambda2 ./ sum(lambda2);
end