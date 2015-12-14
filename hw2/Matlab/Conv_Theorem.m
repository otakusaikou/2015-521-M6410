clc;
clear all;
Img = rgb2gray(imread('machine.png'));               % Read image
[M , N] = size(Img);                    % Get size of image

Mask = [-1 0 1;-2 0 2;-1 0 1];          % Mask
%Mask = [1 0;0 -1];
%Mask = rand(10, 10);
%Mask = rand(100, 100);
%Mask = rand(M, N);
[w, h] = size(Mask);                    % Get size of mask

M = size(Img, 1) + size(Mask, 1);       % Add size of matrix and image
N = size(Img, 2) + size(Mask, 2);

tic;
% Fast Fourier transfomrm method
% Transform image and mask from spatial domain to frequency domain
ImgFFT = fft2(double(Img), M, N);
MaskFFT = fft2(double(Mask), M, N);

ConvFFT = (ImgFFT.*MaskFFT);            % Convolution of image and mask

ResImg1 = ifft2(ConvFFT);               % Result of FFT method
% Adjust dimensions
ResImg1 = ResImg1(floor(w/2)+1:end-floor(w/2)-rem(M,2), floor(h/2)+1:end-floor(h/2)-rem(N,2));
toc;
% Transform result of first method to frequency domain
ResImg1FFT = fft2(ResImg1);

tic;
% Regular convolution method
% Result of regular convolution
ResImg2 = conv2(double(Img), double(Mask), 'same');

% Transform result of second method to frequency domain
ResImg2FFT = fft2(ResImg2);
toc;

% Plot results
% Original image
subplot(2,4,1);
imshow(Img);
title('Original', 'fontsize', 15, 'FontWeight','bold');

% Mask in spatial domain
subplot(2,4,2);
imshow(Mask);
title('Mask', 'fontsize', 15, 'FontWeight','bold');

% Result of regular convolution
subplot(2,4,3);
imshow(uint8(ResImg2));
title('Regular Convolution', 'fontsize', 15, 'FontWeight','bold');

% Result of regular convolution in frequency domain
subplot(2,4,4);
imshow(log(abs(fftshift(ResImg2FFT)) + 1), []);
title({'Regular Convolution', '(Frequency domain)'}, 'fontsize', 15, 'FontWeight','bold');

% Original image in frequency domain
subplot(2,4,5);
imshow(log(abs(fftshift(ImgFFT)) + 1), []);
title({'Original', '(Frequency domain)'}, 'fontsize', 15, 'FontWeight','bold');

% Mask in spatial domain
subplot(2,4,6);
imshow(log(abs(fftshift(MaskFFT)) + 1), [])
title({'Mask', '(Frequency domain)'}, 'fontsize', 15, 'FontWeight','bold');

% Result of FFT method in frequency domain
subplot(2,4,7);
imshow(log(abs(fftshift(ResImg1FFT)) + 1), []);
title({'Result with FFT', '(Frequency domain)'}, 'fontsize', 15, 'FontWeight','bold');

% Result of FFT method in spatial domain
subplot(2,4,8);
imshow(uint8(ResImg1));
title('Result with FFT', 'fontsize', 15, 'FontWeight','bold');
