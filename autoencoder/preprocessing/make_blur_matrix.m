pkg load image
h = fspecial('gaussian',9,2)
csvwrite('gaussian_blur.matrix', h)
