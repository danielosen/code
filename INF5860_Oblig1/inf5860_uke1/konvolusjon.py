import numpy as np
from matplotlib import pyplot as plt
import time

def main():
  img = plt.imread('lena.png')
  plt.imshow(img)

  start = time.time()
  out1 = sobel_filter(img)#sobel_filter(img)
 # out2 = blur_filter(img)
  print 'Calculation time:', time.time()-start, 'sec'
  plt.figure()
  plt.imshow(out1.mean(2), vmin=out1.min(), vmax=out1.max(), cmap='gray')
  #plt.figure()
  #plt.imshow(out2, vmin=out2.min(), vmax=out2.max())
  plt.show()


def convolution(image, kernel):
  """
  Write a general function to convolve an image with an arbitrary kernel.
  """
  #edge cases handled by duplicating pixels
  #assume kernel has a well-defined center
  #Lf ignores edges entirely, thats why they are 0 (default) on edges
  image_x,image_y,image_z = image.shape #rgba on z
  #convention to flip kernel???
  kernel = kernel[::-1,::-1]
  kernel_x,kernel_y = kernel.shape
  kernel_halfsize_x = int(np.floor(kernel_x/2.0))
  kernel_halfsize_y = int(np.floor(kernel_y/2.0))

  copy = np.zeros(image.shape)
  
  for x in range(kernel_halfsize_x,image_x - kernel_halfsize_x):
    for y in range(kernel_halfsize_y,image_y - kernel_halfsize_y):
      for z in range(image_z):
        copy[x,y,z] = np.sum(image[x - kernel_halfsize_x : x + kernel_halfsize_x +1,y - kernel_halfsize_y : y + kernel_halfsize_y + 1,z]*kernel)
  return copy


def blur_filter(img):
  """
  Use your convolution function to filter your image with an average filter (box filter)
  with kernal size of 11.
  """
  k_size = 11
  kernel = np.ones((k_size, k_size))*(1.0/k_size**2)
  return convolution(img, kernel)


def sobel_filter(img):
  """
  Use your convolution function to filter your image with a sobel operator
  """
  kernel = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
  img_x = convolution(img,kernel)
  kernel = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
  img_y = convolution(img,kernel)
  img = np.sqrt(img_x*img_x+img_y*img_y)
  return img


if __name__ == '__main__':
  main()
  #mat = np.arange(16).reshape((4,4))
  #kernel = np.array([[0,0,0],[0,1,0],[0,0,0]])
  #print(convolution(mat,kernel))
  #print(mat.shape)