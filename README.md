# SLIC-Superpixel-Segmentation

Compile instructions:

This project is done on a Mac OS X 10.11.1 
With Nsight Eclipse Edition and CUDA Version: 7.5
The Graphics card used is GeForce GT 750 m with compute capabulity 3.0

If you have Nsight Eclipse installed on your Mac you can import this project and run it directly.
If you are using other software such as visual studio then you have to create a new CUDA project and manually add all the source files into a newly created folder called"src", and import the PPM format image under a newly created folder called "inputs", and create a folder called "outputs" for image ouputs.

To change the input file for the program, go to SLIC.cu and under main function, change the variable "filename" to be the path to your input PPM image, and variable N_Pixel and N_Iteration defines the number of desired superpixel and number of iterations to run the program. usually about 10 iteration would be enough to generate convergent result.

To install nsight eclipse edition on your Mac, make sure you have a nVidia graphics card on your machine and then you can go to http://docs.nvidia.com/cuda/cuda-getting-started-guide-for-mac-os-x/#axzz3uFlsjJYU, or search "cuda mac" on google. And follow the instruction there to install the software and CUDA Toolkits.