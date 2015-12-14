
#include "slic_common.h"
#include "slic_sequential.h"
#include "slic_parallel.h"


int main(void) {

	srand(time(NULL));

	/*
	if (argc != 3){
		printf("usage: %s N_Pixel N_Iteration\n",argv[0]);
		exit(1);
	}

	*/
	//int N_Pixel = atoi(argv[1]);
	//int N_Iteration = atoi(argv[2]);


	//create_PPM_img(100,100,20);

	int N_Pixel = 1024;
	int N_Iteration = 50;
	char filename[] = "inputs/lena.ppm";

	printf("N_Pixel %d N_Iteration %d\n",N_Pixel, N_Iteration);

	PPM_Image* img_cpu = read_PPM_img(filename);
	PPM_Image* img_gpu = read_PPM_img(filename);

	//SLIC_Sequential(img_cpu, N_Pixel, N_Iteration);
	//write_PPM_img(img_cpu,"cpu_lena.ppm");

	SLIC_Parallel(img_gpu, N_Pixel, N_Iteration);
	write_PPM_img(img_gpu,"gpu_lena.ppm");

	free(img_cpu->data);
	free(img_cpu);
	free(img_gpu->data);
	free(img_gpu);


	return 0;
}






