/*
 * slic_parallel.cu
 *
 *  Created on: Dec 10, 2015
 *      Author: xbin
 */

#include "slic_parallel.h"
#include "math.h"

#define MAX_SHARED_MEM 1024

#define CUDA_CHECK_RETURN(value) {											\
	cudaError_t _m_cudaStat = value;										\
	if (_m_cudaStat != cudaSuccess) {										\
		fprintf(stderr, "Error %s at line %d in file %s\n",					\
				cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);		\
		exit(1);															\
	} }


__global__
void test_kernel(PPM_Pixel* D_img, int Size_img){

	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < Size_img){

		D_img[idx].R *= 0.4;
		D_img[idx].G *= 0.9;
		D_img[idx].B *= 0.1;
	}
}

/*
 * Initialize Center Position and Center Color
 */
__global__
void Init_SLIC_Center_D(PPM_Pixel* D_img, SLIC_Pixel* D_Center, int Cen_Len, int W, int H, int Cen_W){

	int tx = threadIdx.x + blockIdx.x * blockDim.x;
	int ty = threadIdx.y + blockIdx.y * blockDim.y;

	int offset = Cen_Len/2;

	int Cen_X = tx * Cen_Len + offset;
	int Cen_Y = ty * Cen_Len + offset;

	if (Cen_X < W && Cen_Y < H){

		int Cen_img_Idx = Cen_X + Cen_Y * W;
		int Cen_Idx = tx + ty * Cen_W;

		D_Center[Cen_Idx].R = D_img[Cen_img_Idx].R;
		D_Center[Cen_Idx].G = D_img[Cen_img_Idx].G;
		D_Center[Cen_Idx].B = D_img[Cen_img_Idx].B;
		D_Center[Cen_Idx].X = Cen_X;
		D_Center[Cen_Idx].Y = Cen_Y;
	}
}
/*
 * Initialize M and S variable
 */
__global__
void Init_Max_Vector_D(double* M, double* S, int m, int s, int K){

	int tx = threadIdx.x + blockIdx.x * blockDim.x;

	if (tx < K){
		M[tx] = m;
		S[tx] = s;
	}
}
/*
 * Initialize Label for each pixel
 */
__global__
void Init_Label_D(int* D_Label, int Size_img){

	int tx = threadIdx.x + blockIdx.x * blockDim.x;

	if (tx < Size_img){
		D_Label[tx] = -1;
	}
}
/*
 * Initialize Distance Vector for each pixel
 */
__global__
void Reset_Distvec_D(double* D_Distvec, int Size_img){


	int tx = threadIdx.x + blockIdx.x * blockDim.x;


	if (tx < Size_img){

		D_Distvec[tx] = -1;
	}
}
/*
 * Initialize Sum of each Center Vector
 */
__global__
void Reset_Sum_Center_D(SLIC_Pixel* D_Sum_Cen, int* D_Size_Cen, int K){

	int tx = threadIdx.x + blockIdx.x * blockDim.x;

	if (tx < K){

		D_Sum_Cen[tx].R = 0;
		D_Sum_Cen[tx].G = 0;
		D_Sum_Cen[tx].B = 0;
		D_Sum_Cen[tx].X = 0;
		D_Sum_Cen[tx].Y = 0;
		D_Size_Cen[tx] = 0;
	}
}
/*
 * Associate each center to the pixels within its search region
 */
__global__
void Advertise_Label_D(
						PPM_Pixel* D_img,
						int k,
						SLIC_Pixel* D_Center,
						double* D_rgb,
						double* D_xy,
						double* M,
						double* S,
						double* D_Distvec,
						int* D_Label,
						int W,
						int H,
						int offset
						){

	int tx = threadIdx.x + blockIdx.x * blockDim.x;
	int ty = threadIdx.y + blockIdx.y * blockDim.y;

	double cen_r = D_Center[k].R;
	double cen_g = D_Center[k].G;
	double cen_b = D_Center[k].B;
	double cen_x = D_Center[k].X;
	double cen_y = D_Center[k].Y;

	int x = cen_x - offset + tx;
	int y = cen_y - offset + ty;

	if ( 0 <= x && x < W && 0 <= y && y < H){

		int i = x + y * W;

		double r = (double)D_img[i].R;
		double g = (double)D_img[i].G;
		double b = (double)D_img[i].B;
		// calcuate distance vector
		D_rgb[i] =  (r - cen_r) * (r - cen_r) +
					(g - cen_g) * (g - cen_g) +
					(b - cen_b) * (b - cen_b);

		D_xy[i] =	(x - cen_x) * (x - cen_x) +
					(y - cen_y) * (y - cen_y);

		double D_distance = D_rgb[i] / M[k] + D_xy[i] / S[k];
		//double D_distance = D_xy[i] / S[k];
		//double D_distance = D_rgb[i] / M[k];
		// update distance vector for each pixel
		if (D_distance < D_Distvec[i] || D_Distvec[i] == -1){
			// assign new center to this pixel
			D_Distvec[i] = D_distance;
			D_Label[i] = k;
		}
	}
}
/*
 * Sum up Center Vectors
 */
__global__
void Sum_Center_Info(PPM_Pixel* D_img, int* D_Label, SLIC_Pixel* D_Sum_Cen, int* D_Size_Cen, int Size_img, int W, int Cen_Idx_Start, int Block_Size){


	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	int tx = threadIdx.x;

	if(idx < Size_img){

		if ( Cen_Idx_Start <= D_Label[idx] && D_Label[idx] < Cen_Idx_Start + Block_Size){

			__shared__ float histo_R[MAX_SHARED_MEM];
			__shared__ float histo_G[MAX_SHARED_MEM];
			__shared__ float histo_B[MAX_SHARED_MEM];
			__shared__ float histo_X[MAX_SHARED_MEM];
			__shared__ float histo_Y[MAX_SHARED_MEM];
			__shared__ float histo_S[MAX_SHARED_MEM];

			int stride = blockDim.x * gridDim.x;

			// initialize histogram to 0
			if ( tx < Block_Size){

				histo_R[tx] = 0;
				histo_G[tx] = 0;
				histo_B[tx] = 0;
				histo_X[tx] = 0;
				histo_Y[tx] = 0;
				histo_S[tx] = 0;

	 			__syncthreads();
			}

			int Label;

	 		while ( idx < Size_img){

	 			Label = D_Label[idx] - Cen_Idx_Start;

				double r = (double)D_img[idx].R;
				double g = (double)D_img[idx].G;
				double b = (double)D_img[idx].B;
				double x = idx % W;
				double y = idx / W;

	 			atomicAdd( &(histo_R[Label]), r );
	 			atomicAdd( &(histo_G[Label]), g );
	 			atomicAdd( &(histo_B[Label]), b );
	 			atomicAdd( &(histo_X[Label]), x );
	 			atomicAdd( &(histo_Y[Label]), y );
	 			atomicAdd( &(histo_S[Label]), 1 );

				idx += stride;
			}
			__syncthreads();

			if (tx < Block_Size){

				int k = Cen_Idx_Start + tx;

				atomicAdd( &(D_Sum_Cen[k].R), histo_R[tx] );
				atomicAdd( &(D_Sum_Cen[k].G), histo_G[tx] );
				atomicAdd( &(D_Sum_Cen[k].B), histo_B[tx] );
				atomicAdd( &(D_Sum_Cen[k].X), histo_X[tx] );
				atomicAdd( &(D_Sum_Cen[k].Y), histo_Y[tx] );
				atomicAdd( &(D_Size_Cen[k]), histo_S[tx] );

			}
		}
	}
}
/*
 * Update Center Vectors for next iteration
 * Reset Sum of Center Vectors to zero
 */
__global__
void Update_Center(SLIC_Pixel* D_Center, SLIC_Pixel* D_Sum_Cen, int* D_Size_Cen, int K){

	int tx = threadIdx.x + blockIdx.x * blockDim.x;

	if (tx < K){

		double inverse_size_cen = 1 / ((double)D_Size_Cen[tx]);

		D_Center[tx].R = D_Sum_Cen[tx].R * inverse_size_cen;
		D_Center[tx].G = D_Sum_Cen[tx].G * inverse_size_cen;
		D_Center[tx].B = D_Sum_Cen[tx].B * inverse_size_cen;
		D_Center[tx].X = D_Sum_Cen[tx].X * inverse_size_cen;
		D_Center[tx].Y = D_Sum_Cen[tx].Y * inverse_size_cen;

		D_Sum_Cen[tx].R = 0;
		D_Sum_Cen[tx].G = 0;
		D_Sum_Cen[tx].B = 0;
		D_Sum_Cen[tx].X = 0;
		D_Sum_Cen[tx].Y = 0;
		D_Size_Cen[tx] = 0;


	}

}
/*
 * Draw Contour Line based on the generated Label
 */
void Draw_Contour_D(PPM_Image* img, int* Label,int version){

	printf("Draw_Contour..\n");

	int W = img->W;
	int H = img->H;
	int Size_img = img->Size_img;

	int direction_x[8] = {-1, -1,  0,  1, 1, 1, 0, -1};
	int direction_y[8] = { 0, -1, -1, -1, 0, 1, 1,  1};

	// set a label indicating whether each pixel will be used as contour line
	int* on_contour = (int*)malloc(Size_img * sizeof(int));
	for (int i=0; i< Size_img; i++){
		on_contour[i] = 0; // initially no pixel is used as contour
	}

	int idx = 0;
	for (int y=0; y < H; y++){
		for(int x=0; x < W; x++){
			// check surrounding pixels to see if reached border
			for (int i=0; i < 8; i++){
				int x_near = x + direction_x[i];
				int y_near = y + direction_y[i];
				// check if neighbor is within picture
				if ( 0 <= x_near && x_near < W && 0 <= y_near && y_near < H){
					int idx_near = x_near + y_near * W;

					if ( on_contour[idx] == 0){
						if (Label[idx] != Label[idx_near]){
							// set this pixel to be the contour line
							img->data[idx].R = 120;
							img->data[idx].G = 90;
							img->data[idx].B = 250;
							on_contour[idx] = 1;
						}
					}
				}
			}
			// go to the next pixel
			idx ++;
		}
	}
	char filename[] = "label_---.ppm";

	filename[6] = version / 100 + '0';
	filename[7] = (version / 10) % 10 + '0';
	filename[8] = version % 10 + '0';

	write_PPM_img(img, filename);

}
/*
 * Perform SLIC Segmentation using GPU Acceleration
 */
void SLIC_Parallel(PPM_Image* img, unsigned int K, unsigned int N_Iteration){

	printf("SLIC_parallel...\n");
	// img dimension
	int W = img->W;
	int H = img->H;
	int Size_img = img->Size_img;
	// pixel dimension
	double Cen_Len_d = sqrt((double)Size_img / (double)K);
	int Cen_Len = ceil(Cen_Len_d);
	int Cen_W = W / Cen_Len;
	int Cen_H = H / Cen_Len;
	printf("Num Cen W %d\nNum Cen H %d\n",Cen_W, Cen_H);
	printf("Cen_Len %d\n",Cen_Len);
	//---------------------------------------------------------------------------------------------------------
	// Load image data to global memory
	//---------------------------------------------------------------------------------------------------------
	PPM_Pixel* D_img;
	cudaMalloc((void**)&D_img, Size_img * sizeof(PPM_Pixel));
	CUDA_CHECK_RETURN(cudaMemcpy(D_img,img->data,3 * Size_img * sizeof(unsigned char),cudaMemcpyHostToDevice));
	printf("data loaded to global...\n");
	//------------------------------------------------------------------------
	// Center Info
	//------------------------------------------------------------------------
	SLIC_Pixel* D_Center;	// for calculation
	SLIC_Pixel* D_Sum_Cen;	// for sum up after each iteration
	int* D_Size_Cen;		// the size of each center after each iteration
	CUDA_CHECK_RETURN(cudaMalloc((void**)&D_Center, K * sizeof(SLIC_Pixel)));
	CUDA_CHECK_RETURN(cudaMalloc((void**)&D_Sum_Cen, K * sizeof(SLIC_Pixel)));
	CUDA_CHECK_RETURN(cudaMalloc((void**)&D_Size_Cen, K * sizeof(int)));
	printf("center info allocated...\n");
	//------------------------------------------------------------
	// M and S variable
	//------------------------------------------------------------
	double* M;
	double* S;
	double* H_M = (double*)malloc(K * sizeof(double));
	double* H_S = (double*)malloc(K * sizeof(double));
	CUDA_CHECK_RETURN(cudaMalloc((void**)&M, K * sizeof(double)));
	CUDA_CHECK_RETURN(cudaMalloc((void**)&S, K * sizeof(double)));
	printf("M S allocated...\n");
	//-----------------------------------------------------------
	// rgb and xy
	//-----------------------------------------------------------
	double* D_rgb;
	double* D_xy;
	double* H_rgb = (double*)malloc(Size_img * sizeof(double));
	double* H_xy = (double*)malloc(Size_img * sizeof(double));
	CUDA_CHECK_RETURN(cudaMalloc((void**)&D_rgb, Size_img * sizeof(double)));
	CUDA_CHECK_RETURN(cudaMalloc((void**)&D_xy, Size_img * sizeof(double)));
	//---------------------------------------------------------------------------
	// Distance vectors for each pixel
	//---------------------------------------------------------------------------
	double* D_Distvec;
	CUDA_CHECK_RETURN(cudaMalloc((void**)&D_Distvec, Size_img * sizeof(double)));
	printf("D_Distvec allocated...\n");
	//----------------------------------------------------------------------
	// Labels for each pixel
	//----------------------------------------------------------------------
	int* D_Label;
	CUDA_CHECK_RETURN(cudaMalloc((void**)&D_Label, Size_img * sizeof(int)));
	int* Label = (int*)malloc(Size_img * sizeof(int));
	printf("D_Label allocated...\n");
	printf("---------------------------\n");
	//###########################
	// center kernel dimension 1D
	//###########################
	int Block_Size_Cen_1D = K;
	if (Block_Size_Cen_1D > 512){
		Block_Size_Cen_1D = 512;
	}
	int Grid_Size_Cen_1D = (K-1)/Block_Size_Cen_1D + 1;
	printf("Block_Size_Cen_1D %d\n",Block_Size_Cen_1D);
	printf("Grid_Size_Cen_1D %d\n",Grid_Size_Cen_1D);
	printf("---------------------------\n");
	//##############################
	// center sum kernel dimesion 1D
	//##############################
	int Block_Sum_Cen_1D = K;
	int Sum_Cen_Itr = 1;	// if one block can cover all the number of centers, 1 iteration is enough
	if (Block_Sum_Cen_1D > MAX_SHARED_MEM){
		Block_Sum_Cen_1D = MAX_SHARED_MEM;
		Sum_Cen_Itr = (K-1)/MAX_SHARED_MEM + 1;
	}
	int Grid_Sum_Cen_1D = (Size_img-1)/Block_Sum_Cen_1D + 1;
	printf("Block_Sum_Cen_1D %d\n",Block_Sum_Cen_1D);
	printf("Grid_Sum_Cen_1D %d\n",Grid_Sum_Cen_1D);
	printf("Sum_Cen_Itr %d\n",Sum_Cen_Itr);
	printf("---------------------------\n");
	//##########################
	// image kernel dimension 1D
	//##########################
	int Block_Size_Img_1D = 256;
	int Grid_Size_Img_1D = (Size_img-1)/Block_Size_Img_1D + 1;
	printf("Block_Size_Img_1D %d\n",Block_Size_Img_1D);
	printf("Grid_Size_Img_1D %d\n",Grid_Size_Img_1D);
	printf("---------------------------\n");
	//###########################
	// center kernel dimension 2D
	//###########################
	int Block_Len_Cen_2D = 16;
	int Grid_Len_Cen_2D = (Cen_W-1)/Block_Len_Cen_2D + 1;
	dim3 Grid_Cen_2D(Grid_Len_Cen_2D, Grid_Len_Cen_2D, 1);
	dim3 Block_Cen_2D(Block_Len_Cen_2D, Block_Len_Cen_2D, 1);
	printf("Block_Len_Cen_2D %d %d\n",Block_Len_Cen_2D, Block_Len_Cen_2D*Block_Len_Cen_2D);
	printf("Grid_Len_Cen_2D %d %d\n",Grid_Len_Cen_2D, Grid_Len_Cen_2D*Grid_Len_Cen_2D);
	printf("---------------------------\n");
	//#################################################
	// Search regions for each center
	// center advertise kernel dimension 2D
	//#################################################
	int Search_Size_2D = (2 * Cen_Len) * (2 * Cen_Len);
	int Block_Len_Search_2D = 2 * Cen_Len;
	int Block_Size_Search_2D = Block_Len_Search_2D * Block_Len_Search_2D;
	if (Block_Size_Search_2D > 1024){
		Block_Size_Search_2D = 1024;
		Block_Len_Search_2D = 32;
	}
	int Grid_Len_Search_2D = (Search_Size_2D-1)/Block_Size_Search_2D + 1;
	printf("Search_Size_2D %d\n Grid_Len_Search_2D %d\n Block_Len_Search_2D %d\n",	Search_Size_2D,
										Grid_Len_Search_2D,
										Block_Len_Search_2D);
	dim3 DimGrid_Search_2D( Grid_Len_Search_2D,
							Grid_Len_Search_2D,
							1);
	dim3 DimBlock_Search_2D( Block_Len_Search_2D,
							 Block_Len_Search_2D,
							 1);
	printf("---------------------------\n");
	// init center position and color space kernel
	Init_SLIC_Center_D<<<Grid_Cen_2D,Block_Cen_2D>>>(D_img, D_Center, Cen_Len, W, H, Cen_W);
	printf(">>> Center initialized...\n");
	// init label kernel
	Init_Label_D<<<Grid_Size_Img_1D, Block_Size_Img_1D>>>(D_Label, Size_img);
	printf(">>> Label initialized...\n");
	// reset sum of center info
	Reset_Sum_Center_D<<<Grid_Sum_Cen_1D,Block_Sum_Cen_1D>>>( D_Sum_Cen, D_Size_Cen, K);
	printf(">>> Sum Center reseted...\n");
	// init m s kernel
	Init_Max_Vector_D<<<Grid_Size_Cen_1D, Block_Size_Cen_1D>>>(M,S,10,Cen_Len*Cen_Len,K);
	printf(">>> m s initialized...\n");
	// Iterate through the SLIC segmentation process
	int Itr = 0;
	while(Itr < N_Iteration){
		printf("Itr %d\n",Itr);
		Itr++;
		// Reset Distance Vector for each pixel
		Reset_Distvec_D<<<Grid_Size_Img_1D, Block_Size_Img_1D>>>(D_Distvec, Size_img);
		CUDA_CHECK_RETURN(cudaDeviceSynchronize());	// Wait for the GPU launched work to complete
		CUDA_CHECK_RETURN(cudaGetLastError());
		//----------------------------------------------------------------------------------------------------------------------------------------
		// Each center searches within 2S * 2S region to associate itself to the pixels inside the search region
		//----------------------------------------------------------------------------------------------------------------------------------------
		for (int k=0; k < K; k+=2){

			// each center seaches within 2S * 2S region to advertise itself to pixels in the region
			Advertise_Label_D<<<DimGrid_Search_2D, DimBlock_Search_2D>>>(D_img, k, D_Center, D_rgb, D_xy, M, S, D_Distvec, D_Label, W, H, Cen_Len);
			// next center

			//CUDA_CHECK_RETURN(cudaDeviceSynchronize());	// Wait for the GPU launched work to complete
			//CUDA_CHECK_RETURN(cudaGetLastError());
		}
		//*/ loop through all the centers
		for (int k=1; k < K; k+=2){

			// each center seaches within 2S * 2S region to advertise itself to pixels in the region
			Advertise_Label_D<<<DimGrid_Search_2D, DimBlock_Search_2D>>>(D_img, k, D_Center, D_rgb, D_xy, M, S, D_Distvec, D_Label, W, H, Cen_Len);
			// next center

			//CUDA_CHECK_RETURN(cudaDeviceSynchronize());	// Wait for the GPU launched work to complete
			//CUDA_CHECK_RETURN(cudaGetLastError());

		}

		CUDA_CHECK_RETURN(cudaDeviceSynchronize());	// Wait for the GPU launched work to complete
		CUDA_CHECK_RETURN(cudaGetLastError());
		//*/
		//-----------------------------------------------------------------------------------------------------------------------------
		// Sum up and update the center vectors
		//-----------------------------------------------------------------------------------------------------------------------------
		Sum_Center_Info<<<Grid_Sum_Cen_1D, Block_Sum_Cen_1D>>>(D_img, D_Label, D_Sum_Cen, D_Size_Cen, Size_img, W, 0, Block_Sum_Cen_1D);
		// if number of center is more than one kernel can handle, sum up the rest of the centers chunk by chunk
		int Block_Size_Remaining_Cen_1D;
		int Grid_Size_Remaining_Cen_1D;
		for (int i=1;i<Sum_Cen_Itr;i++){
			// calculate the chunk size of this portion of center to be sumed up
			Block_Size_Remaining_Cen_1D = K - i * Block_Sum_Cen_1D;
			if (Block_Size_Remaining_Cen_1D > MAX_SHARED_MEM){
				Block_Size_Remaining_Cen_1D = MAX_SHARED_MEM;
			}
			Grid_Size_Remaining_Cen_1D = (Size_img-1)/Block_Size_Remaining_Cen_1D + 1;
			Sum_Center_Info<<<Grid_Size_Remaining_Cen_1D,Block_Size_Remaining_Cen_1D>>>(D_img, D_Label, D_Sum_Cen, D_Size_Cen, Size_img, W, i * Block_Sum_Cen_1D, Block_Size_Remaining_Cen_1D);
			//CUDA_CHECK_RETURN(cudaDeviceSynchronize());	// Wait for the GPU launched work to complete
			//CUDA_CHECK_RETURN(cudaGetLastError());
		}
		CUDA_CHECK_RETURN(cudaDeviceSynchronize());	// Wait for the GPU launched work to complete
		CUDA_CHECK_RETURN(cudaGetLastError());
		// update the centers by normalize with the size of each center region
		Update_Center<<<Grid_Size_Cen_1D,Block_Size_Cen_1D>>>(D_Center, D_Sum_Cen, D_Size_Cen, K);
		//----------------------------------------------------------------------------------------
		// Update the M and S variables to be the maximum for each center
		//----------------------------------------------------------------------------------------
		int* H_Label = (int*)malloc(Size_img * sizeof(int));
		CUDA_CHECK_RETURN(cudaMemcpy(H_Label,D_Label,Size_img*sizeof(int),cudaMemcpyDeviceToHost));
		CUDA_CHECK_RETURN(cudaMemcpy(H_rgb,D_rgb,Size_img*sizeof(double),cudaMemcpyDeviceToHost));
		CUDA_CHECK_RETURN(cudaMemcpy(H_xy,D_xy,Size_img*sizeof(double),cudaMemcpyDeviceToHost));
		CUDA_CHECK_RETURN(cudaDeviceSynchronize());	// Wait for the GPU launched work to complete
		CUDA_CHECK_RETURN(cudaGetLastError());
		// update M and S for each center to be the maximum
		for ( int i=0; i < Size_img; i++ ){
			if (H_M[H_Label[i]] < H_rgb[i]){
				H_M[H_Label[i]] = H_rgb[i];
			}
			if (H_S[H_Label[i]] < H_xy[i]){
				H_S[H_Label[i]] = H_xy[i];
			}
		}
		// feed the new M and S variable back to global memory
		CUDA_CHECK_RETURN(cudaMemcpy(M,H_M,K*sizeof(double),cudaMemcpyHostToDevice));
		CUDA_CHECK_RETURN(cudaMemcpy(S,H_S,K*sizeof(double),cudaMemcpyHostToDevice));

	}
	//---------------------------------------------------------------------------------------
	// Copy labels back to host and Draw Contour based on the label
	//---------------------------------------------------------------------------------------
	CUDA_CHECK_RETURN(cudaMemcpy(Label,D_Label,Size_img*sizeof(int),cudaMemcpyDeviceToHost));
	// draw contour
	Draw_Contour_D(img, Label, Itr);

	//------------------------------------------
	// Clean up global memory
	//------------------------------------------
	CUDA_CHECK_RETURN(cudaFree(D_img));
	CUDA_CHECK_RETURN(cudaFree(D_Center));
	CUDA_CHECK_RETURN(cudaFree(D_Sum_Cen));
	CUDA_CHECK_RETURN(cudaFree(D_Size_Cen));
	CUDA_CHECK_RETURN(cudaFree(M));
	CUDA_CHECK_RETURN(cudaFree(S));
	CUDA_CHECK_RETURN(cudaFree(D_rgb));
	CUDA_CHECK_RETURN(cudaFree(D_xy));
	CUDA_CHECK_RETURN(cudaFree(D_Distvec));
	CUDA_CHECK_RETURN(cudaFree(D_Label));

}















