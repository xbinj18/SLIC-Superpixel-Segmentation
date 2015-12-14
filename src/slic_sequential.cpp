
#include "slic_sequential.h"
#include <time.h>
#include <assert.h>

#include <limits.h>

#define max(a,b) \
   ({ __typeof__ (a) _a = (a); \
       __typeof__ (b) _b = (b); \
     _a > _b ? _a : _b; })

#define min(a,b) \
   ({ __typeof__ (a) _a = (a); \
       __typeof__ (b) _b = (b); \
     _a < _b ? _a : _b; })



void Init_Distance(double* D_distvec, unsigned int img_size){

	for(int i=0; i < img_size; i++){
		D_distvec[i] = -1;
	}
}

void Init_Dist_Var(double* M, double* S, int m, int Cen_size, int num_Cen){

	for (int i=0; i < num_Cen; i++){
		M[i] = m;
		S[i] = Cen_size;
	}
}

int Init_SLIC_center(
								double* Cen_R,
								double* Cen_G,
								double* Cen_B,
								double* Cen_X,
								double* Cen_Y,
								int Cen_Len,
								PPM_Image* img
){


	int center_x, center_y, center_idx, x, y;
	int center_count = 0;
	int offset = Cen_Len / 2;

	int W = img->W;
	int H = img->H;

	for (y=0; y < H ; y++){
		center_y =  y * Cen_Len + offset;
		if (center_y >= H){
			break;
		}

		for (x=0; x < W; x++){
			center_x =  x * Cen_Len + offset;
			if (center_x >= W){
						break;
			}

			center_idx = center_x + center_y * W;

			Cen_R[center_count] = (double)img->data[center_idx].R;
			Cen_G[center_count] = (double)img->data[center_idx].G;
			Cen_B[center_count] = (double)img->data[center_idx].B;
			Cen_X[center_count] = (double)center_x;
			Cen_Y[center_count] = (double)center_y;

			center_count += 1;
		}
	}

	return center_count;

}


void Draw_contour(PPM_Image* img, int* Label){

	printf("Draw_contour..\n");

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
//*/

}

void SLIC_Sequential(PPM_Image* img, unsigned int K, unsigned int N_Iteration){

	printf("SLIC_seq\n");

	unsigned int Size_img = img->Size_img;
	int W = img->W;
	int H = img->H;
	// local copy of img
	PPM_Image local_img;
	local_img.W = W;
	local_img.H = H;
	local_img.Size_img = Size_img;
	PPM_Pixel* local_data = (PPM_Pixel*)malloc(Size_img * sizeof(PPM_Pixel));
	local_img.data = local_data;
	Reset_PPM_local(img ,&local_img);

	// SLIC_pixel center info
	double Cen_Len_d = sqrt((double)Size_img / (double)K);
	int Cen_Len = (unsigned int)Cen_Len_d;
	int offset = Cen_Len;
	//
	printf("    Cen_Len : %d\nK : %d",Cen_Len, K);
	// center pixel
	double* Cen_R = (double*)malloc(K * sizeof(double));
	double* Cen_G = (double*)malloc(K * sizeof(double));
	double* Cen_B = (double*)malloc(K * sizeof(double));
	double* Cen_X = (double*)malloc(K * sizeof(double));
	double* Cen_Y = (double*)malloc(K * sizeof(double));
	// initialize centers
	int num_Cen = Init_SLIC_center(Cen_R, Cen_G, Cen_B, Cen_X, Cen_Y, Cen_Len, img);
	// center sum
	double* sum_r = (double*)malloc(num_Cen * sizeof(double));
	double* sum_g = (double*)malloc(num_Cen * sizeof(double));
	double* sum_b = (double*)malloc(num_Cen * sizeof(double));
	double* sum_x = (double*)malloc(num_Cen * sizeof(double));
	double* sum_y = (double*)malloc(num_Cen * sizeof(double));
	double* size_pixel = (double*)malloc(num_Cen * sizeof(double));
	printf("    num_Cen : %d\n",num_Cen );

	// Perturb Center

	// distance vectors for each pixel
	double* D_rgb = (double*)malloc(Size_img * sizeof(double));
	double* D_xy = (double*)malloc(Size_img * sizeof(double));
	double* D_distvec = (double*)malloc(Size_img * sizeof(double));
	// labels for each pixel
	int* L_center = (int*)malloc(Size_img * sizeof(int));
	// initialize each pixel
	for(int i=0;i<Size_img;i++){
		L_center[i] = -1;
	}
	// set value for m and s for each center
	double* M = (double*)malloc(num_Cen * sizeof(double));
	double* S = (double*)malloc(num_Cen * sizeof(double));
	Init_Dist_Var(M,S, 10, Cen_Len * Cen_Len, num_Cen);

	// perform segmentation for N_Iteration times of iteration
	int Itr = 0;

	//debug
	while (Itr < N_Iteration){

		printf(" Itr : %d\n",Itr);
		Itr++;
		Init_Distance(D_distvec, Size_img);

		// loop through the centers
		for (int k=0; k < num_Cen; k++){

			double cen_r = Cen_R[k];
			double cen_g = Cen_G[k];
			double cen_b = Cen_B[k];
			double cen_x = Cen_X[k];
			double cen_y = Cen_Y[k];

			// define search area around this center
			int x_L = max(0, cen_x - offset);
			int y_U = max(0, cen_y - offset);
			int x_R = min(W, cen_x + offset);
			int y_D = min(H, cen_y + offset);

			// loop through the area
			for (int y = y_U; y < y_D; y++){
				for (int x = x_L; x < x_R; x++){
					// check boundary
					assert(0 <= x && x < W && 0 <= y && y < H);

					int i = x + y * W;

					double r = (double)img->data[i].R;
					double g = (double)img->data[i].G;
					double b = (double)img->data[i].B;

					D_rgb[i] = 	(r - cen_r) * (r - cen_r) +
								(g - cen_g) * (g - cen_g) +
								(b - cen_b) * (b - cen_b);

					D_xy[i] = 	(x - cen_x) * (x - cen_x) +
								(y - cen_y) * (y - cen_y);

					double D_distance = D_rgb[i] / M[k] + D_xy[i] / S[k];
					//double D_distance = D_xy[i] / S[k];
					//double D_distance = D_rgb[i] / M[k];

					if (D_distance < D_distvec[i] || D_distvec[i] == -1){
						// assign new center to this pixel
						D_distvec[i] = D_distance;
						L_center[i] = k;
					}
				}
			}
			// next center

		}
		// update M and S for each center to be the maximum
		for ( int i=0; i < Size_img; i++ ){
			if (M[L_center[i]] < D_rgb[i]){
				M[L_center[i]] = D_rgb[i];
			}
			if (S[L_center[i]] < D_xy[i]){
				S[L_center[i]] = D_xy[i];
			}
		}
		// update center
		// reset info for each center
		for (int i=0; i< num_Cen; i++){
			sum_r[i] = 0;
			sum_g[i] = 0;
			sum_b[i] = 0;
			sum_x[i] = 0;
			sum_y[i] = 0;
			size_pixel[i] = 0;
		}
		// sum up the vectors of each pixel to their centers
		for (int i=0; i < Size_img; i++){
			int L_idx = L_center[i];
			sum_r[L_idx] += (double)img->data[i].R;
			sum_g[L_idx] += (double)img->data[i].G;
			sum_b[L_idx] += (double)img->data[i].B;
			sum_x[L_idx] += (double)(i % W);
			sum_y[L_idx] += (double)(i / W);
			size_pixel[L_idx] ++;
		}
		// update the position and color of the new center
		for (int k=0; k < num_Cen; k++){
			double inverse_size_pixel = 1 / size_pixel[k];
			Cen_R[k] = sum_r[k] * inverse_size_pixel;
			Cen_G[k] = sum_g[k] * inverse_size_pixel;
			Cen_B[k] = sum_b[k] * inverse_size_pixel;
			Cen_X[k] = sum_x[k] * inverse_size_pixel;
			Cen_Y[k] = sum_y[k] * inverse_size_pixel;
		}
		// next iteration
		char file_version[] = "new_lena_--.ppm";
		//file_version[15] = '\0';
		file_version[9] = Itr / 10 + '0';
		file_version[10] = Itr % 10 + '0';

		Draw_contour(&local_img, L_center);
		write_PPM_img(&local_img,file_version);
		Reset_PPM_local(img,&local_img);

	}

	// clean up
	free(D_rgb);
	free(D_xy);
	free(D_distvec);

	free(Cen_R);
	free(Cen_G);
	free(Cen_B);
	free(Cen_X);
	free(Cen_Y);

	free(sum_r);
	free(sum_g);
	free(sum_b);
	free(sum_x);
	free(sum_y);
	free(size_pixel);

	Draw_contour(img, L_center);

	free(L_center);

}
