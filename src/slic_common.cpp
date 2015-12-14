/*
 * slic_common.cpp
 *
 *  Created on: Dec 10, 2015
 *      Author: xbin
 */

#include "slic_common.h"




#include "slic_common.h"
#include <assert.h>
#include <math.h>

#define MAX_DIGIT 3


PPM_Image* read_PPM_img(char* filename){


	FILE* fileP = fopen(filename,"rb");
	char header_buff[MAX_HEADER_LEN];
	int c,rgb_depth;
	PPM_Image* img;

	// open image file
	if (!fileP){
		printf("Failed to open file\n");
		exit(1);
	}
	// read format header
	if (!fgets(header_buff,sizeof(header_buff),fileP)){
		printf("Failed to read header\n");
		exit(1);
	}
	// check format
	if ( !(header_buff[0] == 'P' && header_buff[1] == '6') ){

		fprintf(stderr,"Unsupported image format, should be 'P6'\n");
		exit(1);
	}
	// fill in image info
	img = (PPM_Image *)malloc(sizeof(PPM_Image));
	if (!img){
		fprintf(stderr,"Failed to allocate memory for image\n");
		exit(1);
	}
	// skip commets
	c = getc(fileP);
	while ( c == '#'){
		while(getc(fileP) != '\n'){
			c = getc(fileP);
		}
	}
	ungetc(c, fileP);
	// read dimension of image
	if (fscanf(fileP,"%d %d\n",&img->W,&img->H) != 2){
		fprintf(stderr,"Failed to get image size info\n");
		exit(1);
	}
	img->Size_img = img->W * img->H;
	if (img->Size_img < 0){
		fprintf(stderr,"imge too large %d by %d\n",img->W,img->H);
		exit(1);
	}
	// read rgb depth
	if (fscanf(fileP,"%d\n",&rgb_depth) != 1){
		fprintf(stderr,"Failedd to get rgb depth\n");
		exit(1);
	}
	// check rgb depth
	if (rgb_depth != PPM_RGB_DEPTH){
		fprintf(stderr,"Unsupported RGB color depth, should be 255 colors\n");
		exit(1);
	}
	// allocate memory for image data
	img->data = (PPM_Pixel*)malloc( (img->Size_img * sizeof(PPM_Pixel)) );
	if (!img->data){
		fprintf(stderr,"Failed to allocate memory for image data\n");
		exit(1);
	}
	// read pixel data
	if (fread(img->data, 3, img->Size_img, fileP ) != img->Size_img){
		fprintf(stderr,"Failed to load image\n");
		exit(1);
	}

	return img;

}


void write_PPM_img(PPM_Image* outImage, char* filename){

	printf("write_PPM_img...");

	if (outImage == NULL){
		fprintf(stderr,"outImage is NULL\n");
		exit(1);
	}

	char path_filename[30] = "outputs/";
	strcat(path_filename,filename);
	path_filename[strlen(path_filename)] = '\0';

	FILE* fileP = fopen(path_filename,"wb");
	if (fileP == NULL){
		fprintf(stderr,"Failed to open |%s|\n",path_filename);
		exit(1);
	}

	fprintf(fileP,"P6\n");
	fprintf(fileP,"%u %u\n",outImage->W,outImage->H);
	fprintf(fileP,"255\n");

	printf("|%s| %d * %d\n",path_filename, outImage->W,outImage->H);

	if ( fwrite(outImage->data, 3, outImage->Size_img, fileP ) != outImage->Size_img){

		fprintf(stderr,"Failed to write output image\n");
		exit(1);
	}

	fclose(fileP);
}





// debugging

void write_PPM_RGB(PPM_Pixel* pixel, int R, int G, int B){

	pixel->R = R;
	pixel->G = G;
	pixel->B = B;

}

void print_center(
					double* Cen_X,
					double* Cen_Y,
					int num_Cen
){

	for (int k=0; k< num_Cen; k++){

		double x = Cen_X[k];
		double y = Cen_Y[k];


		printf(" center [%d] %f %f\n",k,x,y);
	}

}

int in_range(PPM_Image* info, int x, int y){


	//printf("_____ %d %d in range? ", x, y);
	if ( 0 <= x && x < info->W && 0 <= y && y < info->H){
	//	printf("yes\n");
		return 1;
	}

	else{
	//	printf("no\n");
		return 0;
	}

}

void print_magnify_pixel(PPM_Image* info, int idx, int size){

	//printf("magnify %d\n",idx);

	int x = idx % info->W;
	int y = idx / info->W;


	for (int y1 = y - size; y1 < y + size; y1++){
		for ( int x1 = x - size; x1 < x + size; x1++){
			if ( in_range(info, x1, y1)){

				int i = x1 + y1 * info->W;
				write_PPM_RGB(&info->data[i], 250, 250, 250);
			}
		}
	}
}


void print_PPM_label(int* label, PPM_Image* info, int version){



	for (int i =0; i< info->Size_img; i++){

		info->data[i].R = (125 * (label[i] + 1) ) % 255;
		info->data[i].G = (30 *  (label[i] + 1) ) % 255;
		info->data[i].B = (20 *  (label[i] + 1) ) % 255;

	}

		char filename[] = "label_---.ppm";

		filename[6] = version / 100 + '0';
		filename[7] = (version / 10) % 10 + '0';
		filename[8] = version % 10 + '0';

		write_PPM_img(info, filename);
}

void print_PPM_select(
						int* selected,
						PPM_Image* info,
						int version
){

	int Size_img = info->Size_img;

	PPM_Image outImage;
	outImage.W = info->W;
	outImage.H = info->H;
	outImage.Size_img = info->Size_img;

	PPM_Pixel* data = (PPM_Pixel*)malloc(Size_img * sizeof(PPM_Pixel));

	for (int i =0; i< Size_img; i++){

		double R = info->data[i].R;
		double G = info->data[i].G;
		double B = info->data[i].B;

		data[i].R = R;
		data[i].G = G;
		data[i].B = B;

		if (selected[i]){

			data[i].R = (int)(R *  0.5 ) % 255;
			data[i].G = (int)(G *  0.5 ) % 255;
			data[i].B = (int)(B *  0.5 ) % 255;
			selected[i] = 0;
		}
	}

	outImage.data = data;


	char filename[] = "select_--.ppm";
	filename[7] = version / 10 + '0';
	filename[8] = version%10 + '0';

	write_PPM_img(&outImage, filename);

	free(data);
}

void print_PPM_center(
						double* Cen_X,
						double* Cen_Y,
						int num_Cen,
						PPM_Image* info,
						int version
){

	int W = info->W;
	//int H = info->H;
	int Size_img = info->Size_img;

	for (int i=0; i<num_Cen; i++){

		int cen_x = (int)Cen_X[i];
		int cen_y = (int)Cen_Y[i];

		int idx = cen_x + cen_y * W;

		assert(idx < Size_img);

		info->data[idx].R = 250;
		info->data[idx].G = 250;
		info->data[idx].B = 120;

	}

	if (version == 0){

		char filename[] = "center_--.ppm";

		filename[7] = version / 10 + '0';
		filename[8] = version%10 + '0';

		write_PPM_img(info, filename);

	}


}

void Reset_PPM_local(PPM_Image* info, PPM_Image* local_img){

	for (int i=0; i<info->Size_img; i++){

		local_img->data[i].R = info->data[i].R;
		local_img->data[i].G = info->data[i].G;
		local_img->data[i].B = info->data[i].B;
	}

}



void print_img_info(PPM_Image* img_info){

	printf("img_info\n");
	printf("-----------------\n");
	printf(" width		%d\n",img_info->W);
	printf(" height 	%d\n",img_info->H);
	printf(" size 		%d\n",img_info->Size_img);
	printf(" First 3 pixel\n");

	int i;
	for(i=0;i<3;i++){
		printf("   r %d\n   g %d\n   b %d\n---\n",img_info->data[i].R,img_info->data[i].G,img_info->data[i].B);
	}
	printf(" Last 3 pixel\n");
	for(i=img_info->Size_img - 3; i<img_info->Size_img; i++){
		printf("   r %d\n   g %d\n   b %d\n---\n",img_info->data[i].R,img_info->data[i].G,img_info->data[i].B);
	}

	printf("-----------------\n");

}


void print_number( int N){

	int digit,i;
	for (digit = 1;digit < MAX_DIGIT; digit ++){
		if (N < pow(10,digit) ){
			for(i=MAX_DIGIT-digit;i>0;i--){
				printf(".");
			}
			printf("%d",N);
			return;
		}
	}
}

void print_label(int* L_center, PPM_Image* img){

	int img_size = img->Size_img;
	int W = img->W;

	int old_y = 0;

	for (int i=0;i<img_size;i++){

		int x = i % W;
		int y = i / W;

	if (y != old_y)printf("\n");

		old_y = y;

		//printf("[%d][%d]   |%d|\n",y,x,L_center[i]);
		print_number(L_center[i]);
		//print_number(i);
		//printf("%d",i);

	}

	printf("\n\n");

}

void create_PPM_img(int W, int H, int S){


	PPM_Image img;
	img.W = W;
	img.H = H;
	img.Size_img = W*H;

	img.data = (PPM_Pixel*)malloc(W*H*sizeof(PPM_Pixel));

	int c = 0;


	for (int ys = 0; ys < H; ys += S){
		for (int xs = 0; xs < W; xs+= S){

			c++;
			for (int y=ys;y<ys + S;y++){
				for (int x=xs;x<xs + S;x++){
					int i = x + y * W;

					img.data[i].R = (unsigned char)(((c % 2)* 80) % 255);
					img.data[i].G = (unsigned char)(((c % 2)*160) % 255);
					img.data[i].B = (unsigned char)(((c % 2)*240) % 255);
				}
			}
		}
	}

	printf("create\n");


	char filename[] = "input_--_--_--.ppm";
	filename[6] = W / 10;
	filename[7] = W % 10;
	filename[9] = H / 10;
	filename[10] = H % 10;
	filename[12] = S / 10;
	filename[13] = S % 10;

	write_PPM_img(&img,filename);

	free(img.data);


}











