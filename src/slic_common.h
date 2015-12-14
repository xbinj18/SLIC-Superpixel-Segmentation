/*
 * slic_common.h
 *
 *  Created on: Dec 10, 2015
 *      Author: xbin
 */

#ifndef SLIC_COMMON_H_
#define SLIC_COMMON_H_



#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_HEADER_LEN 20
#define PPM_RGB_DEPTH 255

/*
typedef struct {

	double R;
	double G;
	double B;

	double X;
	double Y;

}SLIC_Pixel;
*/

typedef struct {

	float R;
	float G;
	float B;

	float X;
	float Y;


}SLIC_Pixel;


typedef struct{

	unsigned char R;
	unsigned char G;
	unsigned char B;

}PPM_Pixel;

typedef struct{

	int W;
	int H;
	int Size_img;

	PPM_Pixel* data;

}PPM_Image;



PPM_Image* read_PPM_img(char* filename);
void write_PPM_img(PPM_Image* outImage, char* filename);


void print_PPM_label(int* label, PPM_Image* info, int version);

void create_PPM_img(int W, int H, int S);

void print_label(int* L_center, PPM_Image* img);
void print_PPM_select(
						int* selected,
						PPM_Image* info,
						int version
);

void print_center(
					double* Cen_X,
					double* Cen_Y,
					int num_Cen
);
void print_PPM_center(
						double* Cen_X,
						double* Cen_Y,
						int Cen_Len,
						PPM_Image* info,
						int version
);

void Reset_PPM_local(PPM_Image* info, PPM_Image* local_img);
void print_img_info(PPM_Image* img_info);



#endif /* SLIC_COMMON_H_ */
