#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <stdio.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv/cv.h>
using namespace cv;
using namespace std;
int sigma = 1;
int tam = 5 * sigma;
float umbral = 35;
Mat vec_mascara_der;
Mat vec_mascara;
Mat gx, gy, mag, ori, contornos;
Mat votacion, orientacion;
Mat puntoVx;
Mat puntoVy;

/**
 * Funcion que calcula el valor de gaussiana correspodiente al valor x
 */
float gaussiana(int x) {
	return (1 / (sigma * sqrt(2 * M_PI))) * (exp(-0.5 * (pow((x) / sigma, 2))));//* (0.39 / 0.1621);
}

/**
 * Funcion que calcula la derivada de la gaussiana correspondiente al valor x
 */
float derivGaussiana(int x) {
	return ((-x / pow(sigma, 2)) * (exp(-pow(x, 2) / 2 * pow(sigma, 2))));
}

/**
 * Metodo que calcula los vectores máscara que se utilizan para calcular el gradiente
 */
void mascara() {
	int aux = (tam / 2);
	int j = 0;
	//Se calcula la gaussiana y su derivada para un vector del tamaño elegido
	for (int i = aux; i >= -aux; i--) {
		float gaus = gaussiana(i);
		float der = derivGaussiana(i);
		vec_mascara_der.at<float>(0, j) = der;
		vec_mascara.at<float>(j, 0) = gaus;
		j++;
	}
}

/**
 * Metodo que aplica el operador de canny a la imagen
 */
void canny(Mat image) {
	/*
	 * Matrices donde se guardan los resultados
	 */
	vec_mascara_der = Mat::zeros(1, tam, CV_32F);
	vec_mascara = Mat::zeros(tam, 1, CV_32F);
	gx = Mat::zeros(image.rows, image.cols, image.type());
	gy = Mat::zeros(image.rows, image.cols, image.type());
	mag = Mat::zeros(image.rows, image.cols, CV_32F);
	ori = Mat::zeros(image.rows, image.cols, CV_32F);
	Mat GX = Mat::zeros(gx.rows, gx.cols, CV_8U);
	Mat GY = Mat::zeros(gy.rows, gy.cols, CV_8U);
	Mat MAG = Mat::zeros(mag.rows, mag.cols, CV_8U);
	Mat ORI = Mat::zeros(ori.rows, ori.cols, CV_8U);
	// calculo de las mascaras
	mascara();
	// calculo del gradiente x e y
	sepFilter2D(image, gx, CV_16S, vec_mascara_der, vec_mascara);
	sepFilter2D(image, gy, CV_16S, vec_mascara.t(), -vec_mascara_der.t());
	/*
	 * Para cada pixel se calcula el modulo y la orientacion del gradiente
	 * Ademas se hacen ajustes para poder mostrarlos por pantalla en imagen
	 */
	for (int i = 0; i < GX.rows; i++) {
		for (int j = 0; j < GX.cols; j++) {
			GX.at<uchar>(i, j) = gx.at<short>(i, j) / 2 + 128;
			GY.at<uchar>(i, j) = gy.at<short>(i, j) / 2 + 128;
			mag.at<float>(i, j) = sqrt(
					pow(gx.at<short>(i, j), 2) + pow(gy.at<short>(i, j), 2));
			MAG.at<uchar>(i, j) = sqrt(
					pow(gx.at<short>(i, j), 2) + pow(gy.at<short>(i, j), 2));
			float thetaMostrar = (atan2(gy.at<short>(i, j), gx.at<short>(i, j)));
			//-PI-PI to 0-PI
			if (thetaMostrar < 0) {
				thetaMostrar = CV_PI + (CV_PI + thetaMostrar);
			}
			ORI.at<uchar>(i, j) = (thetaMostrar / M_PI) * 128;
			ori.at<float>(i, j) =
					(atan2(gy.at<short>(i, j), gx.at<short>(i, j)));
		}
	}
	imshow("Cannyx", GX);
	imshow("Cannyy", GY);
	imshow("OriCanny", ORI);
	imshow("MagCanny", MAG);
}

/**
 * Metodo que aplica el operador de sobel a la imagen
 */
void sobel(Mat img) {
	/*
	 * Variables necesarias
	 */
	Mat blurred=img.clone();
	Mat sobelx = Mat::zeros(img.rows, img.cols, img.type());
	Mat sobely = Mat::zeros(img.rows, img.cols, img.type());
	Mat GX = Mat::zeros(img.rows, img.cols, CV_8U);
	Mat GY = Mat::zeros(img.rows, img.cols, CV_8U);
	Mat mag = Mat::zeros(img.rows, img.cols, CV_32F);
	Mat ori = Mat::zeros(img.rows, img.cols, CV_32F);
	Mat MAG = Mat::zeros(mag.rows, mag.cols, CV_8U);
	Mat ORI = Mat::zeros(ori.rows, ori.cols, CV_8U);
	// filtro gaussiana para reducir ruido
	for (int i = 1; i <= 5; i = i + 2) {
		GaussianBlur(blurred, blurred, Size(i, i), sigma, sigma);
	}

	// calculo del gradiente x e y
	Sobel(blurred, sobelx, CV_16S, 1, 0, 3);
	Sobel(blurred, sobely, CV_16S, 0, 1, 3);
	/*
	 * Para cada pixel se calculo el modulo y la orientacion del gradiente
	 * Se aplican unos ajustes para poder mostrar los resultados en imagen
	 */
	for (int i = 0; i < GX.rows; i++) {
		for (int j = 0; j < GX.cols; j++) {
			GX.at<uchar>(i, j) = sobelx.at<short>(i, j) / 2 + 128;
			GY.at<uchar>(i, j) = sobely.at<short>(i, j) / 2 + 128;
			mag.at<float>(i, j) = sqrt(
					pow(sobelx.at<short>(i, j), 2)
							+ pow(sobely.at<short>(i, j), 2));
			MAG.at<uchar>(i, j) = sqrt(
					pow(sobelx.at<short>(i, j), 2)
							+ pow(sobely.at<short>(i, j), 2));
			float thetaMostrar = ((float) atan2(sobely.at<short>(i, j),
					sobelx.at<short>(i, j)));
			//-PI-PI to 0-PI
			if (thetaMostrar < 0) {
				thetaMostrar = CV_PI + (CV_PI + thetaMostrar);
			}
			ORI.at<uchar>(i, j) = (thetaMostrar / M_PI) * 128;
			ori.at<float>(i, j) = ((float) atan2(sobely.at<short>(i, j),
					sobelx.at<short>(i, j)));
		}
	}
	imshow("Sobelx", GX);
	imshow("Sobely", GY);
	imshow("OriSobel", ORI);
	imshow("MagSobel", MAG);
}

/**
 * Metodo que realiza la interseccion de la recta dada por theta y rho con la linea del horizonte
 */
void votar_recta(float theta, float rho, Mat image) {
	int voto = (rho - (0 * sin(theta))) / cos(theta);
	voto += image.cols / 2;
	if (voto >= 0 && voto <= image.cols) {
		votacion.at<short>(voto, 0) += 1;
	}
}

int main(int argc, char **argv) {

	if (argc != 3) {
		cout << " Usage: display_image ImageToLoadAndDisplay" << endl;
		return -1;
	}

	Mat image, img;

	image = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);   // Read the file
	img = imread(argv[1]);
	if (!image.data)                              // Check for invalid input
	{
		cout << "Could not open or find the image" << std::endl;
		return -1;
	}
	sigma = atoi(argv[2]);
	imshow("image", image);                // Show our image inside it.
	// operador de sobel
	Mat imgSobel = image.clone();
	sobel(imgSobel);
	// operador de canny
	Mat imgCanny = image.clone();
	canny(imgCanny);
	// variables para la votacion
	votacion = Mat::zeros(image.cols, 1, CV_16S);
	orientacion = Mat::zeros(image.cols, 1, CV_16S);
	puntoVx = Mat::zeros(image.rows, image.cols, CV_16U);
	puntoVy = Mat::zeros(image.rows, image.cols, CV_16U);
	//
	int x;
	int y;
	float theta;
	float rho;
	int ncols = image.cols;
	int nrows = image.rows;
	/*
	 * Algoritmo con el que cada pixel vota a un punto de la recta del horizonte
	 */
	for (int i = 0; i < image.rows; i++) {
		for (int j = 0; j < image.cols; j++) {
			if (mag.at<float>(i, j) >= umbral) {
				x = j - (ncols / 2);
				y = (nrows / 2) - i;
				theta = ori.at<float>(i, j);
				rho = (x * cos(theta)) + (y * sin(theta));
				/*
				 * Filtro de rectas horizontales y verticales
				 */
				if (!(theta <= CV_PI + 0.10 && theta >= CV_PI - 0.10)
						&& !(theta <= 0 + 0.10 && theta >= 0 - 0.10)
						&& !(theta <= -CV_PI + 0.10 && theta >= -CV_PI - 0.10)
						&& !(theta <= CV_PI / 2 + 0.10
								&& theta >= CV_PI / 2 - 0.10)
						&& !(theta <= 3 * CV_PI / 2 + 0.10
								&& theta >= 3 * CV_PI / 2 - 0.10)) {

					//dibujar recta

					votar_recta(theta, rho, image);
				}
			}
		}
	}
	/*
	 * El punto que recibe mas puntos es el punto de fuga
	 */
	double min1, max1;
	Point minLugar, maxLugar;
	minMaxLoc(votacion, &min1, &max1, &minLugar, &maxLugar);
	cout << maxLugar.y << endl;
	line(img, Point(maxLugar.y - 10, 256 - 10),
			Point(maxLugar.y + 10, 256 + 10), Scalar(0, 0, 255), 1);
	line(img, Point(maxLugar.y - 10, 256 + 10),
			Point(maxLugar.y + 10, 256 - 10), Scalar(0, 0, 255), 1);
	circle(img, cvPoint(maxLugar.y, 256), 2, CV_RGB(255, 0, 0), -1, 8, 0);
	imshow("img", img);
	waitKey(0);                            // Wait for a keystroke in the window

}
