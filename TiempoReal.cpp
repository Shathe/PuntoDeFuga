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
float umbral = 30;
Mat vec_mascara_der;
Mat vec_mascara;
Mat gx, gy, mag, ori, contornos;
Mat votacion, votacionRecta, orientacion;
Mat Rho;
Mat Theeta;
int rectasRegistradas = 0;
/**
 * Metodo que aplica el operador de sobel a la imagen
 */
void sobel(Mat img) {
	/*
	 * Variables necesarias
	 */
	Mat blurred;
	Mat sobelx = Mat::zeros(img.rows, img.cols, img.type());
	Mat sobely = Mat::zeros(img.rows, img.cols, img.type());
	Mat GX = Mat::zeros(img.rows, img.cols, CV_8U);
	Mat GY = Mat::zeros(img.rows, img.cols, CV_8U);
	Mat mag = Mat::zeros(img.rows, img.cols, CV_32F);
	Mat ori = Mat::zeros(img.rows, img.cols, CV_32F);
	Mat MAG = Mat::zeros(mag.rows, mag.cols, CV_8U);
	Mat ORI = Mat::zeros(ori.rows, ori.cols, CV_8U);
	// filtro gaussiana para reducir ruido
	GaussianBlur(img, blurred, Size(5, 5), sigma, sigma);
	// calculo del gradiente x e y
	Sobel(blurred, sobelx, CV_16S, 1, 0, 5);
	Sobel(blurred, sobely, CV_16S, 0, 1, 5);
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
	//imshow("Sobelx", sobelx);
	//imshow("Sobely", sobely);
	//imshow("OriSobel", ORI);
	//imshow("MagSobel", MAG);
}
// Finds the intersection of two lines, or returns false.
// The lines are defined by (o1, p1) and (o2, p2).
bool intersection(Point2f o1, Point2f p1, Point2f o2, Point2f p2, Point2f &r) {
	Point2f x = o2 - o1;
	Point2f d1 = p1 - o1;
	Point2f d2 = p2 - o2;

	float cross = d1.x * d2.y - d1.y * d2.x;
	if (abs(cross) < /*EPS*/1e-8)
		return false;

	double t1 = (x.x * d2.y - x.y * d2.x) / cross;
	r = o1 + d1 * t1;
	return true;
}
/**
 * Funcion que calcula el valor de gaussiana correspodiente al valor x
 */
float gaussiana(int x) {
	return (1 / (sigma * sqrt(2 * M_PI))) * (exp(-0.5 * (pow((x) / sigma, 2)))); //* (0.39 / 0.1621);
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
 * Metodo que realiza la interseccion de la recta dada por theta y rho con la linea del horizonte
 */
void votar_recta(float theta, float rho, Mat image) {
	int voto = (rho - (0 * sin(theta))) / cos(theta);
	voto += image.cols / 2;
	if (voto >= 0 && voto <= image.cols) {
		votacion.at<short>(voto, 0) += 1;
	}
}

/*
 * Se registra una recta
 */
void registrar_recta(float theta, float rho) {
	Rho.at<float>(rectasRegistradas, 0) = rho;
	Theeta.at<float>(rectasRegistradas, 0) = theta;
	rectasRegistradas++;
}
/*
 * Se gerena una interseccion entre 2 rectas aletatorias si su orientacion no es similar y
 * se guarda la interseccion como un voto si entra en la imagen
 */
void generar_votar_interseccion(Mat image, Mat img) {
	int recta1 = rand() % rectasRegistradas;
	int recta2 = rand() % rectasRegistradas;
	if (recta1 != recta2) {
		float rho1 = Rho.at<float>(recta1, 0);
		float theta1 = Theeta.at<float>(recta1, 0);
		float rho2 = Rho.at<float>(recta2, 0);
		float theta2 = Theeta.at<float>(recta2, 0);
		//miras si las oreientaciones no se parecen mucho
		float a = theta1;
		float b = theta2;
		if (b < 0)
			b += CV_PI;
		if (a < 0)
			b += CV_PI;
		if (abs(a - b) > 0.35) {
			//se hace la interseccion si su orientacion no se parece
			int y1 = 0;
			int x1 = (rho1 - (y1 * sin(theta1))) / cos(theta1);
			int y2 = 512;
			int x2 = (rho1 - (y2 * sin(theta1))) / cos(theta1);

			int y3 = 0;
			int x3 = (rho2 - (y3 * sin(theta2))) / cos(theta2);

			int y4 = 512;
			int x4 = (rho2 - (y4 * sin(theta2))) / cos(theta2);

			Point2f inters;
			if (intersection(Point2f(x1, y1), Point2f(x2, y2), Point2f(x3, y3),
					Point2f(x4, y4), inters)) {

				float i = inters.x + image.cols / 2;
				float j = (image.rows / 2) - inters.y;
				if (i >= 0 && i < image.cols && j >= 0 && j < image.rows) {

					votacionRecta.at<short>(i, j) += 1;
				}
			}

		}

	}

}
int main(int argc, char **argv) {

	Mat image, img;
	sigma = atoi(argv[2]);
	//VideoCapture cap(0); // open the default camera
	VideoCapture cap(1);
	if (cap.isOpened()) {
		// check if we succeeded

		while (1) {
			cap >> img;

			cvtColor(img, image, CV_BGR2GRAY);

			//imshow("image", image);                // Show our image inside it.
			vec_mascara_der = Mat::zeros(1, tam, CV_32F);
			vec_mascara = Mat::zeros(tam, 1, CV_32F);

			gx = Mat::zeros(image.rows, image.cols, image.type());
			gy = Mat::zeros(image.rows, image.cols, image.type());
			mag = Mat::zeros(image.rows, image.cols, CV_32F);
			ori = Mat::zeros(image.rows, image.cols, CV_32F);

			mascara();

			sepFilter2D(image, gx, CV_16S, vec_mascara_der, vec_mascara);
			sepFilter2D(image, gy, CV_16S, vec_mascara.t(),
					-vec_mascara_der.t());
			for (int i = 0; i < gx.rows; i++) {
				for (int j = 0; j < gx.cols; j++) {

					mag.at<float>(i, j) = sqrt(
							pow(gx.at<short>(i, j), 2)
									+ pow(gy.at<short>(i, j), 2));

					ori.at<float>(i, j) = (atan2(gy.at<short>(i, j),
							gx.at<short>(i, j)));
				}
			}
			//Votacion votacion  votacionRecta
			votacionRecta = Mat::zeros(image.cols, image.rows, CV_16S);
			Rho = Mat::zeros(image.rows * image.cols, 1, CV_32F);
			Theeta = Mat::zeros(image.rows * image.cols, 1, CV_32F);
			rectasRegistradas = 0;
			//
			int x;
			int y;
			float theta;
			float rho;
			int ncols = image.cols;
			int nrows = image.rows;

			for (int i = 0; i < image.rows; i++) {
				for (int j = 0; j < image.cols; j++) {
					if (mag.at<float>(i, j) >= umbral) {
						x = j - (ncols / 2);
						y = (nrows / 2) - i;
						theta = ori.at<float>(i, j);
						rho = (x * cos(theta)) + (y * sin(theta));

						if (!(theta <= CV_PI + 0.20 && theta >= CV_PI - 0.20)
								&& !(theta <= 0 + 0.20 && theta >= 0 - 0.20)
								&& !(theta <= -CV_PI + 0.20
										&& theta >= -CV_PI - 0.20)
								&& !(theta <= CV_PI / 2 + 0.20
										&& theta >= CV_PI / 2 - 0.20)
								&& !(theta <= 3 * CV_PI / 2 + 0.20
										&& theta >= 3 * CV_PI / 2 - 0.20)) {

							//dibujar recta

							registrar_recta(theta, rho);
						}
					}
				}
			}

			double min1, max1;
			Point minLugar, maxLugar;

			for (int i = 0; i < rectasRegistradas * log(rectasRegistradas);
					i++) {
				generar_votar_interseccion(image, img);
			}

			/*
			 * Se obtiene el punto mas intersectado
			 */

			minMaxLoc(votacionRecta, &min1, &max1, &minLugar, &maxLugar);
			line(img, Point(maxLugar.y - 10, maxLugar.x - 10),
					Point(maxLugar.y + 10, maxLugar.x + 10),
					Scalar(255, 0, 255), 1);
			line(img, Point(maxLugar.y - 10, maxLugar.x + 10),
					Point(maxLugar.y + 10, maxLugar.x - 10),
					Scalar(255, 0, 255), 1);

			imshow("img", img);
			if (waitKey(1) == 27)
				break; // stop capturing by pressing ESC
		}
	}

}
