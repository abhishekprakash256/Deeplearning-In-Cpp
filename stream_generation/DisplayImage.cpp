//This file generates data stream
//To change number of samples, change the variable num_of_samples
//Import datasets from different datasets, this code is generating streams from Fashion-MNIST

#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <sys/time.h>
#include <iostream>
#include <math.h> 
#include <stdlib.h>
using namespace std;
#include <vector>
#include <fstream>
#include <string>
#include <emmintrin.h>
#include <cstdio>
#include <cstring>

using namespace cv;


double X[60000][784];
double y[60000];
int row, cols;
int read_limit;






int ReverseInt (int i)
{
    unsigned char ch1, ch2, ch3, ch4;
    ch1=i&255;
    ch2=(i>>8)&255;
    ch3=(i>>16)&255;
    ch4=(i>>24)&255;
    return((int)ch1<<24)+((int)ch2<<16)+((int)ch3<<8)+ch4;
}

void Read_dataset(int NumberOfImages, int DataOfAnImage,double arr[][784])
{

    ifstream file ("./train-images-idx3-ubyte",ios::binary);
    if (file.is_open())
    {
        int magic_number=0;
        int number_of_images=0;
        int n_rows=0;
        int n_cols=0;
        file.read((char*)&magic_number,sizeof(magic_number));
        magic_number= ReverseInt(magic_number);
        file.read((char*)&number_of_images,sizeof(number_of_images));
        number_of_images= ReverseInt(number_of_images);
        file.read((char*)&n_rows,sizeof(n_rows));
        n_rows= ReverseInt(n_rows);
        file.read((char*)&n_cols,sizeof(n_cols));
        n_cols= ReverseInt(n_cols);
        for(int i=0;i<number_of_images;++i)
        {
            for(int r=0;r<n_rows;++r)
            {
                for(int c=0;c<n_cols;++c)
                {
                    unsigned char temp=0;
                    file.read((char*)&temp,sizeof(temp));
                    arr[i][(n_rows*r)+c]= (double)temp;
                }
            }
        }
    }
}


int main(int argc, char** argv)
{


    Read_dataset(read_limit,row*cols, X);
    ofstream MyFile("train-idx3-ubyte", ofstream::binary);
    int k;
    int i;
    int num_of_samples;
    num_of_samples = 5;
    int offset = 17;

    read_limit = 10000;
    row = 28;
    cols = 28;
    

    /*generate ubyte dataset consisting of streams from: https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwj2iZOy-NT7AhUKSDABHQxCCaYQFnoECBUQAQ&url=http%3A%2F%2Fyann.lecun.com%2Fexdb%2Fmnist%2F&usg=AOvVaw2l4Jn0H3ZvSJ336fACilwX
    This data type has some specific format and offsets and other necessary variables are defined below
    */


    MyFile << "0000 32 bit integer 0x00000803(2051) 000x0D2\n";
    MyFile << "0004 32 bit integer 10000            number_of_images\n";
    MyFile << "0008 32 bit integer 28               number of rows\n";
    MyFile << "0012 32 bit integer 28               number of columns\n";
                                                    
    
    for (k = 0; k<num_of_samples; k++)
    {
        std::string name = "training_";
        std::string data = std::to_string(offset);
        double image[row][cols];
        int i,j;


        for ( i = 0; i<row; i++)
            for ( j = 0; j<cols; j++)
                // cout << i*28+j << endl;
            {
                data = data + " unsigned byte ";
                data = data + std::to_string(X[k][i*row+j]);
                data = data + "              pixel\n";
                image[i][j] = X[k][i*row+j];
                MyFile << data;
            }
                

        cv::Mat A(row,cols,CV_64F);
        std::memcpy(A.data, image, row*cols*sizeof(double));

        std::string id = std::to_string(k);
        name = name + id;
        name = name+".jpg";
        // cout << name << endl;
        imwrite(name, A);
        MyFile.close();


    }
    

return 0;
}
