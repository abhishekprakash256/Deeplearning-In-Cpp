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

using namespace cv;


double X[60000][784];
double y[60000];

//CIFAR
// template <template <typename...> class Container, typename Image, typename Label>
// struct CIFAR10_dataset {
//     Container<Image> training_images; ///< The training images
//     Container<Image> test_images;     ///< The test images
//     Container<Label> training_labels; ///< The training labels
//     Container<Label> test_labels;     ///< The test labels

//     /*!
//      * \brief Resize the training set to new_size
//      *
//      * If new_size is less than the current size, this function has no effect.
//      *
//      * \param new_size The size to resize the training sets to.
//      */
//     void resize_training(std::size_t new_size) {
//         if (training_images.size() > new_size) {
//             training_images.resize(new_size);
//             training_labels.resize(new_size);
//         }
//     }

//     /*!
//      * \brief Resize the test set to new_size
//      *
//      * If new_size is less than the current size, this function has no effect.
//      *
//      * \param new_size The size to resize the test sets to.
//      */
//     void resize_test(std::size_t new_size) {
//         if (test_images.size() > new_size) {
//             test_images.resize(new_size);
//             test_labels.resize(new_size);
//         }
//     }
// };

// /*!
//  * \brief Read a CIFAR 10 data file inside the given containers
//  * \param images The container to fill with the labels
//  * \param path The path to the label file
//  * \param limit The maximum number of elements to read (0: no limit)
//  */
// template <typename Images, typename Labels, typename Func>
// void read_cifar10_file(Images& images, Labels& labels, const std::string& path, std::size_t limit, Func func) {
//     if(limit && limit <= images.size()){
//         return;
//     }

//     std::ifstream file;
//     file.open(path, std::ios::in | std::ios::binary | std::ios::ate);

//     if (!file) {
//         std::cout << "Error opening file: " << path << std::endl;
//         return;
//     }

//     auto file_size = file.tellg();
//     std::unique_ptr<char[]> buffer(new char[file_size]);

//     //Read the entire file at once
//     file.seekg(0, std::ios::beg);
//     file.read(buffer.get(), file_size);
//     file.close();

//     std::size_t start = images.size();

//     size_t size = 10000;
//     size_t capacity = limit - images.size();

//     if(capacity > 0 && capacity < size){
//         size = capacity;
//     }

//     // Prepare the size for the new
//     images.reserve(images.size() + size);
//     labels.resize(labels.size() + size);

//     for(std::size_t i = 0; i < size; ++i){
//         labels[start + i] = buffer[i * 3073];

//         images.push_back(func());

//         for(std::size_t j = 1; j < 3073; ++j){
//             images[start + i][j - 1] = buffer[i * 3073 + j];
//         }
//     }
// }

// /*!
//  * \brief Read all test data.
//  *
//  * The dataset is assumed to be in a cifar-10 subfolder
//  *
//  * \param limit The maximum number of elements to read (0: no limit)
//  * \param func The functor to create the image objects.
//  */
// template <typename Images, typename Labels, typename Functor>
// void read_test(const std::string& folder, std::size_t limit, Images& images, Labels& labels, Functor func) {
//     read_cifar10_file(images, labels, folder + "/test_batch.bin", limit, func);
// }

// /*!
//  * \brief Read all training data
//  *
//  * The dataset is assumed to be in a cifar-10 subfolder
//  *
//  * \param limit The maximum number of elements to read (0: no limit)
//  * \param func The functor to create the image objects.
//  */
// template <typename Images, typename Labels, typename Functor>
// void read_training(const std::string& folder, std::size_t limit, Images& images, Labels& labels, Functor func) {
//     read_cifar10_file(images, labels, folder + "/data_batch_1.bin", limit, func);
//     read_cifar10_file(images, labels, folder + "/data_batch_2.bin", limit, func);
//     read_cifar10_file(images, labels, folder + "/data_batch_3.bin", limit, func);
//     read_cifar10_file(images, labels, folder + "/data_batch_4.bin", limit, func);
//     read_cifar10_file(images, labels, folder + "/data_batch_5.bin", limit, func);
// }

// /*!
//  * \brief Read all test data.
//  *
//  * The dataset is assumed to be in a cifar-10 subfolder
//  *
//  * \param limit The maximum number of elements to read (0: no limit)
//  * \param func The functor to create the image objects.
//  */
// template <typename Images, typename Labels, typename Functor>
// void read_test(std::size_t limit, Images& images, Labels& labels, Functor func) {
//     read_test("./cifar-10-batches-bin", limit, images, labels, func);
// }

// /*!
//  * \brief Read all training data
//  *
//  * The dataset is assumed to be in a cifar-10 subfolder
//  *
//  * \param limit The maximum number of elements to read (0: no limit)
//  * \param func The functor to create the image objects.
//  */
// template <typename Images, typename Labels, typename Functor>
// void read_training(std::size_t limit, Images& images, Labels& labels, Functor func) {
//     read_training("./cifar-10-batches-bin", limit, images, labels, func);
// }

// /*!
//  * \brief Read a CIFAR 10 data file inside the given containers
//  * \param images The container to fill with the labels
//  * \param path The path to the label file
//  * \param limit The maximum number of elements to read (0: no limit)
//  */
// template <typename Images, typename Labels>
// void read_cifar10_file_categorical(Images& images, Labels& labels, const std::string& path, std::size_t limit, size_t start) {
//     if(limit && limit <= start){
//         return;
//     }

//     std::ifstream file;
//     file.open(path, std::ios::in | std::ios::binary | std::ios::ate);

//     if (!file) {
//         std::cout << "Error opening file: " << path << std::endl;
//         return;
//     }

//     auto file_size = file.tellg();
//     std::unique_ptr<char[]> buffer(new char[file_size]);

//     //Read the entire file at once
//     file.seekg(0, std::ios::beg);
//     file.read(buffer.get(), file_size);
//     file.close();

//     size_t size = 10000;
//     size_t capacity = limit - start;

//     if(capacity > 0 && capacity < size){
//         size = capacity;
//     }

//     for(std::size_t i = 0; i < size; ++i){
//         const size_t l = buffer[i * 3073];

//         labels(start + i)(l) = 1.0;

//         for(std::size_t j = 1; j < 3073; ++j){
//             images(start + i)[j - 1] = buffer[i * 3073 + j];
//         }
//     }
// }

// /*!
//  * \brief Read all training data
//  *
//  * The dataset is assumed to be in a cifar-10 subfolder
//  *
//  * \param limit The maximum number of elements to read (0: no limit)
//  * \param func The functor to create the image objects.
//  */
// template <typename Images, typename Labels>
// void read_training_categorical(const std::string& folder, std::size_t limit, Images& images, Labels& labels) {
//     read_cifar10_file_categorical(images, labels, folder + "/data_batch_1.bin", limit, 0);
//     read_cifar10_file_categorical(images, labels, folder + "/data_batch_2.bin", limit, 10000);
//     read_cifar10_file_categorical(images, labels, folder + "/data_batch_3.bin", limit, 20000);
//     read_cifar10_file_categorical(images, labels, folder + "/data_batch_4.bin", limit, 30000);
//     read_cifar10_file_categorical(images, labels, folder + "/data_batch_5.bin", limit, 40000);
// }

// /*!
//  * \brief Read all test data.
//  *
//  * The dataset is assumed to be in a cifar-10 subfolder
//  *
//  * \param limit The maximum number of elements to read (0: no limit)
//  * \param func The functor to create the image objects.
//  */
// template <typename Images, typename Labels>
// void read_test_categorical(const std::string& folder, std::size_t limit, Images& images, Labels& labels) {
//     read_cifar10_file_categorical(images, labels, folder + "/test_batch.bin", limit, 0);
// }

// /*!
//  * \brief Read all training data
//  *
//  * The dataset is assumed to be in a cifar-10 subfolder
//  *
//  * \param limit The maximum number of elements to read (0: no limit)
//  * \param func The functor to create the image objects.
//  */
// template <typename Images, typename Labels>
// void read_training_categorical(std::size_t limit, Images& images, Labels& labels) {
//     read_training_categorical("./cifar-10-batches-bin", limit, images, labels);
// }

// /*!
//  * \brief Read all test data.
//  *
//  * The dataset is assumed to be in a cifar-10 subfolder
//  *
//  * \param limit The maximum number of elements to read (0: no limit)
//  * \param func The functor to create the image objects.
//  */
// template <typename Images, typename Labels>
// void read_test_categorical(std::size_t limit, Images& images, Labels& labels) {
//     read_test_categorical("./cifar-10-batches-bin", limit, images, labels);
// }

// /*!
//  * \brief Read dataset and assume images in 3D (3x32x32)
//  *
//  * The dataset is assumed to be in a cifar-10 subfolder
//  *
//  * \param training_limit The maximum number of elements to read from data set (0: no limit)
//  * \param test_limit The maximum number of elements to read from test set (0: no limit)
//  *
//  * \return The dataset
//  */
// template <template <typename...> class Container, typename Image, typename Label = uint8_t>
// CIFAR10_dataset<Container, Image, Label> read_dataset_3d(std::size_t training_limit = 0, std::size_t test_limit = 0) {
//     CIFAR10_dataset<Container, Image, Label> dataset;

//     read_training(training_limit, dataset.training_images, dataset.training_labels, [] { return Image(3, 32, 32); });
//     read_test(test_limit, dataset.training_images, dataset.training_labels, [] { return Image(3, 32, 32); });

//     return dataset;
// }

// /*!
//  * \brief Read dataset.
//  *
//  * The dataset is assumed to be in a cifar-10 subfolder
//  *
//  * \param training_limit The maximum number of elements to read from data set (0: no limit)
//  * \param test_limit The maximum number of elements to read from test set (0: no limit)
//  *
//  * \return The dataset
//  */
// template <template <typename...> class Container, typename Image, typename Label = uint8_t>
// CIFAR10_dataset<Container, Image, Label> read_dataset_direct(std::size_t training_limit = 0, std::size_t test_limit = 0) {
//     CIFAR10_dataset<Container, Image, Label> dataset;

//     read_training(training_limit, dataset.training_images, dataset.training_labels, [] { return Image(3 * 32 * 32); });
//     read_test(test_limit, dataset.test_images, dataset.test_labels, [] { return Image(3 * 32 * 32); });

//     return dataset;
// }

// /*!
//  * \brief Read dataset.
//  *
//  * The dataset is assumed to be in a cifar-10 subfolder
//  *
//  * \param training_limit The maximum number of elements to read from training set (0: no limit)
//  * \param test_limit The maximum number of elements to read from test set (0: no limit)
//  * \return The dataset
//  */
// template <template <typename...> class Container = std::vector, template <typename...> class Sub = std::vector, typename Pixel = uint8_t, typename Label = uint8_t>
// CIFAR10_dataset<Container, Sub<Pixel>, Label> read_dataset(std::size_t training_limit = 0, std::size_t test_limit = 0) {
//     return read_dataset_direct<Container, Sub<Pixel>, Label>(training_limit, test_limit);
// }

// } //end of namespace cifar



#Fashion

int ReverseInt (int i)
{
    unsigned char ch1, ch2, ch3, ch4;
    ch1=i&255;
    ch2=(i>>8)&255;
    ch3=(i>>16)&255;
    ch4=(i>>24)&255;
    return((int)ch1<<24)+((int)ch2<<16)+((int)ch3<<8)+ch4;
}

void ReadMNIST_train_im(int NumberOfImages, int DataOfAnImage,double arr[][784])
{
   // arr.resize(NumberOfImages,vector<double>(DataOfAnImage));
    ifstream file ("/home/misbah/Downloads/train-images-idx3-ubyte",ios::binary);
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


    ReadMNIST_train_im(10000,784, X);
    // ReadMNIST_train_label(10000, y);
    

    for (int k = 0; k<5; k++)
    {
    	std::string name = "training_";
    	double image[28][28];
    	int i,j;


	    for ( i = 0; i<28; i++)
	    	for ( j = 0; j<28; j++)
	    		// cout << i*28+j << endl;
	    		image[i][j] = X[k][i*28+j];
	    cv::Mat A(28,28,CV_64F);
		std::memcpy(A.data, image, 28*28*sizeof(double));

		std::string id = std::to_string(k);
		name = name + id;
		name = name+".jpg";
		// cout << name << endl;
		imwrite(name, A);

    }
    


return 0;
}
