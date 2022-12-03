#include <iostream>
#include <fstream>
#include <vector>

using namespace std;

int main()
{



    std::ifstream fileRead("fashion_test_ground_truth.dat", std::ios::binary);
    std::vector<int> load;
    int temp;
    while(fileRead.read((char*)&temp, sizeof(int)))
        load.push_back(temp);
    fileRead.close();

    std::ifstream fileRead2("fashion_test_prediction.dat", std::ios::binary);
    std::vector<int> pred;
    int temp2;
    while(fileRead2.read((char*)&temp2, sizeof(int)))
        pred.push_back(temp2);
    fileRead2.close();

    int count = 0;
    for(int i = 0; i < load.size(); i++) 
    {

        if (load[i] != pred[i])
            count +=1;        
    }

    float mnew = count/1000.0;

    std::cout << "M_new: " << mnew << endl;

    return 0;
}
