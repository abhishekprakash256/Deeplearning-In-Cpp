/*
The DCE loss of the model from the paper 
Loss(DCE) = − log P(yi|x) = − log(
K j
=1
P(x ∈ pij|x))

*/

//imports
#include <torch/torch.h>
#include <iostream>
#include <bits/stdc++.h>

using namespace std;

class DCE_loss2
{
    public:
    
    int feed_network(int features, int prototypes, int classes,torch::Tensor tensor1, torch::Tensor tensor2 )
    {
        int diff,prob, one, sum_arr;
        torch::Tensor out;   
        int dist,eps,p,dist_mul ;
        int prodtype[20];
        p = 6;
        int tensor3,tensor4;
        int gamma = 0.6;
        int i=0,j=0;
        int sum1, sum2;
        sum1 = 0,sum2 = 0;
        int M,N; 
        int final_loss;
        for (i = 0; i < M; ++i) {
            for (j = 0; j < N; ++j) {
                int val1;
                torch::Tensor val_1  = tensor1[i][j] ;
                sum1 = sum1 + val1;
            }
        }  

        for (i = 0; i < M; ++i) {
            for (j = 0; j < N; ++j) {
                int val2;
                torch::Tensor val_2  = tensor2[i][j] ;
                sum2 = sum2 + val2;
            }
        } 

        if (sum1 > sum2)
        {
            diff = (sum1 - sum2);
        }
        else
        {
            diff = (sum2 - sum1);

        }

        int sum_arr_two =0;

        for (int i = 0; i < sizeof(prodtype); i++)
            {
                sum_arr_two = sum_arr+ prodtype[i];
            }

        //compute the distance 
        dist = features * diff;

        //calculate the probablty 
        double dist_pow = pow(dist,2);
        prob = dist_pow*gamma ; 

        //calculate the distance in the values

        dist_mul =  diff*features*prototypes;

        double distances_pow = pow(dist_mul,2);

        one = gamma *distances_pow*sum_arr_two;

        int log_val ;
        int div;

        //calulate the div
        div = -(prob/one);

        //take the log 
        log_val = log10(div);

        final_loss = log_val;

        return final_loss;
    }

};

