#include <torch/torch.h>
#include"dce_loss2.cpp"


#include <cstddef>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

using namespace std ;

// Where to find the data
const char* data = "/home/abhishek/Personal/dbms_project/pytorch_cpp/exp/data";  // the data path for fashion mnsit 

const int64_t NumberOfEpochs = 20;
const int64_t LogInterval = 16;
const int64_t Train_size = 16;
const int64_t TestBatchSize = 16;
const int64_t labelled_data = 5;
const int64_t unlablled_data = 5;

// the model 
struct Net0 : torch::nn::Module {
  Net0()
      : conv1(torch::nn::Conv2dOptions(1, 64, 3)),  //(in_channel, out_channel, kernel_size)
        conv2(torch::nn::Conv2dOptions(64, 20, 5)), // added the layer 
        conv3(torch::nn::Conv2dOptions(64,20,5)),  // added the layer
        conv4(torch::nn::Conv2dOptions(128,128,3)),  //added the layer
        conv5(torch::nn::Conv2dOptions(128,256,3)),  //added the layer
        conv6(torch::nn::Conv2dOptions(256,256,3)),  //added the layer
        conv7(torch::nn::Conv2dOptions(256,256,3)),  //added the layer
        conv8(torch::nn::Conv2dOptions(256,512,3)),  //added the layer
        conv9(torch::nn::Conv2dOptions(512,512,3)),  //added the layer
        conv10(torch::nn::Conv2dOptions(512,512,3)),  //added the layer
        conv11(torch::nn::Conv2dOptions(512,512,3)),  //added the layer
        conv12(torch::nn::Conv2dOptions(512,30,5)),  //added the layer
        conv13(torch::nn::Conv2dOptions(30,40,5)),  //added the layer
        fc1(320,50),  // the fully connected layer 
        fc2(50,40),   // the fully connected layer
        fc3(40,20), // the fully connected layer
        fc4(20,10){  // the fully connected layer
    register_module("conv1", conv1);  //add the layer
    register_module("conv2", conv2);  // add the layer 
    register_module("conv3", conv3);  //add the layer 
    register_module("conv4", conv4);   // add the layer 
    register_module("conv6", conv6);   // add the layer 
    register_module("conv7", conv7);   // add the layer 
    register_module("conv8", conv8);   // add the layer 
    register_module("conv9", conv9);   // add the layer 
    register_module("conv10", conv10);   // add the layer 
    register_module("conv11", conv11);   // add the layer 
    register_module("conv12", conv12);   // add the layer
    register_module("conv13", conv13);   // add the layer      
    register_module("conv2_drop", conv2_drop);
    register_module("fc1", fc1);   // add the layer  
    register_module("fc2", fc2);   // add the layer  
    register_module("fc3", fc3);    // add the layer  
  }

  // the forward fucntion for passing the input
  torch::Tensor forward(torch::Tensor x) {
    x = torch::relu(torch::max_pool2d(conv1->forward(x), 2));
    x = torch::relu(
        torch::max_pool2d(conv2_drop->forward(conv2->forward(x)), 2));
    x = x.view({-1, 320});
    x = torch::relu(fc1->forward(x));
    x = torch::dropout(x, 0.25,is_training());
    x = fc2->forward(x);
    x = fc3->forward(x); 
    x = fc4->forward(x);
    return torch::log_softmax(x,1);
  }

  torch::nn::Conv2d conv1;  // register the layer
  torch::nn::Conv2d conv2;  // register the layer
  torch::nn::Conv2d conv3;  // register the layer 
  torch::nn::Conv2d conv4;  //register the layer 
  torch::nn::Conv2d conv5;  //register the layer 
  torch::nn::Conv2d conv6;  //register the layer 
  torch::nn::Conv2d conv7;  //register the layer 
  torch::nn::Conv2d conv8;  //register the layer 
  torch::nn::Conv2d conv9;  //register the layer 
  torch::nn::Conv2d conv10;  //register the layer 
  torch::nn::Conv2d conv11;  //register the layer 
  torch::nn::Conv2d conv12;  //register the layer 
  torch::nn::Conv2d conv13;  //register the layer
  torch::nn::Dropout2d conv2_drop;  // the dropout layer 
  torch::nn::Linear fc1;  // fully connected layer 
  torch::nn::Linear fc2;  // fully connected layer 
  torch::nn::Linear fc3;  // fully connected layer 
  torch::nn::Linear fc4;  // fully connected layer 
};


template <typename DataLoader>
void train_on_unseen_labels( 
   /*
    The funnction is the train function 
    passed the model and use the train model 
    the model is passed and the model is trained 
    with the loss optmizer and use the DCE loss
    */
    // the initilizaton of the paramaters
    size_t epoch,
    Net0& model,
    torch::Device device,
    DataLoader& data_loader,
    torch::optim::Optimizer& optimizer,
    size_t dataset_size) {

  // model trainer and the loss calulater 
  // the optimizer of the model
  model.train();
  size_t batch_idx = 0;
  int label;
  int unseen_label;
  for (auto& batch : data_loader) {
    auto data = batch.data.to(device), targets = batch.target.to(device);
    optimizer.zero_grad();
    DCE_loss2 dce_loss2;
    int x , y , z  ;
    auto output = model.forward(data);
    auto loss = torch::nll_loss(output, targets);
    AT_ASSERT(!std::isnan(loss.template item<float>()));
    loss.backward();
    optimizer.step();

     // calc the loss per epoch defined 
    if (batch_idx++ % LogInterval == 0) {
      std::printf(
          "\rTrain Epoch: %ld [%5ld/%5ld] Loss: %.4f",
          epoch,
          batch_idx * batch.data.size(0),
          dataset_size,
          loss.template item<float>());
    }
  }
}


template <typename DataLoader>
void test_unlabelled(  
      /*
    The funnction is the test function 
    that is used for the model to test on the data
    the test data is used for evaluaton of the accuracy
    */

    // the paramaters for the model
    Net0& model,
    torch::Device device,
    DataLoader& data_loader,
    size_t dataset_size) {
  torch::NoGradGuard no_grad;
  model.eval();
  double test_loss = 0;
  int32_t correct = 0;
  int label;
  int unseen_label;
  // run the loop on the batch for the test data
  // calculate the accuracy 
  for (const auto& batch : data_loader) {
    auto data = batch.data.to(device), targets = batch.target.to(device);
    auto output = model.forward(data);
    test_loss += torch::nll_loss(
                     output,
                     targets,
                     /*weight=*/{},
                     torch::Reduction::Sum)
                     .template item<float>();
    auto pred = output.argmax(1);
    correct += pred.eq(targets).sum().template item<int64_t>();
  }

  //generate the test loss as per dataset_size

  test_loss /= dataset_size;
  std::printf(
      "\nTest set: Average loss on the epochs: %.5f | Accuracy per batch: %.5f\n",
      test_loss,
      static_cast<double>(correct) / dataset_size);
}

auto main_trainer() -> int {
      /*
    The main function used for the trainer of the model
    the test function is also called and tested on the data
    the data is loaded from the dataloader class
    */
  torch::manual_seed(2);
  int lablled_class;
  int unlablled_class;

  // the dataloader using the torch methods 
  torch::DeviceType device_type;
  if (torch::cuda::is_available()) {
    std::cout << "CUDA available! Training on GPU." << std::endl;
    device_type = torch::kCUDA;
  } else {
    std::cout << "Training on CPU." << std::endl;
    device_type = torch::kCPU;
  }
  torch::Device device(device_type);

  Net0 model;
  model.to(device);
    // the train data loader 
  auto train_dataset = torch::data::datasets::MNIST(data)
                           .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
                           .map(torch::data::transforms::Stack<>());
  const size_t train_dataset_size = train_dataset.size().value();
  auto train_loader =
      torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
          std::move(train_dataset), Train_size);
    // the test data loader
  auto test_dataset = torch::data::datasets::MNIST(
                          data, torch::data::datasets::MNIST::Mode::kTest)
                          .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
                          .map(torch::data::transforms::Stack<>());
  const size_t test_dataset_size = test_dataset.size().value();
  auto test_loader =
      torch::data::make_data_loader(std::move(test_dataset), TestBatchSize);
   
   // available inbuild optimizer for the model using the stochastic gradient descen
  torch::optim::SGD optimizer(
      model.parameters(), torch::optim::SGDOptions(0.01).momentum(0.5));

    // run the model as per epochs
  for (size_t epoch = 1; epoch <= NumberOfEpochs; ++epoch) {
    train_on_unseen_labels(epoch, model, device, *train_loader, optimizer, train_dataset_size);
    test_unlabelled(model, device, *test_loader, test_dataset_size);
  }

  return 0;
}
