
## Datasets and Benchmarks of FedDance

## Realistic FL Datasets


We provide real-world datasets for the federated learning community, and plan to release much more soon! Each is associated with its training, validation and testing dataset. A summary of statistics for training datasets can be found in Table. 

CV tasks:

| Dataset   | Data Type   | # of Clients | # of Samples | Example Task | 
|-----------| ----------- |--------------|--------------|    ----------- |
| CIFAR10   |   Image     | Custom       | 60,000          |   Classification  |    
| FashionMNIST    |   Image     | Custom       | 70,000         |   Classification  |


NLP tasks:

| Dataset       | Data Type   |# of Clients  | # of Samples   | Example Task | 
| -----------   | ----------- | -----------  |  ----------- |   ----------- |
|Google Speech  |   Audio     |     2,618    |   104,667        |   Speech recognition |



## Repo Structure

```
Current Folder
|---- data               # Dictionary of each datasets 
|---- donwload.sh        # Download tool of each dataset
    
```

## Example Dataset

### Google Speech Commands
A speech recognition dataset with over ten thousand clips of one-second-long duration. Each clip contains one of the 35 common words (e.g., digits zero to nine, "Yes", "No", "Up", "Down") spoken by thousands of different people. 

### CIFAR10
The CIFAR10 dataset is a widely used dataset in machine learning and computer vision. It contains 60,000 color images across 10 classes. The dataset is divided into 50,000 training images and 10,000 test images. The images in CIFAR10 are 32x32 pixels in size. 

### FashionMNIST
The Fashion-MNIST is a widely utilized dataset for evaluating machine learning algorithms. It comprises 70,000 grayscale images representing various clothing items, classified into 10 distinct categories, including T-shirts, trousers, and sneakers. Each image is of size 28x28 pixels.




### Dataset of System Performance and Availability

#### Heterogeneous System Performance
This is captured in file `device_info/client_device_capacity` containing client id as key and computation and communication speed as value pairs. We use the [AIBench](http://ai-benchmark.com/ranking_deeplearning_detailed.html) dataset and [MobiPerf](https://www.measurementlab.net/tests/mobiperf/) dataset. AIBench dataset provides the computation capacity of different models across a wide range of devices. As specified in real [FL deployments](https://arxiv.org/abs/1902.01046), we focus on the capability of mobile devices that have > 2GB RAM in this benchmark. To understand the network capacity of these devices, we clean up the MobiPerf dataset, and provide the available bandwidth when they are connected with WiFi, which is preferred in FL as well. 

#### Availability of Clients
This is captured in file `device_info/client_behave_trace` as key value pair of client id, duration, finish time and a list of active and inactive time-slots of the client. We use a large-scale real-world user behavior dataset from [FLASH](https://github.com/PKU-Chengxu/FLASH). It comes from a popular input method app (IMA) that can be downloaded from Google Play, and covers 136k users and spans one week from January 31st to February 6th in 2020. This dataset includes 180 million trace items (e.g., battery charge or screen lock) and we consider user devices that are in charging to be available, as specified in real [FL deployments](https://arxiv.org/abs/1902.01046).

