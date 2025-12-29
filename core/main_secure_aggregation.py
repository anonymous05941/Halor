import logging
import math
import pickle
from random import Random
import csv
import numpy as np
import collections
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from random import Random

from collections import OrderedDict

from utils.speech import SPEECH
from utils.transforms_wav import ChangeSpeedAndPitchAudio, ChangeAmplitude, FixAudioLength, ToMelSpectrogram, LoadAudio, ToTensor
from utils.transforms_stft import ToSTFT, StretchAudioOnSTFT, TimeshiftAudioOnSTFT, FixSTFTDimension, ToMelSpectrogramFromSTFT, DeleteSTFT, AddBackgroundNoiseOnSTFT
from utils.speech import BackgroundNoiseDataset

from utils.utils_data import get_data_transform

import random

import math

import copy




def generate_attack_shares(secret_vector, attack_flag, noise_level, constant_bound): 

    if attack_flag == 1:      

        ranges = [bound for bound in secret_vector]                             
        vdim = len(ranges)
        noise = [int(ele) for ele in noise_level * np.random.rand(vdim)]  # [0, noise_level]
    
        
        vector_= [ranges[j] + noise[j] for j in range(len(ranges))]
        vector = vector_


    if attack_flag == 2:      
        vector = [bound for bound in secret_vector]
        # vector = secret_vector.copy()

        random.shuffle(vector) 

    
    return vector






def generate_shares(secret_vector, vector_dim):
    ranges = [(0, bound) for bound in secret_vector]                     

    mask = [np.random.randint(low, high) for low, high in ranges]
    
    return mask








# def MPC_share(num_parties, num_matrix, num_label, ratio = None):
#     share_targets = list(range(0, num_parties))
#     random.shuffle(share_targets)


#     generated_masks = {j: [] for j in range(num_parties)} 
#     received_masks = {j: [] for j in range(num_parties)} 
    
#     all_shares = {j: [] for j in range(num_parties)}  


#     for i in range(num_parties):
#         index = share_targets[i]
        
#         generated_mask = generate_shares(num_matrix[index], num_label)                                                                                      
#         # generated_masks[i].append(generated_mask)
#         generated_masks[index] = generated_masks[index] + generated_mask       
        
#         # 记录收到的mask
#         if i < num_parties-1:
#             next_index = share_targets[i+1]
#             # received_masks[index+1].append(generated_mask) 
#             received_masks[next_index] = received_masks[next_index] + generated_mask 
#         else:
# #            received_masks[0].append(generated_mask)
#             next_index = share_targets[0]
#             received_masks[next_index] = received_masks[next_index] + generated_mask  

#     up_sums = []
#     for cid in range(num_parties):     
#         # up_sums.append(num_matrix[i] + generated_masks[i] - received_masks[i])
#         up_sums.append(num_matrix[cid] + np.array(generated_masks[cid]) - np.array(received_masks[cid]))






#     if ratio != None: 
#         sample_number = math.floor((1-ratio)*num_parties)
#         clients_id_set = [i for i in range(0, num_parties, 1)]
#         resample_id_set = rng.sample(clients_id_set, sample_number)

#         new_matrix = np.zeros((num_parties, num_label))
#         # for i in range(num_parties):
#         for i in resample_id_set:
#             # new_matrix[i] = num_matrix[i]
#             new_matrix[i] = up_sums[i]

#     else:
#         new_matrix = up_sums


#     return new_matrix                      








def Attack_MPC_share(num_parties, num_matrix, num_label, drop_ratio = None, malicious_ratio = None, noise_level = 0, constant_bound = 0, attack_flag = 0):
    share_clients = list(range(0, num_parties))

    share_targets = copy.deepcopy(share_clients)

    random.shuffle(share_targets)


    generated_masks = {j: [] for j in range(num_parties)} 
    received_masks = {j: [] for j in range(num_parties)} 
    
    all_shares = {j: [] for j in range(num_parties)}  



    drop_num = math.floor(num_parties*drop_ratio)
    malicous_num = math.floor(num_parties*malicious_ratio)
    drop_clients = random.sample(share_targets, drop_num)



    for item in drop_clients:
        # print("\n item is:", item)
        while item in share_targets:
            share_targets.remove(item)

    malicous_clients = rng.sample(share_targets, malicous_num)



    for i in range(num_parties):
        index = share_clients[i]

        if index in malicous_clients:
            if attack == 1:
                original_mask = generate_shares(num_matrix[index], num_label) 
                attack_mask = generate_attack_shares(original_mask, attack_flag, noise_level, constant_bound) 
                generated_mask = [np.clip(item, -constant_bound, constant_bound) for item in attack_mask] 

            else:
                original_mask = generate_shares(num_matrix[index], num_label) 
                attack_mask = generate_attack_shares(original_mask, attack_flag, noise_level, constant_bound) 
                generated_mask = [np.clip(item, -constant_bound, constant_bound) for item in attack_mask] 

        else:    
            generated_mask = generate_shares(num_matrix[index], num_label)    

   
        generated_masks[index].extend(generated_mask)       


        if i < num_parties-1:
            next_index = share_clients[i+1]
            # received_masks[index+1].append(generated_mask) 
            # received_masks[next_index] = received_masks[next_index] + generated_mask 
            received_masks[next_index].extend(generated_mask) 
        else:
#            received_masks[0].append(generated_mask)
            next_index = share_clients[0]
            # received_masks[next_index] = received_masks[next_index] + generated_mask  
            received_masks[next_index].extend(generated_mask) 



    new_matrix = np.zeros((num_parties, num_label))
    for cid in range(num_parties):                        
 
        if cid in drop_clients:
            pass
        else:
            j = share_clients.index(cid)            
            if j == 0:
                pre_j = num_parties - 1        
            else:
                pre_j = j -1
        
            pre_cid = share_clients[pre_j]        

            vector = None
            if pre_cid in drop_clients:
                new_matrix[cid] = num_matrix[cid] + generated_masks[cid]
                # new_matrix[cid] = num_matrix[cid] + generated_masks[cid] - received_masks[cid]
            else:
        #     up_sums.append(num_matrix[cid] + np.array(generated_masks[cid]) - np.array(received_masks[cid]))
                vector = num_matrix[cid] + generated_masks[cid] - received_masks[cid]
                # print(new_matrix[cid])   
                if cid in malicous_clients: 
                    vector_ = generate_attack_shares(vector, attack_flag, noise_level, constant_bound)
                    new_matrix[cid] = [np.clip(item, -constant_bound, constant_bound) for item in vector_]               
                else:
                    new_matrix[cid] = vector


    return new_matrix                  












def obtain_targets(data, data_set):
    labels = data.targets

    targets = OrderedDict()                                
    indexToLabel = {}

    for index, label in enumerate(labels):
        if data_set == "Mnist":
            label = label.item()
        if data_set == "fashion_mnist":
            label = label.item()                

        if data_set == "emnist":
            label = label.item()   

        if label not in targets:
            targets[label] = []                           

        targets[label].append(index)                       
        indexToLabel[index] = label

    return targets







def custom_partition2(data, dset, num_clients, partition, numOfLabels, alpha, sample_ratio, ratio=1.0, malicious_r = 0, noise_l = 0, constant_b = 0, attack = 0):          
    initial_targets = obtain_targets(data, dset)
    targets = getTargets(initial_targets)      
    data_len = len(data)

    usedSamples = int(data_len / num_clients)


    keyDir = {key: int(key) for i, key in enumerate(targets.keys())}         
    keyLength = [0] * numOfLabels
    # for key in keyDir.keys():                                             
    #     keyLength[keyDir[key]] = len(targets[key])                        



    count = 0
    count_vec = []
    for key in keyDir.keys():                                             
        keyLength[keyDir[key]] = len(targets[key])                        
        count_vec.append(len(targets[key]))
    

    if dset == "google_speech":
        print('Google_speech training set: original label num vector is', count_vec)
        print('Google_speech training set: original data portion is', count_vec/np.sum(count_vec))
    
    if dset == "cifar10":
        print('Cifar10 training set: original label num vector is', count_vec)
        print('Cifar10 training set: original data portion is', count_vec/np.sum(count_vec))

    if dset == "fashion_mnist":
        print('Fashion_mnist training set: original label num vector is', count_vec)
        print('Fashion_mnist training set: original data portion is', count_vec/np.sum(count_vec))

    print("--------------------------------------------------------------------------------------------")


    average_portion = [1/numOfLabels for i in range(numOfLabels)]

    args = {
        "filter_class": 0,
        "filter_class_ratio": 1
    }

    ratioOfClassWorker = create_mapping(num_clients, partition, numOfLabels, alpha, args)       

    sumRatiosPerClass = np.sum(ratioOfClassWorker, axis=1)                                                 
    for worker in range(num_clients):
        ratioOfClassWorker[worker, :] = ratioOfClassWorker[worker, :] / float(sumRatiosPerClass[worker])  


    global_count = np.zeros((num_clients, numOfLabels))
    global_portion = np.zeros((1, numOfLabels))        
    for worker in range(num_clients):
        
        for label, label_key in enumerate(targets.keys()):
            global_count[worker][keyDir[label_key]] = min(int(usedSamples * ratioOfClassWorker[worker][keyDir[label_key]]), keyLength[keyDir[label_key]])
            # global_count[worker][keyDir[label_key]] = int(usedSamples * ratioOfClassWorker[worker][keyDir[label_key]])



    global_count_total = np.sum(global_count, axis = 0)                      
    global_portion = global_count_total/np.sum(global_count_total)           


    print('Global portion before regulation is', global_portion)                                                              
    print('After parition, total examples bofore regulation is', np.sum(global_count_total))          
    print('After parition, num of each label before regulation is', global_count_total)               
    print("--------------------------------------------------------------------------------------------")





    #Dropout and attack
    A_MPC_matrix = Attack_MPC_share(num_clients, global_count, numOfLabels, sample_ratio, malicious_r, noise_l, constant_b, attack)
    A_global_count_MPC_total = np.sum(A_MPC_matrix, axis = 0)                           
    A_global_count_total = A_global_count_MPC_total  
    A_global_portion = A_global_count_total/np.sum(A_global_count_total)                                                                  
    print("\n")
    print('MPC under attack: Global portion is (', 100*sample_ratio,'%' 'clients are missed and', 100*malicious_r,'%' 'clients are malicious)', A_global_portion)                              
    final_estimation = [item for item in A_global_portion]
    print('MPC under attack: Global portion is (', 100*sample_ratio,'%' 'clients are missed and', 100*malicious_r,'%' 'clients are malicious)', final_estimation)
    print("--------------------------------------------------------------------------------------------")  


    return global_portion



def create_mapping(sizes, partitioning, numOfLabels, dirichlet_alpha, args):                                  
    # numOfLabels = self.getNumOfLabels()                           

    ratioOfClassWorker = None                                     
    if partitioning == 1:                                #1 NONIID-Uniform
        ratioOfClassWorker = np.random.rand(len(sizes), numOfLabels).astype(np.float32)    

    # elif partitioning == 2:                              #2 NONIID-Zipfian      
    #     ratioOfClassWorker = np.random.zipf(self.args.zipf_param, [len(sizes), numOfLabels]).astype(np.float32)  
    elif partitioning == 3:                              #3 NONIID-Balanced
        ratioOfClassWorker = np.ones((sizes, numOfLabels)).astype(np.float32)
    elif partitioning == 8:                            
        dirichlet_list = () 
        dirichlet_list = (dirichlet_alpha,)*numOfLabels       
        ratioOfClassWorker = np.zeros([sizes, numOfLabels]).astype(np.float32)
        for j in range(sizes):
            ratioOfClassWorker[j] = np.random.dirichlet([dirichlet_alpha]*numOfLabels).astype(np.float32)        

    elif partitioning == 6:                           
        logging.info('Method of create_mapping is 6')
        dirichlet_list = [] 
        up_bound1 = 2                                         
        low_bound1 = 50
        up_bound2 = 100
                                                
        ratio1 = 0.38                               
        num1 =  math.floor(sizes*ratio1)             
        num2 = math.ceil(sizes*(1-ratio1))         


        step1 = (up_bound1-dirichlet_alpha)/num1             
        step2 = 2*(up_bound2-low_bound1)/num2                 

        logging.info(f"up_bound1 is {up_bound1}!")            


        ratioOfClassWorker = np.zeros([sizes, numOfLabels]).astype(np.float32)
#            dirichlet_list = (self.args.dirichlet_alpha,)*numOfLabelslen     
        for j in range(sizes):
#                logging.info("==== length of sizes is:{}\n".format(self.args.partitioning, len(sizes), numOfLabels, num_remove_class, np.count_nonzero(ratioOfClassWorker)))   
            if j < (sizes/2):
#               dirichlet_list.append([self.args.dirichlet_alpha+j*step,]*numOfLabelslen)            
                ratioOfClassWorker[j] = np.random.dirichlet([dirichlet_alpha+j*step1,]*numOfLabels).astype(np.float32)
            else:
                ratioOfClassWorker[j] = np.random.dirichlet([low_bound1+j*step2,]*numOfLabels).astype(np.float32)

#            dirichlet_list.append = []            
#            ratioOfClassWorker = np.random.dirichlet(numOfLabelslen, size =len(sizes)).astype(np.float32)        

    elif partitioning == 7:                 
        logging.info('Method of create_mapping is 7')
        dirichlet_list = [] 
        up_bound1 = 2                                       
        low_bound1 = 50
        up_bound2 = 100
                                                
        ratio1 = 0.38                              
        num1 =  math.floor(sizes*ratio1)          
        num2 = math.ceil(sizes*(1-ratio1))              

        step1 = (up_bound1-dirichlet_alpha)/num1           
        step2 = 2*(up_bound2-low_bound1)/num2           


        ratioOfClassWorker = np.zeros([sizes, numOfLabels]).astype(np.float32)
        ratioOfClassWorker1 = np.zeros([sizes, numOfLabels]).astype(np.float32)
#            dirichlet_list = (self.args.dirichlet_alpha,)*numOfLabelslen    
        for j in range(sizes):
#                logging.info("==== length of sizes is:{}\n".format(self.args.partitioning, len(sizes), numOfLabels, num_remove_class, np.count_nonzero(ratioOfClassWorker)))   
            if j < (sizes/2):
#               dirichlet_list.append([self.args.dirichlet_alpha+j*step,]*numOfLabelslen)            
                ratioOfClassWorker1[j] = np.random.dirichlet([dirichlet_alpha+j*step1,]*numOfLabels).astype(np.float32)
            else:
                ratioOfClassWorker1[j] = np.random.dirichlet([low_bound1+j*step2,]*numOfLabels).astype(np.float32)
        
        for j in range(sizes):
            ratioOfClassWorker[sizes-1-j] = ratioOfClassWorker1[j]






    num_remove_class=0
    if args["filter_class"] > 0 or args["filter_class_ratio"] > 0:
        num_remove_class = args["filter_class"] if args["filter_class"] else round(numOfLabels * (1 - args["filter_class_ratio"])) 
        for w in range(len(sizes)):                             
            # randomly filter classes by forcing zero samples
            wrandom = rng.sample(range(numOfLabels), num_remove_class)   
            for wr in wrandom:
                ratioOfClassWorker[w][wr] = 0.0 #0.001           



    #logging.info("==== Class per worker partitioning:{} clients:{} labels:{} rem_lable:{} count:{}  ====\n {} \n".format(self.args.partitioning, len(sizes), numOfLabels, num_remove_class, np.count_nonzero(ratioOfClassWorker), repr(ratioOfClassWorker)))
    logging.info("==== Class per worker partitioning:{} clients:{} labels:{} rem_lable:{} count:{}  ==== \n".format(partitioning, sizes, numOfLabels, num_remove_class, np.count_nonzero(ratioOfClassWorker)))    
    return ratioOfClassWorker




def getTargets(targets):                        
    tempTarget = targets.copy()

    for key in tempTarget:
            rng.shuffle(tempTarget[key])
    return tempTarget

















#Google speech
# seed = 10
# clients = 1000
# label = 35
# #usedSamples = 102
# # sample_constraint = 2,938
# data_dir = '/home/yhchen/paper2/dataset/data/google_speech'
# partition = 8
# d_alpha = 0.1
# job = 'google_speech'

# bkg = '_background_noise_'
# data_aug_transform = transforms.Compose([ChangeAmplitude(), ChangeSpeedAndPitchAudio(), FixAudioLength(), ToSTFT(), StretchAudioOnSTFT(), TimeshiftAudioOnSTFT(), FixSTFTDimension()])
# bg_dataset = BackgroundNoiseDataset(os.path.join(data_dir, bkg), data_aug_transform)
# add_bg_noise = AddBackgroundNoiseOnSTFT(bg_dataset)
# train_feature_transform = transforms.Compose([ToMelSpectrogramFromSTFT(n_mels=32), DeleteSTFT(), ToTensor('mel_spectrogram', 'input')])
# train_dataset = SPEECH(data_dir, dataset= 'train',
#                         transform=transforms.Compose([LoadAudio(),
#                                 data_aug_transform,
#                                 add_bg_noise,
#                                 train_feature_transform]))
# valid_feature_transform = transforms.Compose([ToMelSpectrogram(n_mels=32), ToTensor('mel_spectrogram', 'input')])
# test_dataset = SPEECH(data_dir, dataset='test',
#                         transform=transforms.Compose([LoadAudio(),
#                                 FixAudioLength(),
#                                 valid_feature_transform]))





#Cifar10
# seed = 10  
# clients = 1000
# label = 10
# #usedSamples = 102
# # sample_constraint = 2,938
# data_dir = '/home/yhchen/paper2/dataset/data/cifar10'
# partition = 8
# d_alpha = 0.1
# job = 'cifar10'

# train_transform, test_transform = get_data_transform('cifar')
# train_dataset = datasets.CIFAR10(data_dir, train=True, download=False, transform=train_transform)
# test_dataset = datasets.CIFAR10(data_dir, train=False, download=False, transform=test_transform)

# ratio = 1.0
# malicious_r = 0.05
# noise_l = 50
# constant_b = 50
# attack = 2




#FashionMNIST
seed = 11  
clients = 1000
label = 10
#usedSamples = 102
# sample_constraint = 2,938
data_dir = '/home/yhchen/paper2/dataset/data/'
partition = 8
d_alpha = 0.1
job = 'fashion_mnist'

train_dataset = datasets.FashionMNIST(data_dir, train=True, download=False, transform=transforms.Compose([transforms.Resize(size=224), 
transforms.Grayscale(3), transforms.ToTensor()]))
test_dataset = datasets.FashionMNIST(data_dir, train=False, download=False, transform=transforms.Compose([transforms.Resize(size=224), 
transforms.Grayscale(3), transforms.ToTensor()]))

ratio = 1.0
malicious_r = 0.05
# noise_l = 60
noise_l = 60
constant_b = 60
attack = 2









#set up the data generator to have consistent results
# seed = 10                        
generator = torch.Generator()
generator.manual_seed(seed)                              
np.random.seed(seed)                                        
rng = Random()
rng.seed(seed)
miss_ratio = 0.05
      
expected_portion = custom_partition2(train_dataset, job, clients, partition, label, d_alpha, miss_ratio, ratio, malicious_r, noise_l, constant_b, attack)