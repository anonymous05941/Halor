# -*- coding: utf-8 -*-
import csv
import logging
import os
import pickle
import random
import time
from collections import Counter
# add new modules
from collections import OrderedDict
from pathlib import Path
from random import Random

import numpy as np
import torch
from argParser import args
from fllibs import *
from torch.utils.data import DataLoader

import math   

#set up the data generator to have consistent results
seed = 10                             
generator = torch.Generator()
generator.manual_seed(seed)                                    

def seed_worker(worker_id):
    worker_seed = seed #torch.initial_seed() % 2**32
    np.random.seed(worker_seed)                                 
    random.seed(worker_seed)                                       

class Partition(object):
    """ Dataset partitioning helper """

    def __init__(self, data, index):
        self.data = data
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        data_idx = self.index[index]                                
        return self.data[data_idx]


class DataPartitioner(object):
    """Partition data by trace or random"""                          

    def __init__(self, data, numOfClass=0, seed=10, isTest=False):
        self.partitions = []
        self.rng = Random()
        self.rng.seed(seed)                                

        self.data = data
        self.labels = self.data.targets

        self.args = args
        self.isTest = isTest
        np.random.seed(seed)                                        

        self.data_len = len(self.data)
        self.task = args.task
        self.numOfLabels = numOfClass                             

        #set the number of samples per worker
        self.usedSamples = 0                                          

        #introduce targets dict
        self.targets = OrderedDict()                                
        self.indexToLabel = {}
        

        self.total_partition = []
        self.partition_matrix = np.zeros([self.args.total_clients, self.numOfLabels])

        self.count_partition = np.zeros([1, self.args.num_class])                                  
        self.global_portion = np.zeros([1, self.args.num_class])                                


        # self.history_test_partition = None



        # categarize the samples                                   
        # last_label = None
        # count = 0

#        for index, label in enumerate(self.labels):               
#            if label not in self.targets:
#                self.targets[label] = []                         

#            self.targets[label].append(index)                      
#            self.indexToLabel[index] = label                       


        for index, label in enumerate(self.labels):

            if self.args.data_set == "Mnist":
                label = label.item()
            if self.args.data_set == "fashion_mnist":
                label = label.item()                

            if self.args.data_set == "emnist":
                label = label.item()   

            if label not in self.targets:
                self.targets[label] = []                            

            self.targets[label].append(index)                               
            self.indexToLabel[index] = label



    def getNumOfLabels(self):                                           
        return self.numOfLabels                                   
  
    def getDataLen(self):
        return self.data_len                                    






    def trace_partition(self, data_map_file, ratio=1.0):
        """Read data mapping from data_map_file. Format: <client_id, sample_name, sample_category, category_id>"""
        logging.info(f"Partitioning data by profile {data_map_file}...")             

        clientId_maps = {}
        unique_clientIds = {}
        # load meta data from the data_map_file
        with open(data_map_file) as csv_file:                                       
            csv_reader = csv.reader(csv_file, delimiter=',')
            read_first = True
            sample_id = 0         

            for row in csv_reader:                                                 
                if read_first:                                                     
                    logging.info(f'Trace names are {", ".join(row)}')               
                    read_first = False
                else:
                    client_id = row[0]                                            

                    if client_id not in unique_clientIds:                             
                        unique_clientIds[client_id] = len(unique_clientIds)         

                                                                                      
                    clientId_maps[sample_id] = unique_clientIds[client_id]           
                    
                    sample_id += 1

        # Partition data given mapping
        self.partitions = [[] for _ in range(len(unique_clientIds))]                 

        for idx in range(len(self.data.data)):                                       
            self.partitions[clientId_maps[idx]].append(idx)                          

        for i in range(len(unique_clientIds)):
            self.rng.shuffle(self.partitions[i])                                      
            takelen = max(0, int(len(self.partitions[i]) * ratio))                    
            self.partitions[i] = self.partitions[i][:takelen]







    #add data mapping handlers (uniform, zipf, balanced) and class exclusion   
    def partition_data_helper(self, num_clients, data_map_dir=None):
        tasktype = 'train' if not self.isTest else 'test'                          
        data_map_file = None
        if data_map_dir is not None:                                                
            data_map_file = os.path.join(data_map_dir, tasktype + '.csv')            
            logging.info(f"data_map_file is {os.path.exists(data_map_file)}!")
            #handle the case for reddit dataset where on IBEX mappings are stored on the metadata folder
            if args.data_set == 'reddit' or args.data_set == 'stackoverflow':       
                data_map_dir = os.path.join(args.log_path, 'metadata', args.data_set, tasktype)
                data_map_file = os.path.join(data_map_dir,  'result_' + str(args.process_files_ratio) + '.csv')

        #apply ratio on the data - manipulate the data per uses
        ratio = 1.0
        if not self.isTest and self.args.train_ratio < 1.0:                      
            ratio = self.args.train_ratio
        elif self.isTest and self.args.test_ratio < 1.0:
            ratio = self.args.test_ratio


        if self.isTest:                                               
            if self.args.partitioning < 0 or data_map_file is None or num_clients < args.total_worker:        
                # logging.info(f"Start test uniform_partition!")
#                self.uniform_partition(num_clients=num_clients, ratio=ratio) 


                #-----------------------------------------
                if args.data_regulation == 0:                                             
                    # logging.info(f"Utilize self.uniform_partition!")                      
                    # self.uniform_partition(num_clients=num_clients, ratio=ratio)
                    # logging.info(f"Utilize self.uniform_partition2 to acheive non-iid testing!")                                                                                      #按照全局分布划分
                    # self.uniform_partition2(num_clients=num_clients, ratio=ratio)
                    logging.info(f"Utilize custom_partition_test to acheive non-uniform testing!")
                    self.custom_partition_test(num_clients=num_clients, ratio=ratio)
                else:      
                    # logging.info(f"Utilize self.uniform_partition!")                       
                    # self.uniform_partition(num_clients=num_clients, ratio=ratio)
                    # logging.info(f"Utilize self.uniform_partition2 to acheive data reshape!")                                                                                      #按照全局分布划分
                    # self.uniform_partition2(num_clients=num_clients, ratio=ratio)
                    logging.info(f"Utilize custom_partition_test to acheive non-uniform testing!")
                    self.custom_partition_test(num_clients=num_clients, ratio=ratio)
                


            else:       
                self.trace_partition(data_map_file, ratio=ratio)
        elif self.args.partitioning <= 0:                                    
            if self.args.partitioning < 0 or data_map_file is None:         
                self.uniform_partition(num_clients=num_clients, ratio=ratio)
            else:                                                                                         
                self.trace_partition(data_map_file, ratio=ratio)            
        else:                                                                
#            self.custom_partition(num_clients=num_clients, ratio=ratio)


            #--------------------------------------------------------------------
            if args.data_regulation == 0:                                             
                logging.info(f"Utilize self.custom_partition!")             
                self.custom_partition(num_clients=num_clients, ratio=ratio)
            else:    
                logging.info(f"Utilize self.custom_partition2 to acheive data reshape!")                                                                                   #按照全局分布划分
                self.custom_partition2(num_clients=num_clients, ratio=ratio)
                # logging.info(f"Utilize self.custom_partition3 to acheive data reshape!") 
                # self.custom_partition3(num_clients=num_clients, ratio=ratio)         
                # logging.info(f"Utilize self.custom_partition2_1 to acheive data reshape!")                                                                                   #按照全局分布划分
                # self.custom_partition2_1(num_clients=num_clients, ratio=ratio)                                     
                # logging.info(f"Utilize self.custom_partition2_2 to acheive data reshape!")                                                                                   #按照全局分布划分
                # self.custom_partition2_2(num_clients=num_clients, ratio=ratio) 
            #--------------------------------------------------------------------



    def uniform_partition(self, num_clients, ratio=1.0):                
        num_clients = 4                                              
        # random uniform partition                                      
        numOfLabels = self.getNumOfLabels()                                
        #update the data length to account for the ratio
        data_len = min(self.getDataLen(), int(self.getDataLen() * ratio))  
        logging.info(f"Uniform partitioning data, ratio: {ratio} applied for {data_len} samples of {numOfLabels} labels on {num_clients} clients ...")

        logging.info(f"Uniform partitioning, number of clients is {num_clients}!")


        indexes = list(range(data_len))
        self.rng.shuffle(indexes)                                       

        for _ in range(num_clients):
            part_len = int(1. / num_clients * data_len)                                       
            self.partitions.append(indexes[0:part_len])                
            indexes = indexes[part_len:]




    #--------------------------------------------------------------
    def uniform_partition2(self, num_clients, ratio=1.0): 
        num_clients = 4      

        # random uniform partition                                    
        numOfLabels = self.getNumOfLabels()                           
        #update the data length to account for the ratio
        data_len = min(self.getDataLen(), int(self.getDataLen() * ratio))  
        logging.info(f"Uniform partitioning data, ratio: {ratio} applied for {data_len} samples of {numOfLabels} labels on {num_clients} clients ...")

        logging.info(f"Uniform partitioning, number of clients is {num_clients}!")






#        global_portion_file = 'mapping_part_11_03_Test1_portion_partition' + str(self.args.partitioning) #alpha = 0.1
#        global_portion_file = 'mapping_part_11_05_Test2_portion_partition' + str(self.args.partitioning)  #alpha = 0.5
        # global_portion_file = 'mapping_part_11_06_Test3_portion_partition' + str(self.args.partitioning)  #alpha = 0.1
        # global_portion_file ='mapping_part_11_06_Test4_portion_partition' + str(self.args.partitioning) + '_alpha' + str(self.args.dirichlet_alpha)
        # global_portion_file ='mapping_part_11_06_Test5_portion_partition' + str(self.args.partitioning) + '_alpha' + str(self.args.dirichlet_alpha)

        #EMNIST
        # global_portion_file = 'mapping_part_11_27_Test_Emnist1_portion_partition' + str(self.args.partitioning) + '_alpha' + str(self.args.dirichlet_alpha)
        # global_portion_file = 'mapping_part_11_28_Test_Emnist2_portion_partition' + str(self.args.partitioning) + '_alpha' + str(self.args.dirichlet_alpha)

        #Tiny-imagenet
        # global_portion_file = 'mapping_part_11_28_Test_Tiny_imagenet1_portion_partition' + str(self.args.partitioning) + '_alpha' + str(self.args.dirichlet_alpha)

        #Cifar10
#        global_portion_file = 'mapping_part_11_29_Test1_Cifar10_portion_partition' + str(self.args.partitioning) + '_alpha' + str(self.args.dirichlet_alpha)          
        # global_portion_file = 'mapping_part_11_29_Test2_Cifar10_portion_partition' + str(self.args.partitioning) + '_alpha' + str(self.args.dirichlet_alpha) 



        #从这里开始时第二波
        #Google_speech
        # global_portion_file = 'slave18_mapping_part_12_07_Test6_Google_speech_portion_partition' + str(self.args.partitioning)  + '_alpha' + str(self.args.dirichlet_alpha) 
        # global_portion_file = 'slave18_mapping_part_12_08_Test7_Google_speech_portion_partition' + str(self.args.partitioning)  + '_alpha' + str(self.args.dirichlet_alpha)
        # global_portion_file = 'slave18_mapping_part_12_09_Test8_Google_speech_portion_partition' + str(self.args.partitioning)  + '_alpha' + str(self.args.dirichlet_alpha)
        # global_portion_file = 'slave18_mapping_part_12_16_Test9_Google_speech_portion_partition' + str(self.args.partitioning)  + '_alpha' + str(self.args.dirichlet_alpha)

        #Cifar10
        # global_portion_file = 'slave18_mapping_part_12_09_Test3_Cifar10_portion_partition' + str(self.args.partitioning)  + '_alpha' + str(self.args.dirichlet_alpha) 
        global_portion_file = 'slave18_mapping_part_12_10_Test4_Cifar10_portion_partition' + str(self.args.partitioning)  + '_alpha' + str(self.args.dirichlet_alpha)
        # global_portion_file = 'slave18_mapping_part_12_13_Test5_Cifar10_portion_partition' + str(self.args.partitioning)  + '_alpha' + str(self.args.dirichlet_alpha)
        # global_portion_file = 'slave18_mapping_part_12_15_Test6_Cifar10_portion_partition' + str(self.args.partitioning)  + '_alpha' + str(self.args.dirichlet_alpha)

        #EMNIST
        # global_portion_file = 'slave18_mapping_part_12_10_Test3_EMNIST_portion_partition' + str(self.args.partitioning) + '_alpha' + str(self.args.dirichlet_alpha)


        folder = os.path.join(args.log_path, 'metadata', args.data_set, 'data_mappings')
        #上面的文件夹理论上是MAIN_PATH/core/evals/metadata/cifar10/data_mappings  
        if not os.path.isdir(folder):                                       
            Path(folder).mkdir(parents=True, exist_ok=True)

        glo_portion_file = os.path.join(folder, global_portion_file)             
        if args.this_rank != 1:                                           
            while (not os.path.exists(glo_portion_file)):             
                time.sleep(120)


        if os.path.exists(glo_portion_file):                           
            with open(glo_portion_file, 'rb') as fin3:                   
                logging.info(f'Loading partitioning from file {global_portion_file} for testing')
                
                global_portion1 = pickle.load(fin3)                


        targets = self.getTargets()                                         
        keyDir = {key: int(key) for i, key in enumerate(targets.keys())}   

        max_c5 = 0
        for c5 in list(targets.keys()):                                 
            if global_portion1[keyDir[c5]] > global_portion1[max_c5]:
                max_c5 = keyDir[c5]

        test_partition = [[] for i in range(numOfLabels)]

        average_portion = 1/numOfLabels

        local_portion = [g*(average_portion/global_portion1[max_c5]) for g in global_portion1] 
       

#        print(f'local portion is {local_portion}')
#        print(f'global_portion1 uniform2 is {global_portion1}')

        
        for c3 in list(targets.keys()): 
            indexes =  targets[keyDir[c3]]
            self.rng.shuffle(indexes)
            test_partition[keyDir[c3]].extend(indexes)

        self.partitions = [[] for l in range(num_clients)]
            
        start = np.zeros(numOfLabels, dtype=int)    
        start1 = np.zeros(numOfLabels, dtype=int)   
        for o in range(num_clients):          
            
            # start = np.zeros(numOfLabels, dtype=int)    
            count = 0
            for c4 in list(targets.keys()):        
                end = min(start[keyDir[c4]]+int(len(test_partition[keyDir[c4]]) / num_clients), len(test_partition[keyDir[c4]])) 
                count = count + (end-start[keyDir[c4]]) 
                start[keyDir[c4]] = start[keyDir[c4]] + int(len(test_partition[keyDir[c4]]) / num_clients)


            # start1 = np.zeros(numOfLabels, dtype=int)  
            for c4 in list(targets.keys()):  
#                print(f'start is {int(len(test_partition[keyDir[c4]]) / num_clients)}')        
                end1 = min(start1[keyDir[c4]]+int(len(test_partition[keyDir[c4]]) / num_clients), len(test_partition[keyDir[c4]])) 
#                end = int(end) 
#                print(f'start[keyDir[c4]] is {start[keyDir[c4]]}') 
#                print(f'keyDir[c4] is {keyDir[c4]}')                                  
#                print(f'end is {end}')   
#                print(f'local portion for uniform testing is {local_portion}')
                count_c4 = end1 - start1[keyDir[c4]]-1
                sample_number = min(count_c4, int(count*local_portion[keyDir[c4]]))


                set_c4 = self.rng.sample(test_partition[keyDir[c4]][start1[keyDir[c4]]: end1], sample_number)

                self.partitions[o].extend(set_c4)

                start1[keyDir[c4]] = start1[keyDir[c4]] + int(len(test_partition[keyDir[c4]]) / num_clients)



#                self.partitions[o].extend(test_partition[keyDir[c4]][start[keyDir[c4]]: end])
#                start[keyDir[c4]] = start[keyDir[c4]] + int(len(test_partition[keyDir[c4]]) / num_clients)                    
    #--------------------------------------------------------------







    def custom_partition(self, num_clients, ratio=1.0):                    
        logging.info(f"Start custom_partition!")
        # custom partition                                                
        numOfLabels = self.getNumOfLabels()                                
        #update the data length to account for the ratio
        data_len = min(self.getDataLen(), int(self.getDataLen() * ratio))  
        sizes = [1.0 / num_clients for _ in range(num_clients)]            

        #get # of samples per worker
        #set the number of samples per worker                      
        self.usedSamples = self.args.used_samples if self.args.used_samples >= 0 else (self.args.batch_size + 1)  
        # get number of samples per worker
        if self.usedSamples <= 0:                                          
            self.usedSamples = int(data_len / num_clients)

        #Verify if the custom client partitioning exists
        num_class = numOfLabels
        num_remove_classes = 0
        if self.args.filter_class > 0:                                      
            num_remove_classes = self.args.filter_class
        elif self.args.filter_class_ratio > 0:                              
            num_remove_classes = round(numOfLabels * (1 - self.args.filter_class_ratio))
        num_class -= num_remove_classes                                    


        # filename = 'training_mapping_part_11_06_Test3_raw_' + str(self.args.partitioning) + '_clients' + str(num_clients) + '_data' + str(data_len) + '_labels'\
        #             + str(num_class) + '_samples' + str(self.usedSamples) 

        # filename = 'training_mapping_part_11_06_Test4_raw_' + str(self.args.partitioning) + '_clients' + str(num_clients) + '_data' + str(data_len) + '_labels'\
        #             + str(num_class) + '_samples' + str(self.usedSamples) + '_alpha' + str(self.args.dirichlet_alpha) 

        # filename = 'training_mapping_part_11_06_Test5_raw_' + str(self.args.partitioning) + '_clients' + str(num_clients) + '_data' + str(data_len) + '_labels'\
        #             + str(num_class) + '_samples' + str(self.usedSamples) + '_alpha' + str(self.args.dirichlet_alpha)
        
        #EMNIST
        # filename = 'training_mapping_part_11_27_Test_Emnist1_raw_partition' + str(self.args.partitioning) + '_clients' + str(num_clients) + '_data' + str(data_len) + '_labels'\
        #             + str(num_class) + '_samples' + str(self.usedSamples) + '_alpha' + str(self.args.dirichlet_alpha)

        # filename = 'training_mapping_part_11_28_Test_Emnist2_raw_partition' + str(self.args.partitioning) + '_clients' + str(num_clients) + '_data' + str(data_len) + '_labels'\
        #             + str(num_class) + '_samples' + str(self.usedSamples) + '_alpha' + str(self.args.dirichlet_alpha)

        #Cifar10
        # filename = 'training_mapping_part_11_29_Test1_Cifar10_raw_partition' + str(self.args.partitioning) + '_clients' + str(num_clients) + '_data' + str(data_len) + '_labels'\
        #              + str(num_class) + '_samples' + str(self.usedSamples) + '_alpha' + str(self.args.dirichlet_alpha)        
        # filename = 'training_mapping_part_11_29_Test2_Cifar10_raw_partition' + str(self.args.partitioning) + '_clients' + str(num_clients) + '_data' + str(data_len) + '_labels'\
        #             + str(num_class) + '_samples' + str(self.usedSamples) + '_alpha' + str(self.args.dirichlet_alpha) 




        #Google_speech
        # filename = 'slave18_training_mapping_part_12_07_Test6_Google_speech_raw_partition' + str(self.args.partitioning) + '_clients' + str(num_clients) + '_data' + str(data_len) + '_labels'\
        #             + str(num_class) + '_samples' + str(self.usedSamples) + '_alpha' + str(self.args.dirichlet_alpha)
        # filename = 'slave18_training_mapping_part_12_08_Test7_Google_speech_raw_partition' + str(self.args.partitioning) + '_clients' + str(num_clients) + '_data' + str(data_len) + '_labels'\
        #             + str(num_class) + '_samples' + str(self.usedSamples) + '_alpha' + str(self.args.dirichlet_alpha)
        # filename = 'slave18_training_mapping_part_12_09_Test8_Google_speech_raw_partition' + str(self.args.partitioning) + '_clients' + str(num_clients) + '_data' + str(data_len) + '_labels'\
        #             + str(num_class) + '_samples' + str(self.usedSamples) + '_alpha' + str(self.args.dirichlet_alpha)
        # filename = 'slave18_training_mapping_part_12_16_Test9_Google_speech_raw_partition' + str(self.args.partitioning) + '_clients' + str(num_clients) + '_data' + str(data_len) + '_labels'\
                    # + str(num_class) + '_samples' + str(self.usedSamples) + '_alpha' + str(self.args.dirichlet_alpha)



        # filename = 'slave18_balanced_training_mapping_part_12_07_Test6_Google_speech_raw_partition' + str(self.args.partitioning) + '_clients' + str(num_clients) + '_data' + str(data_len) + '_labels'\
        #             + str(num_class) + '_samples' + str(self.usedSamples) + '_alpha' + str(self.args.dirichlet_alpha)        


        #Cifar10
        # filename = 'slave18_training_mapping_part_12_09_Test3_Cifar10_raw_partition' + str(self.args.partitioning) + '_clients' + str(num_clients) + '_data' + str(data_len) + '_labels'\
        #             + str(num_class) + '_samples' + str(self.usedSamples) + '_alpha' + str(self.args.dirichlet_alpha)
        filename = 'slave18_training_mapping_part_12_10_Test4_Cifar10_raw_partition' + str(self.args.partitioning) + '_clients' + str(num_clients) + '_data' + str(data_len) + '_labels'\
                    + str(num_class) + '_samples' + str(self.usedSamples) + '_alpha' + str(self.args.dirichlet_alpha)
        # filename = 'slave18_training_mapping_part_12_13_Test5_Cifar10_raw_partition' + str(self.args.partitioning) + '_clients' + str(num_clients) + '_data' + str(data_len) + '_labels'\
        #             + str(num_class) + '_samples' + str(self.usedSamples) + '_alpha' + str(self.args.dirichlet_alpha)
        # filename = 'slave18_training_mapping_part_12_15_Test6_Cifar10_raw_partition' + str(self.args.partitioning) + '_clients' + str(num_clients) + '_data' + str(data_len) + '_labels'\
        #             + str(num_class) + '_samples' + str(self.usedSamples) + '_alpha' + str(self.args.dirichlet_alpha)



        # filename = 'slave18_balanced_training_mapping_part_12_10_Test4_Cifar10_raw_partition' + str(self.args.partitioning) + '_clients' + str(num_clients) + '_data' + str(data_len) + '_labels'\
        #             + str(num_class) + '_samples' + str(self.usedSamples) + '_alpha' + str(self.args.dirichlet_alpha)   


        #EMNIST
        # filename = 'slave18_training_mapping_part_12_10_Test3_EMNIST_raw_partition' + str(self.args.partitioning) + '_clients' + str(num_clients) + '_data' + str(data_len) + '_labels'\
        #             + str(num_class) + '_samples' + str(self.usedSamples) + '_alpha' + str(self.args.dirichlet_alpha)

        
        #FashionMNIST
        # filename = 'slave18_training_mapping_part_2025_06_19_Test3_FashionMNIST_raw_partition' + str(self.args.partitioning) + '_clients' + str(num_clients) + '_data' + str(data_len) + '_labels'\
        #             + str(num_class) + '_samples' + str(self.usedSamples) + '_alpha' + str(self.args.dirichlet_alpha)



        folder = os.path.join(args.log_path, 'metadata', args.data_set, 'data_mappings')
        if not os.path.isdir(folder):                                       
            Path(folder).mkdir(parents=True, exist_ok=True)

        custom_mapping_file = os.path.join(folder, filename)                
        if args.this_rank != 1:                                             
            while (not os.path.exists(custom_mapping_file)):               
                time.sleep(120)




        num_sample = 0

        if os.path.exists(custom_mapping_file):                            
            with open(custom_mapping_file, 'rb') as fin:                   
                logging.info(f'Loading partitioning from file {filename}')
                self.partitions = pickle.load(fin)                           
                for i, part in enumerate(self.partitions):                            
                    labels = [self.indexToLabel[index] for index in part]   

                    num_sample = num_sample + len(part)
                logging.info(f'Total number of samples of the method without data reshape is {num_sample}.')

                    #count_elems = Counter(labels)
                    #logging.info(f'part {i} len: {len(part)} labels: {count_elems.keys()} count: {count_elems.values()}')
            return  

        #get targets
        targets = self.getTargets()                                         
        keyDir = {key: int(key) for i, key in enumerate(targets.keys())}     

        keyLength = [0] * numOfLabels
        for key in keyDir.keys():                                             
            keyLength[keyDir[key]] = len(targets[key])                       
           
        logging.info(f"Custom partitioning {self.args.partitioning} data, {data_len} samples of {numOfLabels}:{num_class} labels on {num_clients} clients, use {self.usedSamples} sample per client ...")
                                                                             
        ratioOfClassWorker = self.create_mapping(sizes)                       


        if ratioOfClassWorker is None:                                       
            return self.uniform_partition(num_clients=num_clients)           

        sumRatiosPerClass = np.sum(ratioOfClassWorker, axis=1)               
        for worker in range(len(sizes)):
            ratioOfClassWorker[worker, :] = ratioOfClassWorker[worker, :] / float(sumRatiosPerClass[worker])   

        # classPerWorker -> Rows are workers and cols are classes
        tempClassPerWorker = np.zeros([len(sizes), numOfLabels])

        


        # split the classes
        for worker in range(len(sizes)):                                    
#            logging.info(f"in the for loop!")             
            self.partitions.append([])                                        
            # enumerate the ratio of classes it should take
#            logging.info(f"len of targets is {len(list(targets.keys()))}!") 
            for c in list(targets.keys()):                                   
                takeLength = int(self.usedSamples * ratioOfClassWorker[worker][keyDir[c]])   
                takeLength = min(takeLength, keyLength[keyDir[c]])            
#                logging.info(f"before index!") 
                indexes = self.rng.sample(targets[c], takeLength)            
                self.partitions[-1] += indexes                               
                
#                logging.info(f"after partition!") 
                labels = [self.indexToLabel[index] for index in self.partitions[-1]]  
                count_elems = Counter(labels)                                
                tempClassPerWorker[worker][keyDir[c]] += takeLength          
#            logging.info(f"after second for loop!")                         
            #logging.info(f'worker: {worker} created partition len: {len(self.partitions[-1])} class/worker: {sum(tempClassPerWorker[worker])} labels:{tempClassPerWorker[worker]} ratios: {ratioOfClassWorker[worker]}')

        del tempClassPerWorker                                               
#        logging.info(f"Before create custom_mapping_file!")               
        #save the partitions as pickle file                                   
        if not os.path.exists(custom_mapping_file):                           
            with open(custom_mapping_file, 'wb') as fout:
                 pickle.dump(self.partitions, fout)                                       
            logging.info(f'Storing partitioning to file {filename}')          

        
















    def custom_partition2(self, num_clients, ratio=1.0):                   
        logging.info(f"Start custom_partition!")
        # custom partition                                                  
        numOfLabels = self.getNumOfLabels()                               
        #update the data length to account for the ratio
        data_len = min(self.getDataLen(), int(self.getDataLen() * ratio))   
        sizes = [1.0 / num_clients for _ in range(num_clients)]            

        #get # of samples per worker
        #set the number of samples per worker                      
        self.usedSamples = self.args.used_samples if self.args.used_samples >= 0 else (self.args.batch_size + 1)  
        # get number of samples per worker
        if self.usedSamples <= 0:                                           
            self.usedSamples = int(data_len / num_clients)

        #Verify if the custom client partitioning exists
        num_class = numOfLabels
        num_remove_classes = 0
        if self.args.filter_class > 0:                                      
            num_remove_classes = self.args.filter_class
        elif self.args.filter_class_ratio > 0:                              
            num_remove_classes = round(numOfLabels * (1 - self.args.filter_class_ratio))
        num_class -= num_remove_classes                                    


        
        # filename = 'training_mapping_part_11_03_Test1_' + str(self.args.partitioning) + '_clients' + str(num_clients) + '_data' + str(data_len) + '_labels'\
        #            + str(num_class) + '_samples' + str(self.usedSamples)    

        # count_matrix_file = 'mapping_part_11_03_Test1_matrix_partition' + str(self.args.partitioning)    

        # global_portion_file = 'mapping_part_11_03_Test1_portion_partition' + str(self.args.partitioning) 

#        filename = 'training_mapping_part_11_05_Test2_' + str(self.args.partitioning) + '_clients' + str(num_clients) + '_data' + str(data_len) + '_labels'\
#                    + str(num_class) + '_samples' + str(self.usedSamples)    

#        count_matrix_file = 'mapping_part_11_05_Test2_matrix_partition' + str(self.args.partitioning)    

#        global_portion_file = 'mapping_part_11_05_Test2_portion_partition' + str(self.args.partitioning) 

        # filename = 'training_mapping_part_11_06_Test3_' + str(self.args.partitioning) + '_clients' + str(num_clients) + '_data' + str(data_len) + '_labels'\
        #             + str(num_class) + '_samples' + str(self.usedSamples)    

        # count_matrix_file = 'mapping_part_11_06_Test3_matrix_partition' + str(self.args.partitioning)    

        # global_portion_file = 'mapping_part_11_06_Test3_portion_partition' + str(self.args.partitioning) 

        # partition_file = 'training_mapping_part_11_06_Test3_raw_' + str(self.args.partitioning) + '_clients' + str(num_clients) + '_data' + str(data_len) + '_labels'\
        #             + str(num_class) + '_samples' + str(self.usedSamples)

        # filename = 'training_mapping_part_11_06_Test4_' + str(self.args.partitioning) + '_clients' + str(num_clients) + '_data' + str(data_len) + '_labels'\
        #             + str(num_class) + '_samples' + str(self.usedSamples) + '_alpha' + str(self.args.dirichlet_alpha)    

        # count_matrix_file = 'mapping_part_11_06_Test4_matrix_partition' + str(self.args.partitioning) + '_alpha' + str(self.args.dirichlet_alpha)     

        # global_portion_file = 'mapping_part_11_06_Test4_portion_partition' + str(self.args.partitioning) + '_alpha' + str(self.args.dirichlet_alpha)  

        # partition_file = 'training_mapping_part_11_06_Test4_raw_' + str(self.args.partitioning) + '_clients' + str(num_clients) + '_data' + str(data_len) + '_labels'\
        #             + str(num_class) + '_samples' + str(self.usedSamples) + '_alpha' + str(self.args.dirichlet_alpha) 

        # filename = 'training_mapping_part_11_06_Test5_' + str(self.args.partitioning) + '_clients' + str(num_clients) + '_data' + str(data_len) + '_labels'\
        #             + str(num_class) + '_samples' + str(self.usedSamples) + '_alpha' + str(self.args.dirichlet_alpha)    

        # count_matrix_file = 'mapping_part_11_06_Test5_matrix_partition' + str(self.args.partitioning) + '_alpha' + str(self.args.dirichlet_alpha)     

        # global_portion_file = 'mapping_part_11_06_Test5_portion_partition' + str(self.args.partitioning) + '_alpha' + str(self.args.dirichlet_alpha)  

        # partition_file = 'training_mapping_part_11_06_Test5_raw_' + str(self.args.partitioning) + '_clients' + str(num_clients) + '_data' + str(data_len) + '_labels'\
        #             + str(num_class) + '_samples' + str(self.usedSamples) + '_alpha' + str(self.args.dirichlet_alpha)

        # filename = 'training_mapping_part_11_27_Test_Emnist1_partition' + str(self.args.partitioning) + '_clients' + str(num_clients) + '_data' + str(data_len) + '_labels'\
        #             + str(num_class) + '_samples' + str(self.usedSamples) + '_alpha' + str(self.args.dirichlet_alpha)    

        # count_matrix_file = 'mapping_part_11_27_Test_Emnist1_matrix_partition' + str(self.args.partitioning) + '_alpha' + str(self.args.dirichlet_alpha)     

        # global_portion_file = 'mapping_part_11_27_Test_Emnist1_portion_partition' + str(self.args.partitioning) + '_alpha' + str(self.args.dirichlet_alpha)  

        # partition_file = 'training_mapping_part_11_27_Test_Emnist1_raw_partition' + str(self.args.partitioning) + '_clients' + str(num_clients) + '_data' + str(data_len) + '_labels'\
        #             + str(num_class) + '_samples' + str(self.usedSamples) + '_alpha' + str(self.args.dirichlet_alpha)

        # filename = 'training_mapping_part_11_28_Test_Emnist2_partition' + str(self.args.partitioning) + '_clients' + str(num_clients) + '_data' + str(data_len) + '_labels'\
        #             + str(num_class) + '_samples' + str(self.usedSamples) + '_alpha' + str(self.args.dirichlet_alpha)    

        # count_matrix_file = 'mapping_part_11_28_Test_Emnist2_matrix_partition' + str(self.args.partitioning) + '_alpha' + str(self.args.dirichlet_alpha)     

        # global_portion_file = 'mapping_part_11_28_Test_Emnist2_portion_partition' + str(self.args.partitioning) + '_alpha' + str(self.args.dirichlet_alpha)  

        # partition_file = 'training_mapping_part_11_28_Test_Emnist2_raw_partition' + str(self.args.partitioning) + '_clients' + str(num_clients) + '_data' + str(data_len) + '_labels'\
        #             + str(num_class) + '_samples' + str(self.usedSamples) + '_alpha' + str(self.args.dirichlet_alpha)

        #Tiny-imagenet
        # filename = 'training_mapping_part_11_28_Test_Tiny_imagenet1_partition' + str(self.args.partitioning) + '_clients' + str(num_clients) + '_data' + str(data_len) + '_labels'\
        #              + str(num_class) + '_samples' + str(self.usedSamples) + '_alpha' + str(self.args.dirichlet_alpha)    

        # count_matrix_file = 'mapping_part_11_28_Test_Tiny_imagenet1_matrix_partition' + str(self.args.partitioning) + '_alpha' + str(self.args.dirichlet_alpha)     

        # global_portion_file = 'mapping_part_11_28_Test_Tiny_imagenet1_portion_partition' + str(self.args.partitioning) + '_alpha' + str(self.args.dirichlet_alpha)  

        # partition_file = 'training_mapping_part_11_28_Test_Tiny_imagenet1_raw_partition' + str(self.args.partitioning) + '_clients' + str(num_clients) + '_data' + str(data_len) + '_labels'\
        #              + str(num_class) + '_samples' + str(self.usedSamples) + '_alpha' + str(self.args.dirichlet_alpha)

        #Cifar10
        # filename = 'training_mapping_part_11_29_Test1_Cifar10_partition' + str(self.args.partitioning) + '_clients' + str(num_clients) + '_data' + str(data_len) + '_labels'\
        #             + str(num_class) + '_samples' + str(self.usedSamples) + '_alpha' + str(self.args.dirichlet_alpha)    

        # count_matrix_file = 'mapping_part_11_29_Test1_Cifar10_matrix_partition' + str(self.args.partitioning) + '_alpha' + str(self.args.dirichlet_alpha)     

        # global_portion_file = 'mapping_part_11_29_Test1_Cifar10_portion_partition' + str(self.args.partitioning) + '_alpha' + str(self.args.dirichlet_alpha)  

        # partition_file = 'training_mapping_part_11_29_Test1_Cifar10_raw_partition' + str(self.args.partitioning) + '_clients' + str(num_clients) + '_data' + str(data_len) + '_labels'\
        #             + str(num_class) + '_samples' + str(self.usedSamples) + '_alpha' + str(self.args.dirichlet_alpha)


        # filename = 'training_mapping_part_11_29_Test2_Cifar10_partition' + str(self.args.partitioning) + '_clients' + str(num_clients) + '_data' + str(data_len) + '_labels'\
        #             + str(num_class) + '_samples' + str(self.usedSamples) + '_alpha' + str(self.args.dirichlet_alpha)    

        # count_matrix_file = 'mapping_part_11_29_Test2_Cifar10_matrix_partition' + str(self.args.partitioning) + '_alpha' + str(self.args.dirichlet_alpha)     

        # global_portion_file = 'mapping_part_11_29_Test2_Cifar10_portion_partition' + str(self.args.partitioning) + '_alpha' + str(self.args.dirichlet_alpha)  

        # partition_file = 'training_mapping_part_11_29_Test2_Cifar10_raw_partition' + str(self.args.partitioning) + '_clients' + str(num_clients) + '_data' + str(data_len) + '_labels'\
        #             + str(num_class) + '_samples' + str(self.usedSamples) + '_alpha' + str(self.args.dirichlet_alpha)

        #FashionMNIST
        # filename = 'slave18_training_mapping_part_12_04_Test1_FashionMNIST_partition' + str(self.args.partitioning) + '_clients' + str(num_clients) + '_data' + str(data_len) + '_labels'\
        #             + str(num_class) + '_samples' + str(self.usedSamples) + '_alpha' + str(self.args.dirichlet_alpha)    

        # count_matrix_file = 'slave18_mapping_part_12_04_Test1_FashionMNIST_matrix_partition' + str(self.args.partitioning) + '_alpha' + str(self.args.dirichlet_alpha)     

        # global_portion_file = 'slave18_mapping_part_12_04_Test1_FashionMNIST_portion_partition' + str(self.args.partitioning) + '_alpha' + str(self.args.dirichlet_alpha)  

        # partition_file = 'slave18_training_mapping_part_12_04_Test1_FashionMNIST_raw_partition' + str(self.args.partitioning) + '_clients' + str(num_clients) + '_data' + str(data_len) + '_labels'\
        #             + str(num_class) + '_samples' + str(self.usedSamples) + '_alpha' + str(self.args.dirichlet_alpha)

        # filename = 'slave18_training_mapping_part_12_04_Test2_FashionMNIST_partition' + str(self.args.partitioning) + '_clients' + str(num_clients) + '_data' + str(data_len) + '_labels'\
        #             + str(num_class) + '_samples' + str(self.usedSamples) + '_alpha' + str(self.args.dirichlet_alpha)    

        # count_matrix_file = 'slave18_mapping_part_12_04_Test2_FashionMNIST_matrix_partition' + str(self.args.partitioning) + '_alpha' + str(self.args.dirichlet_alpha)     

        # global_portion_file = 'slave18_mapping_part_12_04_Test2_FashionMNIST_portion_partition' + str(self.args.partitioning) + '_alpha' + str(self.args.dirichlet_alpha)  

        # partition_file = 'slave18_training_mapping_part_12_04_Test2_FashionMNIST_raw_partition' + str(self.args.partitioning) + '_clients' + str(num_clients) + '_data' + str(data_len) + '_labels'\
        #             + str(num_class) + '_samples' + str(self.usedSamples) + '_alpha' + str(self.args.dirichlet_alpha)





        #Google_speech
        # filename = 'slave18_training_mapping_part_12_07_Test6_Google_speech_partition' + str(self.args.partitioning) + '_clients' + str(num_clients) + '_data' + str(data_len) + '_labels'\
        #             + str(num_class) + '_samples' + str(self.usedSamples) + '_alpha' + str(self.args.dirichlet_alpha)    

        # count_matrix_file = 'slave18_mapping_part_12_07_Test6_Google_speech_matrix_partition' + str(self.args.partitioning) + '_alpha' + str(self.args.dirichlet_alpha)     

        # global_portion_file = 'slave18_mapping_part_12_07_Test6_Google_speech_portion_partition' + str(self.args.partitioning) + '_alpha' + str(self.args.dirichlet_alpha)  

        # partition_file = 'slave18_training_mapping_part_12_07_Test6_Google_speech_raw_partition' + str(self.args.partitioning) + '_clients' + str(num_clients) + '_data' + str(data_len) + '_labels'\
        #             + str(num_class) + '_samples' + str(self.usedSamples) + '_alpha' + str(self.args.dirichlet_alpha)

        
        # filename = 'slave18_training_mapping_part_12_08_Test7_Google_speech_partition' + str(self.args.partitioning) + '_clients' + str(num_clients) + '_data' + str(data_len) + '_labels'\
        #             + str(num_class) + '_samples' + str(self.usedSamples) + '_alpha' + str(self.args.dirichlet_alpha)    

        # count_matrix_file = 'slave18_mapping_part_12_08_Test7_Google_speech_matrix_partition' + str(self.args.partitioning) + '_alpha' + str(self.args.dirichlet_alpha)     

        # global_portion_file = 'slave18_mapping_part_12_08_Test7_Google_speech_portion_partition' + str(self.args.partitioning) + '_alpha' + str(self.args.dirichlet_alpha)  

        # partition_file = 'slave18_training_mapping_part_12_08_Test7_Google_speech_raw_partition' + str(self.args.partitioning) + '_clients' + str(num_clients) + '_data' + str(data_len) + '_labels'\
        #             + str(num_class) + '_samples' + str(self.usedSamples) + '_alpha' + str(self.args.dirichlet_alpha)

        
        # filename = 'slave18_training_mapping_part_12_09_Test8_Google_speech_partition' + str(self.args.partitioning) + '_clients' + str(num_clients) + '_data' + str(data_len) + '_labels'\
        #             + str(num_class) + '_samples' + str(self.usedSamples) + '_alpha' + str(self.args.dirichlet_alpha)    

        # count_matrix_file = 'slave18_mapping_part_12_09_Test8_Google_speech_matrix_partition' + str(self.args.partitioning) + '_alpha' + str(self.args.dirichlet_alpha)     

        # global_portion_file = 'slave18_mapping_part_12_09_Test8_Google_speech_portion_partition' + str(self.args.partitioning) + '_alpha' + str(self.args.dirichlet_alpha)  

        # partition_file = 'slave18_training_mapping_part_12_09_Test8_Google_speech_raw_partition' + str(self.args.partitioning) + '_clients' + str(num_clients) + '_data' + str(data_len) + '_labels'\
        #             + str(num_class) + '_samples' + str(self.usedSamples) + '_alpha' + str(self.args.dirichlet_alpha)                    

        # filename = 'slave18_training_mapping_part_12_16_Test9_Google_speech_partition' + str(self.args.partitioning) + '_clients' + str(num_clients) + '_data' + str(data_len) + '_labels'\
        #             + str(num_class) + '_samples' + str(self.usedSamples) + '_alpha' + str(self.args.dirichlet_alpha)    

        # count_matrix_file = 'slave18_mapping_part_12_16_Test9_Google_speech_matrix_partition' + str(self.args.partitioning) + '_alpha' + str(self.args.dirichlet_alpha)     

        # global_portion_file = 'slave18_mapping_part_12_16_Test9_Google_speech_portion_partition' + str(self.args.partitioning) + '_alpha' + str(self.args.dirichlet_alpha)  

        # partition_file = 'slave18_training_mapping_part_12_16_Test9_Google_speech_raw_partition' + str(self.args.partitioning) + '_clients' + str(num_clients) + '_data' + str(data_len) + '_labels'\
        #             + str(num_class) + '_samples' + str(self.usedSamples) + '_alpha' + str(self.args.dirichlet_alpha)


        #Cifar10 
        # filename = 'slave18_training_mapping_part_12_09_Test3_Cifar10_partition' + str(self.args.partitioning) + '_clients' + str(num_clients) + '_data' + str(data_len) + '_labels'\
        #             + str(num_class) + '_samples' + str(self.usedSamples) + '_alpha' + str(self.args.dirichlet_alpha)    

        # count_matrix_file = 'slave18_mapping_part_12_09_Test3_Cifar10_matrix_partition' + str(self.args.partitioning) + '_alpha' + str(self.args.dirichlet_alpha)     

        # global_portion_file = 'slave18_mapping_part_12_09_Test3_Cifar10_portion_partition' + str(self.args.partitioning) + '_alpha' + str(self.args.dirichlet_alpha)  

        # partition_file = 'slave18_training_mapping_part_12_09_Test3_Cifar10_raw_partition' + str(self.args.partitioning) + '_clients' + str(num_clients) + '_data' + str(data_len) + '_labels'\
        #             + str(num_class) + '_samples' + str(self.usedSamples) + '_alpha' + str(self.args.dirichlet_alpha)        

        
        filename = 'slave18_training_mapping_part_12_10_Test4_Cifar10_partition' + str(self.args.partitioning) + '_clients' + str(num_clients) + '_data' + str(data_len) + '_labels'\
                    + str(num_class) + '_samples' + str(self.usedSamples) + '_alpha' + str(self.args.dirichlet_alpha)    

        count_matrix_file = 'slave18_mapping_part_12_10_Test4_Cifar10_matrix_partition' + str(self.args.partitioning) + '_alpha' + str(self.args.dirichlet_alpha)     

        global_portion_file = 'slave18_mapping_part_12_10_Test4_Cifar10_portion_partition' + str(self.args.partitioning) + '_alpha' + str(self.args.dirichlet_alpha)  

        partition_file = 'slave18_training_mapping_part_12_10_Test4_Cifar10_raw_partition' + str(self.args.partitioning) + '_clients' + str(num_clients) + '_data' + str(data_len) + '_labels'\
                    + str(num_class) + '_samples' + str(self.usedSamples) + '_alpha' + str(self.args.dirichlet_alpha)

        # filename = 'slave18_training_mapping_part_12_13_Test5_Cifar10_partition' + str(self.args.partitioning) + '_clients' + str(num_clients) + '_data' + str(data_len) + '_labels'\
        #             + str(num_class) + '_samples' + str(self.usedSamples) + '_alpha' + str(self.args.dirichlet_alpha)    

        # count_matrix_file = 'slave18_mapping_part_12_13_Test5_Cifar10_matrix_partition' + str(self.args.partitioning) + '_alpha' + str(self.args.dirichlet_alpha)     

        # global_portion_file = 'slave18_mapping_part_12_13_Test5_Cifar10_portion_partition' + str(self.args.partitioning) + '_alpha' + str(self.args.dirichlet_alpha)  

        # partition_file = 'slave18_training_mapping_part_12_13_Test5_Cifar10_raw_partition' + str(self.args.partitioning) + '_clients' + str(num_clients) + '_data' + str(data_len) + '_labels'\
        #             + str(num_class) + '_samples' + str(self.usedSamples) + '_alpha' + str(self.args.dirichlet_alpha)
        
        # filename = 'slave18_training_mapping_part_12_15_Test6_Cifar10_partition' + str(self.args.partitioning) + '_clients' + str(num_clients) + '_data' + str(data_len) + '_labels'\
        #             + str(num_class) + '_samples' + str(self.usedSamples) + '_alpha' + str(self.args.dirichlet_alpha)    

        # count_matrix_file = 'slave18_mapping_part_12_15_Test6_Cifar10_matrix_partition' + str(self.args.partitioning) + '_alpha' + str(self.args.dirichlet_alpha)     

        # global_portion_file = 'slave18_mapping_part_12_15_Test6_Cifar10_portion_partition' + str(self.args.partitioning) + '_alpha' + str(self.args.dirichlet_alpha)  

        # partition_file = 'slave18_training_mapping_part_12_15_Test6_Cifar10_raw_partition' + str(self.args.partitioning) + '_clients' + str(num_clients) + '_data' + str(data_len) + '_labels'\
        #             + str(num_class) + '_samples' + str(self.usedSamples) + '_alpha' + str(self.args.dirichlet_alpha)

        #EMNIST
        # filename = 'slave18_training_mapping_part_12_10_Test3_EMNIST_partition' + str(self.args.partitioning) + '_clients' + str(num_clients) + '_data' + str(data_len) + '_labels'\
        #             + str(num_class) + '_samples' + str(self.usedSamples) + '_alpha' + str(self.args.dirichlet_alpha)    

        # count_matrix_file = 'slave18_mapping_part_12_10_Test3_EMNIST_matrix_partition' + str(self.args.partitioning) + '_alpha' + str(self.args.dirichlet_alpha)     

        # global_portion_file = 'slave18_mapping_part_12_10_Test3_EMNIST_portion_partition' + str(self.args.partitioning) + '_alpha' + str(self.args.dirichlet_alpha)  

        # partition_file = 'slave18_training_mapping_part_12_10_Test3_EMNIST_raw_partition' + str(self.args.partitioning) + '_clients' + str(num_clients) + '_data' + str(data_len) + '_labels'\
        #             + str(num_class) + '_samples' + str(self.usedSamples) + '_alpha' + str(self.args.dirichlet_alpha)


        #FashionMNIST
        # filename = 'slave18_training_mapping_part_2025_06_19_Test3_FashionMNIST_partition' + str(self.args.partitioning) + '_clients' + str(num_clients) + '_data' + str(data_len) + '_labels'\
        #             + str(num_class) + '_samples' + str(self.usedSamples) + '_alpha' + str(self.args.dirichlet_alpha)    

        # count_matrix_file = 'slave18_mapping_part_2025_06_19_Test3_FashionMNIST_matrix_partition' + str(self.args.partitioning) + '_alpha' + str(self.args.dirichlet_alpha)     

        # global_portion_file = 'slave18_mapping_part_2025_06_19_Test3_FashionMNIST_portion_partition' + str(self.args.partitioning) + '_alpha' + str(self.args.dirichlet_alpha)  

        # partition_file = 'slave18_training_mapping_part_2025_06_19_Test3_FashionMNIST_raw_partition' + str(self.args.partitioning) + '_clients' + str(num_clients) + '_data' + str(data_len) + '_labels'\
        #             + str(num_class) + '_samples' + str(self.usedSamples) + '_alpha' + str(self.args.dirichlet_alpha)




        folder = os.path.join(args.log_path, 'metadata', args.data_set, 'data_mappings')

        if not os.path.isdir(folder):                                     
            Path(folder).mkdir(parents=True, exist_ok=True)

        custom_mapping_file = os.path.join(folder, filename)                
        if args.this_rank != 1:                                             
            while (not os.path.exists(custom_mapping_file)):                
                time.sleep(120)

        #-------------------------------------------

        targets = self.getTargets()                                          
        keyDir = {key: int(key) for i, key in enumerate(targets.keys())}      

        mapping_matrix_file = os.path.join(folder, count_matrix_file)                  #count_matrix_file      
        if args.this_rank != 1:                                       
            while (not os.path.exists(mapping_matrix_file)):             
                time.sleep(120)

        raw_partition_file = os.path.join(folder, partition_file)                     
        if args.this_rank != 1:                                       
            while (not os.path.exists(raw_partition_file)):             
                time.sleep(120)        
        #-------------------------------------------

        if os.path.exists(custom_mapping_file):                            
            with open(custom_mapping_file, 'rb') as fin:                   

                logging.info(f'Loading partitioning from file {filename}')

                # self.partitions = pickle.load(fin)                           

                self.total_partition = pickle.load(fin)
                self.partitions = self.total_partition


        #-----------------------------------------         
        glo_portion_file = os.path.join(folder, global_portion_file)             
        if args.this_rank != 1:                                           
            while (not os.path.exists(glo_portion_file)):             
                time.sleep(120)
        
        if (args.similarity == 1):
            if os.path.exists(glo_portion_file):                           
                with open(glo_portion_file, 'rb') as fing:                   
                    logging.info(f'Loading partitioning from file {global_portion_file} to obtain global portion')
                    
                    self.global_portion = pickle.load(fing)  

        #-----------------------------------------


        #-----------------------------------------
        if os.path.exists(mapping_matrix_file):             
            with open(mapping_matrix_file, 'rb') as fin1:                    
                logging.info(f'Loading partitioning from file {count_matrix_file}')
                mapping_matrix1 = pickle.load(fin1)                     

            final_partitions = [[] for i in range(num_clients)]
                
            num_sample1 = 0                                                      
 
            for n in range(num_clients):
                for m in list(targets.keys()):                                   
                    c2 = keyDir[m]
#                    print(f'self.partitions[n] is {self.partitions[n]}')
#                    print(f'c2 is {keyDir[m]}')                    
#                    print(f'mapping_matrix1[n][c2] is {mapping_matrix1[n][c2]}')      
#                    print(f'len(self.partitions[n][c2]) - mapping_matrix1[n][c2] is {len(self.partitions[n][c2]) - mapping_matrix1[n][c2]}')              
                    final_partitions[n].extend(self.rng.sample(self.partitions[n][c2], int(mapping_matrix1[n][c2])))
                    num_sample1 = num_sample1 + len(self.partitions[n][c2])      

            self.partitions = final_partitions

            self.partition_matrix = mapping_matrix1
            logging.info(f"Total number of samples (data regulation method) before data reshape is {num_sample1}.")
            logging.info(f"Total number of samples (data regulation method) after data reshape is {np.sum(self.partition_matrix)}.")            
        #------------------------------------------




#                for i, part in enumerate(self.partitions):                        
#                    labels = [self.indexToLabel[index] for index in part]    
#                    #count_elems = Counter(labels)
#                    #logging.info(f'part {i} len: {len(part)} labels: {count_elems.keys()} count: {count_elems.values()}')

            return  








        #get targets       
#        targets = self.getTargets()                                           
#        keyDir = {key: int(key) for i, key in enumerate(targets.keys())}    

        keyLength = [0] * numOfLabels
        for key in keyDir.keys():                                            
            keyLength[keyDir[key]] = len(targets[key])                       

        logging.info(f"Custom partitioning {self.args.partitioning} data, {data_len} samples of {numOfLabels}:{num_class} labels on {num_clients} clients, use {self.usedSamples} sample per client ...")
                                                                             
        ratioOfClassWorker = self.create_mapping(sizes)                       

#        logging.info(f"After create_mapping!")                               
        if ratioOfClassWorker is None:                                        
            return self.uniform_partition(num_clients=num_clients)           

        sumRatiosPerClass = np.sum(ratioOfClassWorker, axis=1)              
        for worker in range(len(sizes)):
            ratioOfClassWorker[worker, :] = ratioOfClassWorker[worker, :] / float(sumRatiosPerClass[worker])   

        # classPerWorker -> Rows are workers and cols are classes
        tempClassPerWorker = np.zeros([len(sizes), numOfLabels])

          





        #--------------------------
        global_count = np.zeros((num_clients, numOfLabels))
        global_portion = np.zeros((1, numOfLabels))        
        #keyDir = {key: int(key) for i, key in enumerate(targets.keys())}    
        for worker in range(num_clients):
#            for label in range(numOfLabels):                                
            for label, label_key in enumerate(targets.keys()):
                global_count[worker][keyDir[label_key]] = min(int(self.usedSamples * ratioOfClassWorker[worker][keyDir[label_key]]), keyLength[keyDir[label_key]])


        global_count_total = np.sum(global_count, axis = 0)                   
        global_portion = global_count_total/np.sum(global_count_total)        


        logging.info(f"Current partition is {self.args.partitioning}.")
        logging.info(f"Global portion before data reshape is {global_portion}.")
        logging.info(f"Total examples bofore data reshape is {np.sum(global_count_total)}.")


        count_matrix = np.zeros((num_clients, numOfLabels))                 

        # logging.info(f"global_portion is {global_portion}.")              
#        logging.info(f"Total number of samples after regulation is {np.sum(global_count_total)}.")        
        #--------------------------


        fi_partition = []
        raw_partition = []
        # split the classes
        for worker in range(len(sizes)):                                   
            self.partitions.append([])                          
            # enumerate the ratio of classes it should take    
            
            #--------------------------
            fi_partition.append([])
            raw_partition.append([])

            for la in range(numOfLabels):
                fi_partition[-1].append([])


#            count_vector = global_portion                                     

            count_vector = np.zeros((numOfLabels, 1))

            max_c = 0

            for c in list(targets.keys()):                                                 
                takeLength = int(self.usedSamples * ratioOfClassWorker[worker][keyDir[c]])   
                takeLength = min(takeLength, keyLength[keyDir[c]])                           

                if takeLength < 1:
                    count_vector[keyDir[c]] = 0  
                else:
                    count_vector[keyDir[c]] = global_portion[keyDir[c]]/ratioOfClassWorker[worker][keyDir[c]]                     
            
                
                
                if count_vector[keyDir[c]] >= count_vector[max_c]:
                    max_c = keyDir[c]
            #--------------------------

#                logging.info(f"before index!") 
                indexes = self.rng.sample(targets[c], takeLength)             


                #---------------------------------------------
#                self.partitions[-1] += indexes                          
#                fi_partition[-1]+= [indexes]
                fi_partition[-1][keyDir[c]]+= indexes
                raw_partition[-1] += indexes
                #---------------------------------------------

#                logging.info(f"after partition!") 

                 
#                labels = [self.indexToLabel[index] for index in self.partitions[-1]] 
#                count_elems = Counter(labels)                                 

                tempClassPerWorker[worker][keyDir[c]] += takeLength        


#            logging.info(f"after second for loop!")                          
            #logging.info(f'worker: {worker} created partition len: {len(self.partitions[-1])} class/worker: {sum(tempClassPerWorker[worker])} labels:{tempClassPerWorker[worker]} ratios: {ratioOfClassWorker[worker]}')
            
            #----------------------------------            
            max_num = int(self.usedSamples * ratioOfClassWorker[worker][max_c])
            max_num = min(max_num, keyLength[max_c])

            for c1 in list(targets.keys()):                                   
                takeLength1 = int(self.usedSamples * ratioOfClassWorker[worker][keyDir[c1]])   
                takeLength1 = min(takeLength1, keyLength[keyDir[c1]])                      
                if count_vector[keyDir[c1]] > 0:
                    count_matrix[worker][keyDir[c1]] = int(min((global_portion[keyDir[c1]]/global_portion[max_c])*max_num, takeLength1))

#                    logging.info(f'partition and matrix test global_portion[keyDir[c1]]/global_portion[max_c])*max_num is {(global_portion[keyDir[c1]]/global_portion[max_c])*max_num}')                        
#                    logging.info(f'partition and matrix test takeLength1 is {takeLength1}')


#                    count_matrix[worker][keyDir[c1]]  = int(min(count_matrix[worker][keyDir[c1]] , keyLength[keyDir[c1]]))   
            #----------------------------------         

        self.total_partition = fi_partition

        global_count_total1 = np.sum(count_matrix, axis = 0)                       
        global_portion1 = global_count_total1/np.sum(global_count_total1)          

        logging.info(f"Global portion after data reshape is {global_portion1}.")                                    
        logging.info(f"Total number of samples after data reshape is {np.sum(global_count_total1)}.")              

        #-----------------------------------

        fi_partition2 = [[] for i in range(num_clients)]
                
        for n in range(num_clients):
            for m in list(targets.keys()):                                   
                c6 = keyDir[m] 
#                logging.info(f'partition and matrix test len(fi_partition[n][c6]) is {len(fi_partition[n][c6])}')                        
#                logging.info(f'partition and matrix test int(count_matrix[n][c6]) is {int(count_matrix[n][c6])}') 
                fi_partition2[n].extend(self.rng.sample(fi_partition[n][c6], int(count_matrix[n][c6])))

        self.partitions = fi_partition2
        #-----------------------------------
        # logging.info(f"Total number of samples after regulation is {np.sum(count_matrix)}.")  



        del tempClassPerWorker                                              
#        logging.info(f"Before create custom_mapping_file!")               
        #save the partitions as pickle file                                   
        if not os.path.exists(custom_mapping_file):                                        
            with open(custom_mapping_file, 'wb') as fout:
#                 pickle.dump(self.partitions, fout)                       
                  pickle.dump(fi_partition, fout)                           
            logging.info(f'Storing partitioning to file {filename}')          


       
        if not os.path.exists(mapping_matrix_file):                                            
            with open(mapping_matrix_file, 'wb') as fout1:
                 pickle.dump(count_matrix, fout1)                                           
            logging.info(f'Storing partitioning to file {mapping_matrix_file}')      


        glo_portion_file = os.path.join(folder, global_portion_file)          
        if not os.path.exists(glo_portion_file):                                         
            with open(glo_portion_file, 'wb') as fout2:
                 pickle.dump(global_portion, fout2)                                           
            logging.info(f'Storing partitioning to file {global_portion_file}')    

#        logging.info(f"global_portion before writing is {global_portion}.")
#        with open(glo_portion_file, 'rb') as fin4:                   
#            logging.info(f'Loading partitioning from file {global_portion_file} for pickle testing')
                
#            global_portion2 = pickle.load(fin4)  
#            logging.info(f"global_portion2 is {global_portion2}.") 



        if not os.path.exists(raw_partition_file):                                         
            with open(raw_partition_file, 'wb') as fout3:
                 pickle.dump(raw_partition, fout3)                                           
            logging.info(f'Storing partitioning to file {partition_file}') 






    def custom_partition2_1(self, num_clients, ratio=1.0):                  
        logging.info(f"Start custom_partition!")
        # custom partition                                              
        numOfLabels = self.getNumOfLabels()                            
        #update the data length to account for the ratio
        data_len = min(self.getDataLen(), int(self.getDataLen() * ratio))   
        sizes = [1.0 / num_clients for _ in range(num_clients)]             

        #get # of samples per worker
        #set the number of samples per worker                 
        self.usedSamples = self.args.used_samples if self.args.used_samples >= 0 else (self.args.batch_size + 1)  
        # get number of samples per worker
        if self.usedSamples <= 0:                                        
            self.usedSamples = int(data_len / num_clients)

        #Verify if the custom client partitioning exists
        num_class = numOfLabels
        num_remove_classes = 0
        if self.args.filter_class > 0:                                      
            num_remove_classes = self.args.filter_class
        elif self.args.filter_class_ratio > 0:                              
            num_remove_classes = round(numOfLabels * (1 - self.args.filter_class_ratio))
        num_class -= num_remove_classes                              


        



        #Google_speech
        filename = 'slave18_training_mapping_part_12_07_Test6_Google_speech_partition' + str(self.args.partitioning) + '_clients' + str(num_clients) + '_data' + str(data_len) + '_labels'\
                    + str(num_class) + '_samples' + str(self.usedSamples) + '_alpha' + str(self.args.dirichlet_alpha)    

        # count_matrix_file = 'slave18_mapping_part_12_07_Test6_Google_speech_matrix_partition' + str(self.args.partitioning) + '_alpha' + str(self.args.dirichlet_alpha)     

        # global_portion_file = 'slave18_mapping_part_12_07_Test6_Google_speech_portion_partition' + str(self.args.partitioning) + '_alpha' + str(self.args.dirichlet_alpha)  

        partition_file = 'slave18_training_mapping_part_12_07_Test6_Google_speech_raw_partition' + str(self.args.partitioning) + '_clients' + str(num_clients) + '_data' + str(data_len) + '_labels'\
                    + str(num_class) + '_samples' + str(self.usedSamples) + '_alpha' + str(self.args.dirichlet_alpha)



        uniform_count_matrix_file = 'slave18_mapping_part_12_07_Test6_Google_speech_matrix_partition' + str(self.args.partitioning) + '_alpha' + str(self.args.dirichlet_alpha) + '_uniform_sampling'     

        uniform_global_portion_file = 'slave18_mapping_part_12_07_Test6_Google_speech_portion_partition' + str(self.args.partitioning) + '_alpha' + str(self.args.dirichlet_alpha) + '_uniform_sampling' 


        # filename = 'slave18_training_mapping_part_12_08_Test7_Google_speech_partition' + str(self.args.partitioning) + '_clients' + str(num_clients) + '_data' + str(data_len) + '_labels'\
        #             + str(num_class) + '_samples' + str(self.usedSamples) + '_alpha' + str(self.args.dirichlet_alpha)    

        # count_matrix_file = 'slave18_mapping_part_12_08_Test7_Google_speech_matrix_partition' + str(self.args.partitioning) + '_alpha' + str(self.args.dirichlet_alpha)     

        # global_portion_file = 'slave18_mapping_part_12_08_Test7_Google_speech_portion_partition' + str(self.args.partitioning) + '_alpha' + str(self.args.dirichlet_alpha)  

        # partition_file = 'slave18_training_mapping_part_12_08_Test7_Google_speech_raw_partition' + str(self.args.partitioning) + '_clients' + str(num_clients) + '_data' + str(data_len) + '_labels'\
        #             + str(num_class) + '_samples' + str(self.usedSamples) + '_alpha' + str(self.args.dirichlet_alpha)

        
        # filename = 'slave18_training_mapping_part_12_09_Test8_Google_speech_partition' + str(self.args.partitioning) + '_clients' + str(num_clients) + '_data' + str(data_len) + '_labels'\
        #             + str(num_class) + '_samples' + str(self.usedSamples) + '_alpha' + str(self.args.dirichlet_alpha)    

        # count_matrix_file = 'slave18_mapping_part_12_09_Test8_Google_speech_matrix_partition' + str(self.args.partitioning) + '_alpha' + str(self.args.dirichlet_alpha)     

        # global_portion_file = 'slave18_mapping_part_12_09_Test8_Google_speech_portion_partition' + str(self.args.partitioning) + '_alpha' + str(self.args.dirichlet_alpha)  

        # partition_file = 'slave18_training_mapping_part_12_09_Test8_Google_speech_raw_partition' + str(self.args.partitioning) + '_clients' + str(num_clients) + '_data' + str(data_len) + '_labels'\
        #             + str(num_class) + '_samples' + str(self.usedSamples) + '_alpha' + str(self.args.dirichlet_alpha)                    

        # filename = 'slave18_training_mapping_part_12_16_Test9_Google_speech_partition' + str(self.args.partitioning) + '_clients' + str(num_clients) + '_data' + str(data_len) + '_labels'\
        #             + str(num_class) + '_samples' + str(self.usedSamples) + '_alpha' + str(self.args.dirichlet_alpha)    

        # count_matrix_file = 'slave18_mapping_part_12_16_Test9_Google_speech_matrix_partition' + str(self.args.partitioning) + '_alpha' + str(self.args.dirichlet_alpha)     

        # global_portion_file = 'slave18_mapping_part_12_16_Test9_Google_speech_portion_partition' + str(self.args.partitioning) + '_alpha' + str(self.args.dirichlet_alpha)  

        # partition_file = 'slave18_training_mapping_part_12_16_Test9_Google_speech_raw_partition' + str(self.args.partitioning) + '_clients' + str(num_clients) + '_data' + str(data_len) + '_labels'\
        #             + str(num_class) + '_samples' + str(self.usedSamples) + '_alpha' + str(self.args.dirichlet_alpha)


        #Cifar10 
        # filename = 'slave18_training_mapping_part_12_09_Test3_Cifar10_partition' + str(self.args.partitioning) + '_clients' + str(num_clients) + '_data' + str(data_len) + '_labels'\
        #             + str(num_class) + '_samples' + str(self.usedSamples) + '_alpha' + str(self.args.dirichlet_alpha)    

        # count_matrix_file = 'slave18_mapping_part_12_09_Test3_Cifar10_matrix_partition' + str(self.args.partitioning) + '_alpha' + str(self.args.dirichlet_alpha)     

        # global_portion_file = 'slave18_mapping_part_12_09_Test3_Cifar10_portion_partition' + str(self.args.partitioning) + '_alpha' + str(self.args.dirichlet_alpha)  

        # partition_file = 'slave18_training_mapping_part_12_09_Test3_Cifar10_raw_partition' + str(self.args.partitioning) + '_clients' + str(num_clients) + '_data' + str(data_len) + '_labels'\
        #             + str(num_class) + '_samples' + str(self.usedSamples) + '_alpha' + str(self.args.dirichlet_alpha)        

        
        # filename = 'slave18_training_mapping_part_12_10_Test4_Cifar10_partition' + str(self.args.partitioning) + '_clients' + str(num_clients) + '_data' + str(data_len) + '_labels'\
        #             + str(num_class) + '_samples' + str(self.usedSamples) + '_alpha' + str(self.args.dirichlet_alpha)    

        # # count_matrix_file = 'slave18_mapping_part_12_10_Test4_Cifar10_matrix_partition' + str(self.args.partitioning) + '_alpha' + str(self.args.dirichlet_alpha)     

        # # global_portion_file = 'slave18_mapping_part_12_10_Test4_Cifar10_portion_partition' + str(self.args.partitioning) + '_alpha' + str(self.args.dirichlet_alpha)  

        # partition_file = 'slave18_training_mapping_part_12_10_Test4_Cifar10_raw_partition' + str(self.args.partitioning) + '_clients' + str(num_clients) + '_data' + str(data_len) + '_labels'\
        #             + str(num_class) + '_samples' + str(self.usedSamples) + '_alpha' + str(self.args.dirichlet_alpha)


        # uniform_count_matrix_file = 'slave18_mapping_part_12_10_Test4_Cifar10_matrix_partition' + str(self.args.partitioning) + '_alpha' + str(self.args.dirichlet_alpha) + '_uniform_sampling'     

        # uniform_global_portion_file = 'slave18_mapping_part_12_10_Test4_Cifar10_portion_partition' + str(self.args.partitioning) + '_alpha' + str(self.args.dirichlet_alpha) + '_uniform_sampling' 


        # filename = 'slave18_training_mapping_part_12_13_Test5_Cifar10_partition' + str(self.args.partitioning) + '_clients' + str(num_clients) + '_data' + str(data_len) + '_labels'\
        #             + str(num_class) + '_samples' + str(self.usedSamples) + '_alpha' + str(self.args.dirichlet_alpha)    

        # count_matrix_file = 'slave18_mapping_part_12_13_Test5_Cifar10_matrix_partition' + str(self.args.partitioning) + '_alpha' + str(self.args.dirichlet_alpha)     

        # global_portion_file = 'slave18_mapping_part_12_13_Test5_Cifar10_portion_partition' + str(self.args.partitioning) + '_alpha' + str(self.args.dirichlet_alpha)  

        # partition_file = 'slave18_training_mapping_part_12_13_Test5_Cifar10_raw_partition' + str(self.args.partitioning) + '_clients' + str(num_clients) + '_data' + str(data_len) + '_labels'\
        #             + str(num_class) + '_samples' + str(self.usedSamples) + '_alpha' + str(self.args.dirichlet_alpha)
        
        # filename = 'slave18_training_mapping_part_12_15_Test6_Cifar10_partition' + str(self.args.partitioning) + '_clients' + str(num_clients) + '_data' + str(data_len) + '_labels'\
        #             + str(num_class) + '_samples' + str(self.usedSamples) + '_alpha' + str(self.args.dirichlet_alpha)    

        # count_matrix_file = 'slave18_mapping_part_12_15_Test6_Cifar10_matrix_partition' + str(self.args.partitioning) + '_alpha' + str(self.args.dirichlet_alpha)     

        # global_portion_file = 'slave18_mapping_part_12_15_Test6_Cifar10_portion_partition' + str(self.args.partitioning) + '_alpha' + str(self.args.dirichlet_alpha)  

        # partition_file = 'slave18_training_mapping_part_12_15_Test6_Cifar10_raw_partition' + str(self.args.partitioning) + '_clients' + str(num_clients) + '_data' + str(data_len) + '_labels'\
        #             + str(num_class) + '_samples' + str(self.usedSamples) + '_alpha' + str(self.args.dirichlet_alpha)




        folder = os.path.join(args.log_path, 'metadata', args.data_set, 'data_mappings')

        if not os.path.isdir(folder):                                      
            Path(folder).mkdir(parents=True, exist_ok=True)

        custom_mapping_file = os.path.join(folder, filename)                
        if args.this_rank != 1:                                             
            while (not os.path.exists(custom_mapping_file)):                
                time.sleep(120)

        #-------------------------------------------

        targets = self.getTargets()                                          
        keyDir = {key: int(key) for i, key in enumerate(targets.keys())}     

        mapping_matrix_file = os.path.join(folder, uniform_count_matrix_file)          
        if args.this_rank != 1:                                       
            while (not os.path.exists(mapping_matrix_file)):             
                time.sleep(120)

        raw_partition_file = os.path.join(folder, partition_file)                     
        if args.this_rank != 1:                                       
            while (not os.path.exists(raw_partition_file)):             
                time.sleep(120)        
        #-------------------------------------------

        if os.path.exists(custom_mapping_file):                             
            with open(custom_mapping_file, 'rb') as fin:                    

                logging.info(f'Loading partitioning from file {filename}')

                # self.partitions = pickle.load(fin)                            

                self.total_partition = pickle.load(fin)
                self.partitions = self.total_partition



        #-----------------------------------------
        if os.path.exists(mapping_matrix_file):             
            with open(mapping_matrix_file, 'rb') as fin1:                    
                logging.info(f'Loading partitioning from file {uniform_count_matrix_file}')
                mapping_matrix1 = pickle.load(fin1)                     

            final_partitions = [[] for i in range(num_clients)]
                
            num_sample1 = 0                                                       
 
            for n in range(num_clients):
                for m in list(targets.keys()):                                   
                    c2 = keyDir[m]
#                    print(f'self.partitions[n] is {self.partitions[n]}')
#                    print(f'c2 is {keyDir[m]}')                    
#                    print(f'mapping_matrix1[n][c2] is {mapping_matrix1[n][c2]}')      
#                    print(f'len(self.partitions[n][c2]) - mapping_matrix1[n][c2] is {len(self.partitions[n][c2]) - mapping_matrix1[n][c2]}')              
                    final_partitions[n].extend(self.rng.sample(self.partitions[n][c2], int(mapping_matrix1[n][c2])))
                    num_sample1 = num_sample1 + len(self.partitions[n][c2])      

            self.partitions = final_partitions

            self.partition_matrix = mapping_matrix1
            logging.info(f"Total number of samples (data regulation method) before data reshape is {num_sample1}.")
            logging.info(f"Total number of samples (data regulation method) after data reshape is {np.sum(self.partition_matrix)}.")            
        #------------------------------------------

            return  



        #get targets
    
#        targets = self.getTargets()                                           
#        keyDir = {key: int(key) for i, key in enumerate(targets.keys())}      

        keyLength = [0] * numOfLabels
        for key in keyDir.keys():                                             
            keyLength[keyDir[key]] = len(targets[key])                        
                          
        logging.info(f"Custom partitioning {self.args.partitioning} data, {data_len} samples of {numOfLabels}:{num_class} labels on {num_clients} clients, use {self.usedSamples} sample per client ...")
                                                                             
        # ratioOfClassWorker = self.create_mapping(sizes)                       

        ratioOfClassWorker = np.ones((len(sizes), numOfLabels)).astype(np.float32)
        

#        logging.info(f"After create_mapping!")                              
        if ratioOfClassWorker is None:                                        
            return self.uniform_partition(num_clients=num_clients)           

        sumRatiosPerClass = np.sum(ratioOfClassWorker, axis=1)               
        for worker in range(len(sizes)):
            ratioOfClassWorker[worker, :] = ratioOfClassWorker[worker, :] / float(sumRatiosPerClass[worker])   

        # classPerWorker -> Rows are workers and cols are classes
        tempClassPerWorker = np.zeros([len(sizes), numOfLabels])



        #--------------------------
        global_count = np.zeros((num_clients, numOfLabels))
        global_portion = np.zeros((1, numOfLabels))        
        num_matrix2 = np.zeros((num_clients, numOfLabels))                    
        num_sample2 = 0

        for n1 in range(num_clients):
            for m1 in list(targets.keys()):                                    
                label = keyDir[m1]             
                num_sample2 = num_sample2 + len(self.total_partition[n1][label])      
                num_matrix2[n1][label] = len(self.total_partition[n1][label]) 




        global_count_total = np.sum(num_matrix2, axis = 0)                   
        global_portion = global_count_total/np.sum(global_count_total)       


        logging.info(f"Current partition is {self.args.partitioning}.")
        logging.info(f"Global portion before data reshape is {global_portion}.")
        logging.info(f"Total examples bofore data reshape is {np.sum(global_count_total)}.")



        target_global_portion = np.ones((numOfLabels, 1)).astype(np.float32)
        target_global_portion = target_global_portion / numOfLabels


        count_matrix = np.zeros((num_clients, numOfLabels))               
        #--------------------------



        # split the classes
        for worker in range(len(sizes)):                                    
            self.partitions.append([])                                
            # enumerate the ratio of classes it should take    
            
            #--------------------------
            count_vector = np.zeros((numOfLabels, 1))

            max_c = 0

            for c in list(targets.keys()):                                           

                if num_matrix2[worker][keyDir[c]] < 1:
                    count_vector[keyDir[c]] = 0  
                else:
                    count_vector[keyDir[c]] = target_global_portion[keyDir[c]]/num_matrix2[worker][keyDir[c]]                     
            
                
                
                if count_vector[keyDir[c]] >= count_vector[max_c]:
                    max_c = keyDir[c]
            #--------------------------


            #----------------------------------            
            max_num = num_matrix2[worker][max_c]


            for c1 in list(targets.keys()):                                          
                if count_vector[keyDir[c1]] > 0:
                    count_matrix[worker][keyDir[c1]] = int(min((max_num, num_matrix2[worker][keyDir[c1]])))
                    tempClassPerWorker[worker][keyDir[c1]] += count_matrix[worker][keyDir[c1]]           

            #----------------------------------        

        global_count_total1 = np.sum(count_matrix, axis = 0)                      
        global_portion1 = global_count_total1/np.sum(global_count_total1)      

        logging.info(f"Global portion after data reshape is {global_portion1}.")                                   
        logging.info(f"Total number of samples after data reshape is {np.sum(global_count_total1)}.")              

        #-----------------------------------

        fi_partition2 = [[] for i in range(num_clients)]
                
        for n in range(num_clients):
            for m in list(targets.keys()):                               
                c6 = keyDir[m] 
#                logging.info(f'partition and matrix test len(fi_partition[n][c6]) is {len(fi_partition[n][c6])}')                        
#                logging.info(f'partition and matrix test int(count_matrix[n][c6]) is {int(count_matrix[n][c6])}') 
                fi_partition2[n].extend(self.rng.sample(self.total_partition[n][c6], int(count_matrix[n][c6])))

        self.partitions = fi_partition2
        self.partition_matrix = count_matrix
        #-----------------------------------
        # logging.info(f"Total number of samples after regulation is {np.sum(count_matrix)}.") 



        del tempClassPerWorker                                        


       
        if not os.path.exists(mapping_matrix_file):                                      
            with open(mapping_matrix_file, 'wb') as fout1:
                 pickle.dump(count_matrix, fout1)                                         
            logging.info(f'Storing partitioning to file {mapping_matrix_file}')          


        glo_portion_file = os.path.join(folder, uniform_global_portion_file)          
        if not os.path.exists(glo_portion_file):                                         
            with open(glo_portion_file, 'wb') as fout2:
                 pickle.dump(target_global_portion, fout2)                                           
            logging.info(f'Storing partitioning to file {uniform_global_portion_file}')    

#        logging.info(f"global_portion before writing is {global_portion}.")
#        with open(glo_portion_file, 'rb') as fin4:                   
#            logging.info(f'Loading partitioning from file {global_portion_file} for pickle testing')
                
#            global_portion2 = pickle.load(fin4)  
#            logging.info(f"global_portion2 is {global_portion2}.") 


        # if not os.path.exists(raw_partition_file):                                         
        #     with open(raw_partition_file, 'wb') as fout3:
        #          pickle.dump(raw_partition, fout3)                                           
        #     logging.info(f'Storing partitioning to file {partition_file}') 






    def custom_partition2_2(self, num_clients, ratio=1.0):                    
        logging.info(f"Start custom_partition!")
        # custom partition                                                  
        numOfLabels = self.getNumOfLabels()                                
        #update the data length to account for the ratio
        data_len = min(self.getDataLen(), int(self.getDataLen() * ratio))   
        sizes = [1.0 / num_clients for _ in range(num_clients)]             

        #get # of samples per worker
        #set the number of samples per worker                       
        self.usedSamples = self.args.used_samples if self.args.used_samples >= 0 else (self.args.batch_size + 1)  
        # get number of samples per worker
        if self.usedSamples <= 0:                                          
            self.usedSamples = int(data_len / num_clients)

        #Verify if the custom client partitioning exists
        num_class = numOfLabels
        num_remove_classes = 0
        if self.args.filter_class > 0:                                      
            num_remove_classes = self.args.filter_class
        elif self.args.filter_class_ratio > 0:                              
            num_remove_classes = round(numOfLabels * (1 - self.args.filter_class_ratio))
        num_class -= num_remove_classes                                    


        



        #Google_speech
        filename = 'slave18_training_mapping_part_12_07_Test6_Google_speech_partition' + str(self.args.partitioning) + '_clients' + str(num_clients) + '_data' + str(data_len) + '_labels'\
                    + str(num_class) + '_samples' + str(self.usedSamples) + '_alpha' + str(self.args.dirichlet_alpha)    

        # count_matrix_file = 'slave18_mapping_part_12_07_Test6_Google_speech_matrix_partition' + str(self.args.partitioning) + '_alpha' + str(self.args.dirichlet_alpha)     

        # global_portion_file = 'slave18_mapping_part_12_07_Test6_Google_speech_portion_partition' + str(self.args.partitioning) + '_alpha' + str(self.args.dirichlet_alpha)  

        partition_file = 'slave18_training_mapping_part_12_07_Test6_Google_speech_raw_partition' + str(self.args.partitioning) + '_clients' + str(num_clients) + '_data' + str(data_len) + '_labels'\
                    + str(num_class) + '_samples' + str(self.usedSamples) + '_alpha' + str(self.args.dirichlet_alpha)


        uniform_count_matrix_file = 'slave18_mapping_part_12_07_Test6_Google_speech_matrix_partition' + str(self.args.partitioning) + '_alpha' + str(self.args.dirichlet_alpha) + '_uniform_sampling'

        #random_dirichlet_alpha = 0.1
        #包括Stage4_slave18_Random_distribution_G1_2_random_sampling_train_test2r和Stage4_slave18_Random_distribution_G1_2_balance_sampling_train_test2r1系列
        # random_count_matrix_file = 'slave18_mapping_part_12_07_Test6_Google_speech_matrix_partition' + str(self.args.partitioning) + '_alpha' + str(self.args.dirichlet_alpha) + '_random_sampling'     
        # random_global_portion_file = 'slave18_mapping_part_12_07_Test6_Google_speech_portion_partition' + str(self.args.partitioning) + '_alpha' + str(self.args.dirichlet_alpha) + '_random_sampling'


        #random_dirichlet_alpha = 2
        #包括Stage4_slave18_Random_distribution_G1_2_random_sampling_train_test2_1
        # random_count_matrix_file = 'slave18_mapping_part_12_07_Test6_Google_speech_matrix_partition' + str(self.args.partitioning) + '_random_sampling' 
        # random_global_portion_file = 'slave18_mapping_part_12_07_Test6_Google_speech_portion_partition' + str(self.args.partitioning) + '_random_sampling'

        #random_dirichlet_alpha = 100
        random_count_matrix_file = 'slave18_mapping_part_12_07_Test6_Google_speech_matrix_partition' + str(self.args.partitioning) + '_random_dirichlet_alpha' + str(100) + '_random_sampling'     
        random_global_portion_file = 'slave18_mapping_part_12_07_Test6_Google_speech_portion_partition' + str(self.args.partitioning) + '_random_dirichlet_alpha' + str(100) + '_random_sampling'        




        # filename = 'slave18_training_mapping_part_12_08_Test7_Google_speech_partition' + str(self.args.partitioning) + '_clients' + str(num_clients) + '_data' + str(data_len) + '_labels'\
        #             + str(num_class) + '_samples' + str(self.usedSamples) + '_alpha' + str(self.args.dirichlet_alpha)    

        # count_matrix_file = 'slave18_mapping_part_12_08_Test7_Google_speech_matrix_partition' + str(self.args.partitioning) + '_alpha' + str(self.args.dirichlet_alpha)     

        # global_portion_file = 'slave18_mapping_part_12_08_Test7_Google_speech_portion_partition' + str(self.args.partitioning) + '_alpha' + str(self.args.dirichlet_alpha)  

        # partition_file = 'slave18_training_mapping_part_12_08_Test7_Google_speech_raw_partition' + str(self.args.partitioning) + '_clients' + str(num_clients) + '_data' + str(data_len) + '_labels'\
        #             + str(num_class) + '_samples' + str(self.usedSamples) + '_alpha' + str(self.args.dirichlet_alpha)

        
        # filename = 'slave18_training_mapping_part_12_09_Test8_Google_speech_partition' + str(self.args.partitioning) + '_clients' + str(num_clients) + '_data' + str(data_len) + '_labels'\
        #             + str(num_class) + '_samples' + str(self.usedSamples) + '_alpha' + str(self.args.dirichlet_alpha)    

        # count_matrix_file = 'slave18_mapping_part_12_09_Test8_Google_speech_matrix_partition' + str(self.args.partitioning) + '_alpha' + str(self.args.dirichlet_alpha)     

        # global_portion_file = 'slave18_mapping_part_12_09_Test8_Google_speech_portion_partition' + str(self.args.partitioning) + '_alpha' + str(self.args.dirichlet_alpha)  

        # partition_file = 'slave18_training_mapping_part_12_09_Test8_Google_speech_raw_partition' + str(self.args.partitioning) + '_clients' + str(num_clients) + '_data' + str(data_len) + '_labels'\
        #             + str(num_class) + '_samples' + str(self.usedSamples) + '_alpha' + str(self.args.dirichlet_alpha)                    

        # filename = 'slave18_training_mapping_part_12_16_Test9_Google_speech_partition' + str(self.args.partitioning) + '_clients' + str(num_clients) + '_data' + str(data_len) + '_labels'\
        #             + str(num_class) + '_samples' + str(self.usedSamples) + '_alpha' + str(self.args.dirichlet_alpha)    

        # count_matrix_file = 'slave18_mapping_part_12_16_Test9_Google_speech_matrix_partition' + str(self.args.partitioning) + '_alpha' + str(self.args.dirichlet_alpha)     

        # global_portion_file = 'slave18_mapping_part_12_16_Test9_Google_speech_portion_partition' + str(self.args.partitioning) + '_alpha' + str(self.args.dirichlet_alpha)  

        # partition_file = 'slave18_training_mapping_part_12_16_Test9_Google_speech_raw_partition' + str(self.args.partitioning) + '_clients' + str(num_clients) + '_data' + str(data_len) + '_labels'\
        #             + str(num_class) + '_samples' + str(self.usedSamples) + '_alpha' + str(self.args.dirichlet_alpha)


        #Cifar10 
        # filename = 'slave18_training_mapping_part_12_09_Test3_Cifar10_partition' + str(self.args.partitioning) + '_clients' + str(num_clients) + '_data' + str(data_len) + '_labels'\
        #             + str(num_class) + '_samples' + str(self.usedSamples) + '_alpha' + str(self.args.dirichlet_alpha)    

        # count_matrix_file = 'slave18_mapping_part_12_09_Test3_Cifar10_matrix_partition' + str(self.args.partitioning) + '_alpha' + str(self.args.dirichlet_alpha)     

        # global_portion_file = 'slave18_mapping_part_12_09_Test3_Cifar10_portion_partition' + str(self.args.partitioning) + '_alpha' + str(self.args.dirichlet_alpha)  

        # partition_file = 'slave18_training_mapping_part_12_09_Test3_Cifar10_raw_partition' + str(self.args.partitioning) + '_clients' + str(num_clients) + '_data' + str(data_len) + '_labels'\
        #             + str(num_class) + '_samples' + str(self.usedSamples) + '_alpha' + str(self.args.dirichlet_alpha)        

        
        # filename = 'slave18_training_mapping_part_12_10_Test4_Cifar10_partition' + str(self.args.partitioning) + '_clients' + str(num_clients) + '_data' + str(data_len) + '_labels'\
        #             + str(num_class) + '_samples' + str(self.usedSamples) + '_alpha' + str(self.args.dirichlet_alpha)    

        # # count_matrix_file = 'slave18_mapping_part_12_10_Test4_Cifar10_matrix_partition' + str(self.args.partitioning) + '_alpha' + str(self.args.dirichlet_alpha)     

        # # global_portion_file = 'slave18_mapping_part_12_10_Test4_Cifar10_portion_partition' + str(self.args.partitioning) + '_alpha' + str(self.args.dirichlet_alpha)  

        # partition_file = 'slave18_training_mapping_part_12_10_Test4_Cifar10_raw_partition' + str(self.args.partitioning) + '_clients' + str(num_clients) + '_data' + str(data_len) + '_labels'\
        #             + str(num_class) + '_samples' + str(self.usedSamples) + '_alpha' + str(self.args.dirichlet_alpha)



        # uniform_count_matrix_file = 'slave18_mapping_part_12_10_Test4_Cifar10_matrix_partition' + str(self.args.partitioning) + '_alpha' + str(self.args.dirichlet_alpha) + '_uniform_sampling'
        
        # #random_dirichlet_alpha = 100
        # random_count_matrix_file = 'slave18_mapping_part_12_10_Test4_Cifar10_matrix_partition' + str(self.args.partitioning) + '_alpha' + str(self.args.dirichlet_alpha) + '_random_sampling'      
        # random_global_portion_file = 'slave18_mapping_part_12_10_Test4_Cifar10_portion_partition' + str(self.args.partitioning) + '_alpha' + str(self.args.dirichlet_alpha) + '_random_sampling' 



        # filename = 'slave18_training_mapping_part_12_13_Test5_Cifar10_partition' + str(self.args.partitioning) + '_clients' + str(num_clients) + '_data' + str(data_len) + '_labels'\
        #             + str(num_class) + '_samples' + str(self.usedSamples) + '_alpha' + str(self.args.dirichlet_alpha)    

        # count_matrix_file = 'slave18_mapping_part_12_13_Test5_Cifar10_matrix_partition' + str(self.args.partitioning) + '_alpha' + str(self.args.dirichlet_alpha)     

        # global_portion_file = 'slave18_mapping_part_12_13_Test5_Cifar10_portion_partition' + str(self.args.partitioning) + '_alpha' + str(self.args.dirichlet_alpha)  

        # partition_file = 'slave18_training_mapping_part_12_13_Test5_Cifar10_raw_partition' + str(self.args.partitioning) + '_clients' + str(num_clients) + '_data' + str(data_len) + '_labels'\
        #             + str(num_class) + '_samples' + str(self.usedSamples) + '_alpha' + str(self.args.dirichlet_alpha)
        
        # filename = 'slave18_training_mapping_part_12_15_Test6_Cifar10_partition' + str(self.args.partitioning) + '_clients' + str(num_clients) + '_data' + str(data_len) + '_labels'\
        #             + str(num_class) + '_samples' + str(self.usedSamples) + '_alpha' + str(self.args.dirichlet_alpha)    

        # count_matrix_file = 'slave18_mapping_part_12_15_Test6_Cifar10_matrix_partition' + str(self.args.partitioning) + '_alpha' + str(self.args.dirichlet_alpha)     

        # global_portion_file = 'slave18_mapping_part_12_15_Test6_Cifar10_portion_partition' + str(self.args.partitioning) + '_alpha' + str(self.args.dirichlet_alpha)  

        # partition_file = 'slave18_training_mapping_part_12_15_Test6_Cifar10_raw_partition' + str(self.args.partitioning) + '_clients' + str(num_clients) + '_data' + str(data_len) + '_labels'\
        #             + str(num_class) + '_samples' + str(self.usedSamples) + '_alpha' + str(self.args.dirichlet_alpha)




        folder = os.path.join(args.log_path, 'metadata', args.data_set, 'data_mappings')
        #上面的文件夹理论上是MAIN_PATH/core/evals/metadata/cifar10/data_mappings  
        if not os.path.isdir(folder):                                       
            Path(folder).mkdir(parents=True, exist_ok=True)

        custom_mapping_file = os.path.join(folder, filename)               
        if args.this_rank != 1:                                             
            while (not os.path.exists(custom_mapping_file)):               
                time.sleep(120)

        #-------------------------------------------

        targets = self.getTargets()                                          
        keyDir = {key: int(key) for i, key in enumerate(targets.keys())}      

        mapping_matrix_file = os.path.join(folder, random_count_matrix_file)          
        if args.this_rank != 1:                                       
            while (not os.path.exists(mapping_matrix_file)):             
                time.sleep(120)

        raw_partition_file = os.path.join(folder, partition_file)                     
        if args.this_rank != 1:                                       
            while (not os.path.exists(raw_partition_file)):             
                time.sleep(120)        
        #-------------------------------------------

        if os.path.exists(custom_mapping_file):                            
            with open(custom_mapping_file, 'rb') as fin:                    

                logging.info(f'Loading partitioning from file {filename}')

                # self.partitions = pickle.load(fin)                           

                self.total_partition = pickle.load(fin)
                self.partitions = self.total_partition



        #-----------------------------------------
        uniform_mapping_matrix_file = os.path.join(folder, uniform_count_matrix_file)         
        if os.path.exists(uniform_mapping_matrix_file):             
            with open(uniform_mapping_matrix_file, 'rb') as fin2:                    
                logging.info(f'Loading partitioning from file {uniform_count_matrix_file}')
                mapping_matrix2 = pickle.load(fin2)  


        if os.path.exists(mapping_matrix_file):             
            with open(mapping_matrix_file, 'rb') as fin1:                    
                logging.info(f'Loading partitioning from file {random_count_matrix_file}')
                mapping_matrix1 = pickle.load(fin1)                     

            final_partitions = [[] for i in range(num_clients)]
                
            num_sample1 = 0                                                     
 
            for n in range(num_clients):
                for m in list(targets.keys()):                                    
                    c2 = keyDir[m]
#                    print(f'self.partitions[n] is {self.partitions[n]}')
#                    print(f'c2 is {keyDir[m]}')                    
#                    print(f'mapping_matrix1[n][c2] is {mapping_matrix1[n][c2]}')      
#                    print(f'len(self.partitions[n][c2]) - mapping_matrix1[n][c2] is {len(self.partitions[n][c2]) - mapping_matrix1[n][c2]}')              
                    final_partitions[n].extend(self.rng.sample(self.partitions[n][c2], int(mapping_matrix1[n][c2])))
                    num_sample1 = num_sample1 + len(self.partitions[n][c2])     

            self.partitions = final_partitions

            self.partition_matrix = mapping_matrix1
            logging.info(f"Total number of samples (data regulation method) before data reshape is {num_sample1}.")
            logging.info(f"Total number of samples (data regulation method) after data reshape is {np.sum(self.partition_matrix)}.")            
        #------------------------------------------

            return  



        #get targets
      
#        targets = self.getTargets()                                           
#        keyDir = {key: int(key) for i, key in enumerate(targets.keys())}     

        keyLength = [0] * numOfLabels
        for key in keyDir.keys():                                            
            keyLength[keyDir[key]] = len(targets[key])                       
           
        logging.info(f"Custom partitioning {self.args.partitioning} data, {data_len} samples of {numOfLabels}:{num_class} labels on {num_clients} clients, use {self.usedSamples} sample per client ...")
                                                                             
#         # ratioOfClassWorker = self.create_mapping(sizes)                       

#         ratioOfClassWorker = np.ones((len(sizes), numOfLabels)).astype(np.float32)
        

# #        logging.info(f"After create_mapping!")                            
#         if ratioOfClassWorker is None:                                        
#             return self.uniform_partition(num_clients=num_clients)           
#         
#     
#         sumRatiosPerClass = np.sum(ratioOfClassWorker, axis=1)                
#         for worker in range(len(sizes)):
#             ratioOfClassWorker[worker, :] = ratioOfClassWorker[worker, :] / float(sumRatiosPerClass[worker])   

        # classPerWorker -> Rows are workers and cols are classes
        tempClassPerWorker = np.zeros([len(sizes), numOfLabels])



        #--------------------------

        global_count = np.zeros((num_clients, numOfLabels))
        global_portion = np.zeros((1, numOfLabels))        
        num_matrix2 = np.zeros((num_clients, numOfLabels))                     
        num_sample2 = 0

        for n1 in range(num_clients):
            for m1 in list(targets.keys()):                                     
                label = keyDir[m1]             
                num_sample2 = num_sample2 + len(self.total_partition[n1][label])      
                num_matrix2[n1][label] = len(self.total_partition[n1][label]) 




        global_count_total = np.sum(num_matrix2, axis = 0)                     
        global_portion = global_count_total/np.sum(global_count_total)         


        logging.info(f"Current partition is {self.args.partitioning}.")
        logging.info(f"Global portion before data reshape is {global_portion}.")
        logging.info(f"Total examples bofore data reshape is {np.sum(global_count_total)}.")



        # target_global_portion = np.ones((numOfLabels, 1)).astype(np.float32)     
        # target_global_portion = target_global_portion / numOfLabels
        # target_global_portion = self.create_mapping([1])                      
        # target_global_portion = target_global_portion / np.sum(target_global_portion, axis=1)                                                                          #实际上的尺寸为1*NumofLabels                         

        count_matrix = np.zeros((num_clients, numOfLabels))                  

        # #--------------------------



        # split the classes

        random_dirichlet_list = () 
        random_dirichlet_alpha = 100
        random_sizes = [1]
        random_dirichlet_list = (random_dirichlet_alpha,)*numOfLabels     
        # ratioOfClassWorker = np.random.dirichlet(numOfLabelslen, size =len(sizes)).astype(np.float32)        
        random_ratioOfClassWorker = np.zeros([len(random_sizes), numOfLabels]).astype(np.float32)
        for j in range(len(random_sizes)):
            random_ratioOfClassWorker[j] = np.random.dirichlet([random_dirichlet_alpha]*numOfLabels).astype(np.float32)  

        target_global_portion = random_ratioOfClassWorker
        target_global_portion = target_global_portion / np.sum(target_global_portion, axis=1) 


        for worker in range(len(sizes)):                                    
            self.partitions.append([])                                       
            # enumerate the ratio of classes it should take    
            
            #--------------------------
            count_vector = np.zeros((numOfLabels, 1))

            max_c = 0

            for c in list(targets.keys()):                                                 

                if num_matrix2[worker][keyDir[c]] < 1:
                    count_vector[keyDir[c]] = 0  
                else:
                    count_vector[keyDir[c]] = target_global_portion[0][keyDir[c]]/num_matrix2[worker][keyDir[c]]                     
            
                
                
                if count_vector[keyDir[c]] >= count_vector[max_c]:
                    max_c = keyDir[c]
            #--------------------------


            #----------------------------------            
            max_num = num_matrix2[worker][max_c]


            for c1 in list(targets.keys()):                                  
                if count_vector[keyDir[c1]] > 0:

                    takeLength = int((target_global_portion[0][keyDir[c1]]/target_global_portion[0][max_c])*max_num)
                    count_matrix[worker][keyDir[c1]] = int(min(takeLength, num_matrix2[worker][keyDir[c1]]))
                    
                    # total_size = np.sum(num_matrix2[worker, :]) 
                    # takeLength = int(total_size*target_global_portion[0][keyDir[c1]])
                    # count_matrix[worker][keyDir[c1]] = min((takeLength, num_matrix2[worker][keyDir[c1]]))


                    tempClassPerWorker[worker][keyDir[c1]] += count_matrix[worker][keyDir[c1]]         

            #----------------------------------        
        
        # random_dirichlet_list = () 
        # random_dirichlet_alpha = 2
        # random_sizes = [1]
        # random_dirichlet_list = (random_dirichlet_alpha,)*numOfLabels       
        # # ratioOfClassWorker = np.random.dirichlet(numOfLabelslen, size =len(sizes)).astype(np.float32)        
        # random_ratioOfClassWorker = np.zeros([len(random_sizes), numOfLabels]).astype(np.float32)
        # for j in range(len(random_sizes)):
        #     random_ratioOfClassWorker[j] = np.random.dirichlet([random_dirichlet_alpha]*numOfLabels).astype(np.float32)  

        # target_global_portion = random_ratioOfClassWorker
        # target_global_portion = target_global_portion / np.sum(target_global_portion, axis=1) 

        # for worker in range(len(sizes)):                                     
        #     self.partitions.append([])                                      
        #     # enumerate the ratio of classes it should take    
        #     # logging.info(f"np.sum(mapping_matrix2[worker, :]) is {np.sum(mapping_matrix2[worker, :])}.")
        #     # logging.info(f"np.sum(num_matrix2[worker, :]) is {np.sum(num_matrix2[worker, :])}.")
        #     # total_size = np.sum(mapping_matrix2[worker, :])
        #     total_size = np.sum(num_matrix2[worker, :])            
        #     for c1 in list(targets.keys()):                                           
        #         takeLength = int(total_size*target_global_portion[0][keyDir[c1]])

        #         # count_matrix[worker][keyDir[c1]] = min(takeLength, mapping_matrix2[worker][keyDir[c1]])
        #         count_matrix[worker][keyDir[c1]] = min(takeLength, num_matrix2[worker][keyDir[c1]])

        #         tempClassPerWorker[worker][keyDir[c1]] += count_matrix[worker][keyDir[c1]]           

            #----------------------------------    





        global_count_total1 = np.sum(count_matrix, axis = 0)                       
        global_portion1 = global_count_total1/np.sum(global_count_total1)          

        logging.info(f"Global portion after data reshape is {global_portion1}.")                                    
        logging.info(f"Total number of samples after data reshape is {np.sum(global_count_total1)}.")               

        #-----------------------------------
        fi_partition2 = [[] for i in range(num_clients)]
                
        for n in range(num_clients):
            for m in list(targets.keys()):                           
                c6 = keyDir[m] 
#                logging.info(f'partition and matrix test len(fi_partition[n][c6]) is {len(fi_partition[n][c6])}')                        
#                logging.info(f'partition and matrix test int(count_matrix[n][c6]) is {int(count_matrix[n][c6])}') 
                fi_partition2[n].extend(self.rng.sample(self.total_partition[n][c6], int(count_matrix[n][c6])))

        self.partitions = fi_partition2
        self.partition_matrix = count_matrix
        #-----------------------------------
        # logging.info(f"Total number of samples after regulation is {np.sum(count_matrix)}.")  



        del tempClassPerWorker                                       


       
        if not os.path.exists(mapping_matrix_file):                               
            with open(mapping_matrix_file, 'wb') as fout1:
                 pickle.dump(count_matrix, fout1)                                             
            logging.info(f'Storing partitioning to file {random_count_matrix_file}')         


        glo_portion_file = os.path.join(folder, random_global_portion_file)          
        if not os.path.exists(glo_portion_file):                                         
            with open(glo_portion_file, 'wb') as fout2:
                 pickle.dump(target_global_portion, fout2)                                           
            logging.info(f'Storing partitioning to file {random_global_portion_file}')    

#        logging.info(f"global_portion before writing is {global_portion}.")
#        with open(glo_portion_file, 'rb') as fin4:                   
#            logging.info(f'Loading partitioning from file {global_portion_file} for pickle testing')
                
#            global_portion2 = pickle.load(fin4)  
#            logging.info(f"global_portion2 is {global_portion2}.") 


        # if not os.path.exists(raw_partition_file):                                         
        #     with open(raw_partition_file, 'wb') as fout3:
        #          pickle.dump(raw_partition, fout3)                                           
        #     logging.info(f'Storing partitioning to file {partition_file}') 








    def custom_partition_test(self, num_clients, ratio=1.0):              
        num_clients = 4                      
        
        # custom partition                                                 
        numOfLabels = self.getNumOfLabels()                                 
        #update the data length to account for the ratio
        data_len = min(self.getDataLen(), int(self.getDataLen() * ratio))   
        sizes = [1.0 / num_clients for _ in range(num_clients)]             

        #get # of samples per worker
        #set the number of samples per worker                       
        self.usedSamples = self.args.used_samples if self.args.used_samples >= 0 else (self.args.batch_size + 1) 
        # get number of samples per worker
        if self.usedSamples <= 0:                                          
            self.usedSamples = int(data_len / num_clients)

        #Verify if the custom client partitioning exists
        num_class = numOfLabels
        num_remove_classes = 0
        if self.args.filter_class > 0:                                      
            num_remove_classes = self.args.filter_class
        elif self.args.filter_class_ratio > 0:                              
            num_remove_classes = round(numOfLabels * (1 - self.args.filter_class_ratio))
        num_class -= num_remove_classes                                     

#              


        #Google_speech
        # filename = 'slave18_testing_mapping_part_12_07_Test6_Google_speech_partition_' + str(self.args.partitioning) + '_clients' + str(num_clients) + '_data' + str(data_len) + '_labels'\
        #             + str(num_class) + '_samples' + str(self.usedSamples) + '_alpha' + str(self.args.dirichlet_alpha)


        # filename = 'slave18_testing_mapping_part_12_07_Test6_Google_speech_partition_3'  + '_clients' + str(num_clients) + '_data' + str(data_len) + '_labels'\
        #             + str(num_class) + '_samples' + str(self.usedSamples) + '_alpha' + str(self.args.dirichlet_alpha)


        # filename = 'slave18_testing_mapping_part_12_07_Test6_Google_speech_partition_8_clients' + str(num_clients) + '_data' + str(data_len) + '_labels'\
        #             + str(num_class) + '_samples' + str(self.usedSamples) + '_alpha' + str(self.args.dirichlet_alpha)


        #Cifar10
        # filename = 'slave18_testing_mapping_part_12_10_Test4_Cifar10_partition_' + str(self.args.partitioning) + '_clients' + str(num_clients) + '_data' + str(data_len) + '_labels'\
        #             + str(num_class) + '_samples' + str(self.usedSamples) + '_alpha' + str(self.args.dirichlet_alpha)

        # filename = 'slave18_testing_mapping_part_12_10_Test4_Cifar10_partition_3' + '_clients' + str(num_clients) + '_data' + str(data_len) + '_labels'\
        #             + str(num_class) + '_samples' + str(self.usedSamples) + '_alpha' + str(self.args.dirichlet_alpha)

        filename = 'slave18_testing_mapping_part_12_10_Test4_Cifar10_partition_' + str(self.args.partitioning) + '_clients' + str(num_clients) + '_data' + str(data_len) + '_labels'\
                    + str(num_class) + '_samples' + str(self.usedSamples) + '_alpha' + str(self.args.dirichlet_alpha)+ 'regeneration_for_test2'



        # filename = 'slave18_testing_mapping_part_12_10_Test4_Cifar10_partition_8_clients' + str(num_clients) + '_data' + str(data_len) + '_labels'\
        #             + str(num_class) + '_samples' + str(self.usedSamples) + '_alpha' + str(self.args.dirichlet_alpha)

        # filename = 'slave18_testing_mapping_part_12_10_Test4_Cifar10_partition_8_clients' + str(num_clients) + '_data' + str(data_len) + '_labels'\
        #             + str(num_class) + '_samples' + str(self.usedSamples) + '_alpha' + str(self.args.dirichlet_alpha)+ 'regeneration_for_test2'        



        #FashionMNIST
        # filename = 'slave18_testing_mapping_part_2025_06_19_Test3_FashionMNIST_partition' + str(self.args.partitioning) + '_clients' + str(num_clients) + '_data' + str(data_len) + '_labels'\
        #             + str(num_class) + '_samples' + str(self.usedSamples) + '_alpha' + str(self.args.dirichlet_alpha)


        folder = os.path.join(args.log_path, 'metadata', args.data_set, 'data_mappings')

        if not os.path.isdir(folder):                                      
            Path(folder).mkdir(parents=True, exist_ok=True)

        custom_mapping_file = os.path.join(folder, filename)                
        if args.this_rank != 1:                                            
            while (not os.path.exists(custom_mapping_file)):              
                time.sleep(120)
        if os.path.exists(custom_mapping_file):                            
            with open(custom_mapping_file, 'rb') as fin:                    
                
                logging.info(f'Loading partitioning from file {filename}')
                self.partitions = pickle.load(fin)                            

                for i, part in enumerate(self.partitions):                             
                    labels = [self.indexToLabel[index] for index in part]     
                    #count_elems = Counter(labels)
                    #logging.info(f'part {i} len: {len(part)} labels: {count_elems.keys()} count: {count_elems.values()}')
            return  

        #get targets
        targets = self.getTargets()                                           
        keyDir = {key: int(key) for i, key in enumerate(targets.keys())}      

        keyLength = [0] * numOfLabels
        for key in keyDir.keys():                                             
            keyLength[keyDir[key]] = len(targets[key])                        
            
        logging.info(f"Custom partitioning test {self.args.partitioning} data, {data_len} samples of {numOfLabels}:{num_class} labels on {num_clients} clients_test, use {self.usedSamples} sample per client_test ...")
                                                                              
        ratioOfClassWorker = self.create_mapping(sizes)                       

        # ratioOfClassWorker = np.ones((len(sizes), numOfLabels)).astype(np.float32)


        if ratioOfClassWorker is None:                                        
            return self.uniform_partition(num_clients=num_clients)           

        sumRatiosPerClass = np.sum(ratioOfClassWorker, axis=1)               
        for worker in range(len(sizes)):
            ratioOfClassWorker[worker, :] = ratioOfClassWorker[worker, :] / float(sumRatiosPerClass[worker])   

        # classPerWorker -> Rows are workers and cols are classes
        tempClassPerWorker = np.zeros([len(sizes), numOfLabels])

        # split the classes
        for worker in range(len(sizes)):                                      
            self.partitions.append([])                                        
            # enumerate the ratio of classes it should take
            for c in list(targets.keys()):                                    
                takeLength = int(self.usedSamples * ratioOfClassWorker[worker][keyDir[c]])   
                takeLength = min(takeLength, keyLength[keyDir[c]])            

                indexes = self.rng.sample(targets[c], takeLength)             

                self.partitions[-1] += indexes                                
                
                labels = [self.indexToLabel[index] for index in self.partitions[-1]]  
                count_elems = Counter(labels)                                 


                self.partitions[-1] += indexes                               


                tempClassPerWorker[worker][keyDir[c]] += takeLength           


            #logging.info(f'worker: {worker} created partition len: {len(self.partitions[-1])} class/worker: {sum(tempClassPerWorker[worker])} labels:{tempClassPerWorker[worker]} ratios: {ratioOfClassWorker[worker]}')

        del tempClassPerWorker                                               

        #save the partitions as pickle file                                  
        if not os.path.exists(custom_mapping_file):                                            
            with open(custom_mapping_file, 'wb') as fout:
                 pickle.dump(self.partitions, fout)                                            
            logging.info(f'Storing partitioning to file {filename}')         










    def create_mapping(self, sizes):                                
        numOfLabels = self.getNumOfLabels()                   

        ratioOfClassWorker = None                                     
        if self.args.partitioning == 1:                                #1 NONIID-Uniform
            ratioOfClassWorker = np.random.rand(len(sizes), numOfLabels).astype(np.float32)    

        elif self.args.partitioning == 2:                              #2 NONIID-Zipfian     
            ratioOfClassWorker = np.random.zipf(self.args.zipf_param, [len(sizes), numOfLabels]).astype(np.float32)  

        elif self.args.partitioning == 3:                              #3 NONIID-Balanced
            ratioOfClassWorker = np.ones((len(sizes), numOfLabels)).astype(np.float32)        
        elif self.args.partitioning == 6:                          
            logging.info('Method of create_mapping is 6')
            dirichlet_list = [] 
            up_bound1 = 2                                          
            low_bound1 = 50
            up_bound2 = 100
                                                   
            ratio1 = 0.38                                        
            num1 =  math.floor(len(sizes)*ratio1)                 
            num2 = math.ceil(len(sizes)*(1-ratio1))              

#            step1 = 2*(up_bound1-self.args.dirichlet_alpha)/len(sizes)          
#            step2 = 2*(up_bound2-low_bound1)/len(sizes)                        

            step1 = (up_bound1-self.args.dirichlet_alpha)/num1           
            step2 = 2*(up_bound2-low_bound1)/num2                        

            logging.info(f"up_bound1 is {up_bound1}!")            


            ratioOfClassWorker = np.zeros([len(sizes), numOfLabels]).astype(np.float32)
#            dirichlet_list = (self.args.dirichlet_alpha,)*numOfLabelslen      
            for j in range(len(sizes)):
#                logging.info("==== length of sizes is:{}\n".format(self.args.partitioning, len(sizes), numOfLabels, num_remove_class, np.count_nonzero(ratioOfClassWorker))) 
                if j < (len(sizes)/2):
 #               dirichlet_list.append([self.args.dirichlet_alpha+j*step,]*numOfLabelslen)            
                    ratioOfClassWorker[j] = np.random.dirichlet([self.args.dirichlet_alpha+j*step1,]*numOfLabels).astype(np.float32)
                else:
                    ratioOfClassWorker[j] = np.random.dirichlet([low_bound1+j*step2,]*numOfLabels).astype(np.float32)

#            dirichlet_list.append = []            
#            ratioOfClassWorker = np.random.dirichlet(numOfLabelslen, size =len(sizes)).astype(np.float32)        
            

        elif self.args.partitioning == 7:                       
            logging.info('Method of create_mapping is 7')
            
            dirichlet_list = [] 
#            up_bound1 = 10                                        
#            low_bound1 = 50
#            up_bound2 = 100
#            step1 = 2*(up_bound1-self.args.dirichlet_alpha)/len(sizes)
#            step2 = 2*(up_bound2-low_bound1)/len(sizes)

            up_bound1 = 2                                          
            low_bound1 = 50
            up_bound2 = 100
                                                   
            ratio1 = 0.38                                     
            num1 =  math.floor(len(sizes)*ratio1)            
            num2 = math.ceil(len(sizes)*(1-ratio1))            

#            step1 = 2*(up_bound1-self.args.dirichlet_alpha)/len(sizes)        
#            step2 = 2*(up_bound2-low_bound1)/len(sizes)                   

            step1 = (up_bound1-self.args.dirichlet_alpha)/num1          
            step2 = 2*(up_bound2-low_bound1)/num2                       


            ratioOfClassWorker = np.zeros([len(sizes), numOfLabels]).astype(np.float32)
            ratioOfClassWorker1 = np.zeros([len(sizes), numOfLabels]).astype(np.float32)
#            dirichlet_list = (self.args.dirichlet_alpha,)*numOfLabelslen       
            for j in range(len(sizes)):
#                logging.info("==== length of sizes is:{}\n".format(self.args.partitioning, len(sizes), numOfLabels, num_remove_class, np.count_nonzero(ratioOfClassWorker)))  
                if j < (len(sizes)/2):
 #               dirichlet_list.append([self.args.dirichlet_alpha+j*step,]*numOfLabelslen)            
                    ratioOfClassWorker1[j] = np.random.dirichlet([self.args.dirichlet_alpha+j*step1,]*numOfLabels).astype(np.float32)
                else:
                    ratioOfClassWorker1[j] = np.random.dirichlet([low_bound1+j*step2,]*numOfLabels).astype(np.float32)
            
            for j in range(len(sizes)):
                ratioOfClassWorker[len(sizes)-1-j] = ratioOfClassWorker1[j]

        elif self.args.partitioning == 8:                                   
            dirichlet_list = () 
            dirichlet_list = (self.args.dirichlet_alpha,)*numOfLabels       
            # ratioOfClassWorker = np.random.dirichlet(numOfLabelslen, size =len(sizes)).astype(np.float32)        
            ratioOfClassWorker = np.zeros([len(sizes), numOfLabels]).astype(np.float32)
            for j in range(len(sizes)):
                ratioOfClassWorker[j] = np.random.dirichlet([self.args.dirichlet_alpha]*numOfLabels).astype(np.float32)  
            



        num_remove_class=0
        if self.args.filter_class > 0 or self.args.filter_class_ratio > 0:
            num_remove_class = self.args.filter_class if self.args.filter_class > 0 else round(numOfLabels * (1 - self.args.filter_class_ratio))  
            for w in range(len(sizes)):                               
                # randomly filter classes by forcing zero samples
                wrandom = self.rng.sample(range(numOfLabels), num_remove_class)  
                for wr in wrandom:
                    ratioOfClassWorker[w][wr] = 0.0 #0.001           



        #logging.info("==== Class per worker partitioning:{} clients:{} labels:{} rem_lable:{} count:{}  ====\n {} \n".format(self.args.partitioning, len(sizes), numOfLabels, num_remove_class, np.count_nonzero(ratioOfClassWorker), repr(ratioOfClassWorker)))
        logging.info("==== Class per worker partitioning:{} clients:{} labels:{} rem_lable:{} count:{}  ==== \n".format(self.args.partitioning, len(sizes), numOfLabels, num_remove_class, np.count_nonzero(ratioOfClassWorker)))    #np.count_nonzero是统计所有非零元素并返回它们的计数
        return ratioOfClassWorker







    def getTargets(self):                            
        tempTarget = self.targets.copy()
        #TODO:why the temp targets are reshuffled each time getTargets is called?
        for key in tempTarget:
             self.rng.shuffle(tempTarget[key])
        return tempTarget

    def log_selection(self,classPerWorker):           
        totalLabels = [0 for i in range(len(classPerWorker[0]))]        
        logging.info("====Total # of workers is :{}, w/ {} labels, {}".format(len(classPerWorker), len(classPerWorker[0]), len(self.partitions)))    
        for index, row in enumerate(classPerWorker):
            rowStr = ''
            numSamples = 0
            for i, label in enumerate(classPerWorker[index]):    
                rowStr += '\t'+str(int(label))                   
                totalLabels[i] += label                           
                numSamples += label                            
            logging.info(str(index) + ':\t' + rowStr + '\t' + 'with sum:\t' + str(numSamples) + '\t' + repr(len(self.partitions[index])))
            logging.info("=====================================\n")
        logging.info("Total selected samples is: {}, with {}\n".format(str(sum(totalLabels)), repr(totalLabels)))
        logging.info("=====================================\n")


    def use(self, partition, istest):

    # def use(self, partition, resample_count_vector, istest):
        # resultIndex = self.partitions[partition]               

        if not istest:
            if args.data_regulation == 0:
                resultIndex = self.partitions[partition]
            else:
                targets = self.getTargets()                                          
                keyDir = {key: int(key) for i, key in enumerate(targets.keys())}

                use_partitions = []
                
                for m in list(targets.keys()):                                   
                    c_num = keyDir[m]
                    use_partitions.extend(self.rng.sample(self.total_partition[partition][c_num], int(self.partition_matrix[partition][c_num])))

# #                    logging.info(f"resample_count_vector is {resample_count_vector}")                
# #                    logging.info(f"np.shape(resample_count_vector) is {np.shape(resample_count_vector)}")
#                     use_partitions.extend(self.rng.sample(self.total_partition[partition][c_num], int(resample_count_vector[c_num])))

                resultIndex = use_partitions
        else:
            resultIndex = self.partitions[partition]        

            # if (self.history_test_partition is None) or (np.all(self.history_test_partition != resample_count_vector)):
            #     self.history_test_partition = resample_count_vector

            #     num_clients = 4      
            #     numOfLabels = self.getNumOfLabels() 

            #     targets = self.getTargets()                                          
            #     keyDir = {key: int(key) for i, key in enumerate(targets.keys())}   
            #     # logging.info(f"np.shape(resample_count_vector) is {np.shape(resample_count_vector)}")
            #     # logging.info(f"resample_count_vector is {resample_count_vector}")                 
            #     max_c5 = 0
            #     for c5 in list(targets.keys()):                                 
            #         if resample_count_vector[keyDir[c5]] > resample_count_vector[max_c5]:
            #             max_c5 = keyDir[c5]

            #     test_partition = [[] for i in range(numOfLabels)]

            #     average_portion = 1/numOfLabels      
            #                                         
            #     local_portion = [g*(average_portion/resample_count_vector[max_c5]) for g in resample_count_vector] 

            #     for c3 in list(targets.keys()): 
            #         indexes =  targets[keyDir[c3]]
            #         self.rng.shuffle(indexes)
            #         test_partition[keyDir[c3]].extend(indexes)

            #     final_test_partitions = [[] for l in range(num_clients)]
                    
            #     start = np.zeros(numOfLabels, dtype=int)    
            #     start1 = np.zeros(numOfLabels, dtype=int)  
            #     for o in range(num_clients):          
                    
            #         count = 0
            #         for c4 in list(targets.keys()):        
            #             end = min(start[keyDir[c4]]+int(len(test_partition[keyDir[c4]]) / num_clients), len(test_partition[keyDir[c4]])) 
            #             count = count + (end-start[keyDir[c4]]) 
            #             start[keyDir[c4]] = start[keyDir[c4]] + int(len(test_partition[keyDir[c4]]) / num_clients)


            #         for c4 in list(targets.keys()):  

            #             end1 = min(start1[keyDir[c4]]+int(len(test_partition[keyDir[c4]]) / num_clients), len(test_partition[keyDir[c4]])) 

            #             count_c4 = end1 - start1[keyDir[c4]]-1
            #             sample_number = min(count_c4, int(count*local_portion[keyDir[c4]]))


            #             set_c4 = self.rng.sample(test_partition[keyDir[c4]][start1[keyDir[c4]]: end1], sample_number)

            #             final_test_partitions[o].extend(set_c4)

            #             start1[keyDir[c4]] = start1[keyDir[c4]] + int(len(test_partition[keyDir[c4]]) / num_clients)

            #     self.partitions = final_test_partitions
            #     resultIndex = self.partitions[partition]
            # else:
            #     resultIndex = self.partitions[partition] 

        exeuteLength = -1 if not istest else int(len(resultIndex) * self.args.test_ratio)    
       
        resultIndex = resultIndex[:exeuteLength]                                             
        self.rng.shuffle(resultIndex)                           

        return Partition(self.data, resultIndex)

    def getSize(self):
        # return the size of samples
        return {'size': [len(partition) for partition in self.partitions]}  
        

        # return {'size': [len(partition) for partition in self.partitions], 'global_client_count': self.obtain_client_count()}


    


def select_dataset(rank, partition, batch_size, isTest=False, collate_fn=None, seed=0):

# def select_dataset(rank, partition, batch_size, isTest=False, collate_fn=None, seed=0, vector = []):
    """Load data given client Id"""                    
    partition = partition.use(rank - 1, isTest)

    # partition = partition.use(rank - 1, vector, isTest)

    dropLast = False if isTest or (args.used_samples < 0) else True               

    num_loaders = min(int(len(partition)/args.batch_size/2), args.num_loaders)   

    if num_loaders == 0:
        time_out = 0
    else:
        time_out = 60

    if collate_fn is not None:
        return DataLoader(partition, batch_size=batch_size, shuffle=True, pin_memory=True, timeout=time_out, num_workers=num_loaders, drop_last=dropLast, collate_fn=collate_fn)
    return DataLoader(partition, batch_size=batch_size, shuffle=True, pin_memory=True, timeout=time_out, num_workers=num_loaders, drop_last=dropLast)

def obtain_client_portion(rank, partition, num_class, seed=0):              
    """获取cleint数据分布"""                                       

    count = np.zeros((1, num_class))
    if (args.similarity == 1) and (args.data_regulation == 0):
        labels = [partition.indexToLabel[index] for index in partition.partitions[rank-1]]   
        for j in labels:
            count[0][int(j)] += 1
    else:
        for i in range(num_class):
            logging.info(f"Current clientid is {rank}, i is {i}  ...")        
            count[0][i] = len(partition.total_partition[rank-1][i])

    client_portion = count/sum(count) 

    return client_portion