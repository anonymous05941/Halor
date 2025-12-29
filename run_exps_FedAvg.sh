#!/usr/bin/env bash

#setup the python path
cwd=`pwd`
PY=`which python`

#first cmd line parameter is the dataset name
dataset=$1     

#the path to the project
export MAIN_PATH=/home/Halor
# the path to the dataset, note $dataset, the dataset name passed as argument to script   
export DATA_PATH=/home/Halor/data/${dataset} 
#the path to the conda envirnoment                          
export CONDA_ENV=/home/anaconda3/envs/halor
#the path to the conda source script
export CONDA_PATH=/home/anaconda3/

#Set WANDB for logging the experiments results or do wandb login from terminal
export WANDB_API_KEY=""  
# The entity or team used for WANDB logging, should be set correctly, typically should be set your WANDB userID
export WANDB_ENTITY=""                                    
                            

#RESET LD Library path
export LD_LIBRARY_PATH=""

#set the path to the config file, use the files with _exp suffix
cd core/evals
config_dir=`ls -1 configs/$dataset/*exp.*`

#other parameters is related to the server IPs and number of GPUs to utilize

#Set the list of IP addresses to run the experiment, each server will run a single experiment in a round robin fashion
declare -a IPS=('127.0.0.1')
# declare -a IPS=('10.119.46.61')  #原来是127.0.0.1
#declare -a IPS=('10.0.0.1' '10.0.0.2' '10.0.0.3' '10.0.0.4')

#Set the list with the number of GPUs to utilize on each server
declare -a GPUS=("2")           
#declare -a GPUS=("4" "4" "4" "4")

#number of servers
SERVERS=1                                

#Tag to identify the experiment on WANDB                        
tags="FedAvg" 

export TAGS=$tags

#---- Setting the main experimental parameters -----         
#number of FL rounds                                            
epochs=1000
#number of local training epochs to run on each client        
steps="1"                                                   
#number of workers per round                                
workers="10" #"10 50"
#the preset deadline for each round
deadlines="100"  
# the aggregation algorithm
aggregators="fedavg"
#target number of clients to complete the round (80% as per Google recommendation)
targetratios="0" #"0.1" for safa #"0.1 0.3" # 0.8 is default, 0 means no round failures   
#total number of clients in the experiment
clients=1000 #"0" "1000"              
#the data partitioning method: -1 IID-uniform, 0 Fedscale mapping, 1 NONIID-Uniform, 2 NONIID-Zipfian, 3 Balanced, 8-Dirichlet
partitions="8" #"-1 0 1 2 3 8"                          
#sampling seed set for randomizing the experiment, the experiment runs 3 times with seeds 0, 1 and 2 and average is taken
sampleseeds="0 1 2"                                            
#introduce artificial dropouts
dropoutratio=0.0 #not used in experiments               


# The overcommitted clients during the selection phase to account for possible dropouts, similar to oort it is set to 30%
overcommit="1.3"
# experiment type: 0 - we relay on round deadlines for completion similar to Google's system setting  
#                  1 - we wait for the target ratio to complete the round and there is no deadline simialr to setting in oort
exptypes="0" #"0 1"



# use Behaviour heterogenity or not: 0: do not use behaviour trace - always available, 1: use behaviour trace, dynamic client availability
randbehv="1" #0 1                                                

#---- Selection scheme -----
#client sampling choice
samplers="random" 


#whether to use stale aggregation -1:use REFL's Stale-Aware Aggregation, 0:no stale aggregation, otherwise the number of rounds for the threshold, e.g., SAFA use 5 for stale rounds threshold
stales="0" #-1 1 5                                        
# The multiplication factor for the weight/importance of new updates: 0: not used similar to 1:Equal weight as of the new updates, 2: divide by fixed weight 2 (half),
stalefactors="0" #"2 0 1 -1 -2 -3 -4"                       
# The beta value for the weighted average of the damping and scaling factors in the stale update
stalebeta=0.35 #0.65                                      
#The scaling coefficient to boost the good stale updates
scalecoff=1




#whether to use availability prioritization 0: do not use priority, 1: use availability prioritization
availprios="0" #0 1                                         
# probability for the oracle to get the availability right (Accuraccy level which should match from the average performance of the time-series model)
availprob=1                                                      
#wether to enable the adaptive selection of the clients: 0: do not use, 1: use
adaptselect=0 #0 1

scale_sys=0.01 #1.0 2.0

scalesyspercent=1.0 #            






if [ $dataset == 'google_speech' ] || [ $dataset == 'google_speech_dummy' ]
then
  config_dir=`ls -1 configs/google_speech/*exp.*`

fi
cd $cwd


# for loop to run the experiments     
count=0
for f in $config_dir;
do
  for exptype in $exptypes;
  do
    for worker in $workers;
    do
      for sampler in $samplers;
      do
        for aggregator in $aggregators;
        do
          for deadline in $deadlines;
          do
            for targetratio in $targetratios
            do
              for availprio in $availprios
              do
                  for stale in $stales;
                  do
                    for step in $steps;
                    do
                    for part in $partitions;
                    do
                     for sampleseed in $sampleseeds
                     do
                        for stalefactor in $stalefactors;
                        do
                        #Export the variables for the yaml file
                        export EPOCHS=$epochs
                        export DATASET=$dataset
                        export WORKERS=$worker
                        export STALEUPDATES=$stale
                        export LOCALSTEPS=$steps
                        export DEADLINE=$deadline
                        export TARGETRATIO=$targetratio
                        export OVERCOMMIT=$overcommit
                        export SAMPLER=$sampler
                        export AGGREGATOR=$aggregator
                        export PARTITION=$part
                        export EXPTYPE=$exptype
                        export CLIENTS=$clients
                        export SAMPLESEED=$sampleseed
                        export AVAILPRIO=$availprio
                        export AVAILPROP=$availprob
                        export RANDBEHV=$randbehv
                        export ADAPT_SELECT=$adaptselect
                        export STALE_FACTOR=$stalefactor
                        export STALE_BETA=$stalebeta
                        export SCALE_COFF=$scalecoff
                        export DROPOUT_RATIO=$dropoutratio
                        export SCALE_SYS=$scale_sys
                        export SCALE_SYS_PERCENT=$scalesyspercent


                        echo "Settings: config=$f dataset=$dataset worker=$worker deadline=$deadline stale=$stale steps=$step sampler=$sampler aggregator=$aggregator exptype=$exptype clients=$CLIENTS"
                        index=`expr $count % $SERVERS`
                        echo "index: $index node info: ${IPS[$index]} ${GPUS[$index]}"
                        #invoke the manager to launch the PS and workers on target node
                        $PY $cwd/core/evals/manager.py submit $cwd/core/evals/$f ${IPS[$index]} ${GPUS[$index]}

                        sleep 5
                        #experiment counter
                        count=`expr $count + 1`

                        done
                      done
                    done
                  done
                done
              done
            done
          done
        done
      done
    done
  done
done