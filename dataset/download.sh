#!/bin/bash

RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m' # no color
DIR="./data"

# Download and decompress datasets

Help()
{
   # Display Help
   echo "We provide example datasets"
   echo "to evalute the performance of Kuiper"
   echo 
   echo "Syntax: ./download.sh [-g|h|v|V]"
   echo "options:"
   echo "-h     Print this Help."
   echo "-s     Download Speech Commands dataset (about 2.3GB)"
   echo "-c     Download CIFAR10 dataset (about 170M)"
}



speech()
{
    if [ ! -d "${DIR}/google_speech/train/" ];
    then
        echo "Downloading Speech Commands dataset(about 2.4GB)..."
        wget -O ${DIR}/google_speech/google_speech.tar.gz https://fedscale.eecs.umich.edu/dataset/google_speech.tar.gz

        echo "Dataset downloaded, now decompressing..." 
        tar -xf ${DIR}/google_speech/google_speech.tar.gz -C ${DIR}

        echo "Removing compressed file..."
        rm -f ${DIR}/google_speech/google_speech.tar.gz

        echo -e "${GREEN}Speech Commands dataset downloaded!${NC}"
    else
        echo -e "${RED}Speech Commands dataset already exists under ${DIR}/google_speech/!"
fi
}




cifar10()
{
    if [ ! -d "${DIR}/cifar10/cifar-10-batches-py/" ];
    then
        echo "Downloading cifar10 dataset(about 170M)..."
        wget -O ${DIR}/cifar10/cifar10.tar.gz https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz

        echo "Dataset downloaded, now decompressing..."
        tar -xf ${DIR}/cifar10/cifar10.tar.gz -C ${DIR}

        echo "Removing compressed file..."
        rm -f ${DIR}/cifar10/cifar10.tar.gz

        echo -e "${GREEN}CIFAR10 dataset downloaded!${NC}"
    else
        echo -e "${RED}CIFAR10 dataset already exists under ${DIR}/cifar-10-batches-py/!"
fi
}


while getopts ":hsoacegildrtw" option; do
   case $option in
      h ) # display Help
         Help
         exit;;    
      s )
         speech   
         ;;
      c )
         cifar10
         ;;         
      \? ) 
         echo -e "${RED}Usage: cmd [-h] [-A] [-o] [-t] [-p]${NC}"
         exit 1;;
   esac
done

if [ $OPTIND -eq 1 ]; then 
    echo -e "${RED}Usage: cmd [-h] [-A] [-o] [-t] [-p]${NC}"; 
fi
