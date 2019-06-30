#!/bin/bash

if [ $# -ne 1 ]; then
    echo "Parameters: 'vera' or 'mm'"
    exit
fi

if [ $1 == "vera" ]; then
    scp -P 51000 -r net config processor root@10.66.31.100:/workplace/ken/repos/st-gcn
elif [ $1 == 'mm' ]; then
    scp -r net config processor zliu6676@129.78.10.182:/home2/zliu6676/action-recognition/st-gcn-all/st-gcn
else
    echo 'Target does not match one of the supported targets'
fi
