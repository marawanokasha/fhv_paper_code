#!/bin/bash

# LSTM

## Sections
python init.py -l 1 -c sections -b 4096
python init.py -l 2 -c sections -b 4096
python init.py -l 3 -c sections -b 2048

python init.py -l 1 -c sections -b 4096 --doTest --testLSTMSize 1000 --testWDropout 0.5 --testUDropout 0.5 --testStackLayers 1
python init.py -l 2 -c sections -b 4096 --doTest --testLSTMSize 1000 --testWDropout 0.5 --testUDropout 0.5 --testStackLayers 1
python init.py -l 3 -c sections -b 2048 --doTest --testLSTMSize 1000 --testWDropout 0.5 --testUDropout 0.5 --testStackLayers 1


## Classes
python init.py -l 1 -c classes -b 2048
python init.py -l 2 -c classes -b 2048
python init.py -l 3 -c classes -b 1024

python init.py -l 1 -c classes -b 2048 --doTest --testLSTMSize 1000 --testWDropout 0.5 --testUDropout 0.5 --testStackLayers 1
python init.py -l 2 -c classes -b 2048 --doTest --testLSTMSize 1000 --testWDropout 0.5 --testUDropout 0.5 --testStackLayers 1
python init.py -l 3 -c classes -b 1024 --doTest --testLSTMSize 1000 --testWDropout 0.5 --testUDropout 0.5 --testStackLayers 1


## Subclasses
python init.py -l 1 -c subclasses -b 2048
python init.py -l 2 -c subclasses -b 2048
python init.py -l 3 -c subclasses -b 1024

python init.py -l 1 -c subclasses -b 4096 --doTest --testLSTMSize 1000 --testWDropout 0.5 --testUDropout 0.5 --testStackLayers 1
python init.py -l 2 -c subclasses -b 4096 --doTest --testLSTMSize 1000 --testWDropout 0.5 --testUDropout 0.5 --testStackLayers 1
python init.py -l 3 -c subclasses -b 1024 --doTest --testLSTMSize 1000 --testWDropout 0.5 --testUDropout 0.5 --testStackLayers 1



# MLP

## Sections
python init_mlp.py -l 1 -c sections -b 4096
python init_mlp.py -l 2 -c sections -b 4096
python init_mlp.py -l 3 -c sections -b 2048

python init_mlp.py -l 1 -c sections -b 4096 --doTest --test1stDropout --test1stSize 1000 --test1stActivation sigmoid --test2ndSize 500 --test2ndActivation relu
python init_mlp.py -l 2 -c sections -b 4096 --doTest --test1stDropout --test1stSize 1000 --test1stActivation sigmoid --test2ndSize 500 --test2ndActivation relu
python init_mlp.py -l 3 -c sections -b 2048 --doTest --test1stDropout --test1stSize 1000 --test1stActivation sigmoid --test2ndSize 500 --test2ndActivation relu


## Classes
python init.py -l 1 -c classes -b 4096
python init.py -l 2 -c classes -b 4096
python init.py -l 3 -c classes -b 2048

python init_mlp.py -l 1 -c classes -b 4096 --doTest --test1stDropout --test1stSize 1000 --test1stActivation sigmoid --test2ndSize 500 --test2ndActivation relu
python init_mlp.py -l 2 -c classes -b 4096 --doTest --test1stDropout --test1stSize 1000 --test1stActivation sigmoid --test2ndSize 500 --test2ndActivation relu
python init_mlp.py -l 3 -c classes -b 2048 --doTest --test1stDropout --test1stSize 1000 --test1stActivation sigmoid --test2ndSize 500 --test2ndActivation relu


## Subclasses
python init.py -l 1 -c subclasses -b 4096
python init.py -l 2 -c subclasses -b 4096
python init.py -l 3 -c subclasses -b 2048

python init_mlp.py -l 1 -c subclasses -b 4096 --doTest --test1stDropout --test1stSize 1000 --test1stActivation sigmoid --test2ndSize 500 --test2ndActivation relu
python init_mlp.py -l 2 -c subclasses -b 4096 --doTest --test1stDropout --test1stSize 1000 --test1stActivation sigmoid --test2ndSize 500 --test2ndActivation relu
python init_mlp.py -l 3 -c subclasses -b 2048 --doTest --test1stDropout --test1stSize 1000 --test1stActivation sigmoid --test2ndSize 500 --test2ndActivation relu
