#!bin/bash
#python init.py -c sections -d bm25 -r 0.001 -i 10

# do parameter search
python init_mlp.py -c sections -d bm25

# do test based on the previously done parameter search
python init_mlp.py -c sections -d bm25 --doTest --test1stActivation relu --test1stSize 1000 --test1stDropout --test2ndActivation sigmoid --test2ndSize 1000
