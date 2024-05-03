
Run this program in the Terminal. PyCharm doesn't display updating plots well.

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import argparse
import random
import sys 

link to our repository- https://github.com/Alicia-Nicklin/EMAT10006_Block5_Assessment.git

```
usage: assignment.py [-h] [-ising_model] [-external EXTERNAL] [-alpha ALPHA] [-test_ising] [-use_network USE_NETWORK] [-defuant] [-beta BETA] [-threshold THRESHOLD] [-test_defuant] [-network NETWORK] [-test_network] [-random_network RANDOM_NETWORK] [-connection_probability CONNECTION_PROBABILITY] [-ring_network RING_NETWORK] [-range RANGE] [-small_world SMALL_WORLD] [-re_wire RE_WIRE] [-plot_opinions PLOT_OPINIONS]

options:
  -h, --help            show this help message and exit
  -ising_model          Ising model with default parameters
  -external EXTERNAL    Ising external value. Defaults to 0
  -alpha ALPHA          Ising temperature value. Defaults to 1
  -test_ising           Run Ising tests
  -use_network USE_NETWORK
                        Uses networks for ising model
  -defuant              Defuant model with default parameters
  -beta BETA            Defuant beta value. Defaults to 0.5
  -threshold THRESHOLD  Defuant threshold value. Defaults to 0.5
  -test_defuant         Run defuant tests
  -network NETWORK      Create a random network, size of n
  -test_network         Run network tests
  -random_network RANDOM_NETWORK
                        Create a random network, size of n
  -connection_probability CONNECTION_PROBABILITY
                        Connection probability. Defaults to 0.3
  -ring_network RING_NETWORK
                        Create a ring network range 1, size of n
  -range RANGE          Network range. Defaults to 2
  -small_world SMALL_WORLD
                        Small-worlds network default parameters, size n
  -re_wire RE_WIRE      Re-wire probability. Defaults to 0.2
  -plot_opinions PLOT_OPINIONS
  
Commands checked:

python assignment.py -ising_model 
python assignment.py -ising_model -external -0.1 
python assignment.py -ising_model -alpha 10
python assignment.py -test_ising

python assignment.py -ising_model -external 0 -alpha 0.01    
python assignment.py -ising_model -external 0 -alpha 10
python assignment.py -ising_model -external 0.1 -alpha 0.001
python assignment.py -ising_model -external 0.1 -alpha 0.5

python assignment.py -defuant 		We only get the graphs if we run the model 10,000 steps and plot each 100th
python assignment.py -defuant -beta 0.1
python assignment.py -defuant -threshold 0.3
python assignment.py -test_defuant

python assignment.py -defuant -beta 0.5 -threshold 0.5
python assignment.py -defuant -beta 0.1 -threshold 0.5
python assignment.py -defuant -beta 0.5 -threshold 0.1
python assignment.py -defuant -beta 0.1 -threshold 0.2

python assignment.py -test_network

python assignment.py -random_network 20
python assignment.py -ring_network 10
python assignment.py -small_world 10
python assignment.py -small_world 10 -re_wire 0.1

python assignment.py -small_world 20 -re_wire 0.2
python assignment.py -small_world 20 -re_wire 0.0
python assignment.py -small_world 20 -re_wire 0.98    



python assignment.py -ising_model -use_network 10 	
python assignment.py -ising_model -use_network 500

```
