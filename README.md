Guys - please work in your branch and create pull requests for me to merge code into the master branch. Thanks.

Please clone your respective branch and commit/push to that, eg:

  git clone https://github.com/Alicia-Nicklin/EMAT10006_Block5_Assessment --branch Networks-dev

Run this program in the Terminal. PyCharm doesn't display updating plots well.

```
usage: assignment.py [-h] [-ising_model ISING_MODEL] [-external EXTERNAL] [-alpha ALPHA] [-test_ising TEST_ISING] [-defuant DEFUANT] [-beta BETA] [-threshold THRESHOLD] [-test_defuant TEST_DEFUANT] [-network NETWORK] [-test_network TEST_NETWORK]
                     [-random_network RANDOM_NETWORK] [-connection_probability CONNECTION_PROBABILITY] [-ring_network RING_NETWORK] [-range RANGE] [-small_world SMALL_WORLD] [-re_wire RE_WIRE]

options:
  -h, --help            show this help message and exit
  -ising_model ISING_MODEL
                        Ising model with default parameters
  -external EXTERNAL    Ising external value. Defaults to 0
  -alpha ALPHA          Ising temperature value. Defaults to 1
  -test_ising TEST_ISING
                        Run Ising tests
  -defuant DEFUANT      Defuant model with default parameters
  -beta BETA            Defuant beta value. Defaults to 0.2
  -threshold THRESHOLD  Defuant threshold value. Defaults to 0.2
  -test_defuant TEST_DEFUANT
                        Run defuant tests
  -network NETWORK      Create a random network, size of n
  -test_network TEST_NETWORK
                        Run network tests
  -random_network RANDOM_NETWORK
                        Create a random network, size of n
  -connection_probability CONNECTION_PROBABILITY
                        Connection probability. Defaults to 0.3
  -ring_network RING_NETWORK
                        Create a ring network with a range of 1 and a size of n
  -range RANGE          Network range. Defaults to 2
  -small_world SMALL_WORLD
                        Create a small-worlds network with default parameters, size n
  -re_wire RE_WIRE      Re-wire probability. Defaults to 0.2
```
