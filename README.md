# Bayesian Attack Graph

## Installation

`pip install -e .`

### Gambit
First download the latest version of Gambit:
1. `cd ~`
2. `mkdir gambit`
3. `cd gambit`
4. `wget https://sourceforge.net/projects/gambit/files/gambit16/16.0.0/gambit-16.0.0.tar.gz`
5. `tar -xvzf gambit-16.0.0.tar.gz`
6. `cd gambit-16.0.0/`

Now, we will perform a local install so that we do not need `sudo`:
7. `./configure --prefix=/home/mxsmith/gambit/`
8. `make`
9. `make install`

## Known Issues
 - Loading the random-uniform policy does not check for epochs >9. This is because it does string matching for ``epoch1'', so ``epoch10'', will be triggered.
