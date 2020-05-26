module load python-anaconda3-base/latest
yes | conda install python=3.7
yes | conda create -n bgg python=3.7
source activate bgg
python --version
