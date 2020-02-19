# Influence-aware Memory
Source code accompanying the paper [Influence-aware Memory for Deep Reinforcement Learning](https://openreview.net/pdf?id=rJlS-ertwr)
## Installation
Clone the repo and cd into it
```console
https://github.com/miguelsuau/influence-aware-memory.git
cd influence-aware-memory
```
if not already installed, install [Anaconda](https://www.anaconda.com/distribution/)

Create a virtual environment and activate it
```console
conda create -n <env-name> python=3.7
conda activate <env-name>
```
Install requirements
```console
pip install --user --requirement requirements.txt
```
Install OpenAI baselines

git clone https://github.com/openai/baselines.git
cd baselines

pip install -e .

## Training models
```console
python experimentor.py --config=<path_to_config_file>
```
# Example: Breakout- InfluenceNet

```console
python experimentor.py --config=./configs/Breakout/InfluenceNet
```
