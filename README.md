# Influence-aware Memory
Source code accompanying the paper [Influence-aware Memory for Deep Reinforcement Learning](https://openreview.net/pdf?id=rJlS-ertwr)
## Installation
1. Clone the repo and cd into it
```
https://github.com/miguelsuau/influence-aware-memory.git
cd influence-aware-memory
```
2. If not already installed, install [Anaconda](https://www.anaconda.com/distribution/)

Create a virtual environment and activate it
```
conda create -n <env-name> python=3.7
conda activate <env-name>
```
3. Install requirements
```
pip install --user --requirement requirements.txt
```
4. Install OpenAI baselines

```
git clone https://github.com/openai/baselines.git
cd baselines
pip install -e .
```

## Training models
```
python experimentor.py --config=<path_to_config_file>
```
#### Example: Breakout - InfluenceNet

```
python experimentor.py --config=./configs/Breakout/InfluenceNet
```
