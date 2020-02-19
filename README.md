# Influence-aware Memory
Source code accompanying the paper [Influence-aware Memory for Deep Reinforcement Learning](https://arxiv.org/pdf/1911.07643.pdf)
## Installation
1. Clone the repo and cd into it
    ```
    git clone https://github.com/miguelsuau/influence-aware-memory.git
    cd influence-aware-memory
    ```
2. (Recommended) Create a virtual environment and activate it

   * Using Anaconda:

     1. If not already installed, install [Anaconda](https://www.anaconda.com/distribution/)
     2. create virtual envirnonment and activate it 
        ```
        conda create -n <env-name> python=3.7
        conda activate <env-name>
        ```

    * Using virtualenv

      1. If not already installed, do
         ```
         pip install virtualenv
         ```
      2. create virtual envirnonment and activate it
         ```
          virtualenv /path/to/venv --python=python3
          . /path/to/venv/bin/activate
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
