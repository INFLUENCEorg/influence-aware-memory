# Influence-aware Memory
Source code accompanying the paper "Influence-aware Memory Architectures for Deep Reinforcement Learning"
## Requirements
   [Singularity](https://sylabs.io/docs/)
## Installation
1. Clone the repo and cd into it
    ```
    git clone https://github.com/miguelsuau/influence-aware-memory.git
    cd influence-aware-memory
    ```
2. Build Singularity container

   ```
   sudo singularity build influence-aware-memory.sif influence-aware-memory.def
   ```
## Training models
```
singularity run influence-aware-memory.sif python experimentor.py with <path_to_config_file>
```
#### Example: Warehouse - IAM-manual

```
singularity run influence-aware-memory.sif python experimentor.py with ./configs/warehouse/IAM_manual.yaml
```
