# 04 Continous User Feedback

## Clustering
### Environment
The file 'clustering_requirements.txt' may be used to create an environment using:
 `$ conda create --name <env> --file clustering_requirements.txt`

The file 'config.ini' contains all constants used throughout the system. In order to avoid spread of information, best
add new constants to this file, as well as parameterization of methods etc. Usage:

``` 
import configparser
config = configparser.ConfigParser()
config.read('config.ini')
config['DATA']['TestUserEmbeddingPath']
```

### Usage
The information flow of the system is demonstrated in `src/clustering/demo.ipynb`.
Each clustering algorithm inherits from the class ClusteringAlg, which predetermines the information flow. When adding
a new clustering algorithm, add a new wrapper class similar to the existing ones. 
