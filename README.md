![header.png](media/header.png)
# 04 Continous User Feedback
## Streamlit App
Install pip environment via `pip install -r requirements.txt`

Run with 
`streamlit run badpun.py` from root directory.

### Structure
The directory `pages/` holds the streamlit subpages in the order in which they are shown in the interface. 
`Start.py` is the starting point of the whole app. 

## Clustering
### Environment
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
