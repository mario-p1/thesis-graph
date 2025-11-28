# Thesis Graph
This repository investigates the structural and textual relationships that arise in the supervision and evaluation of bachelor theses. We construct a graph-based representation linking student authors, supervising mentors, and examination committee members, and analyze historical interaction patterns to characterize mentorship relationships.

The primary contribution is a predictive framework that, given a bachelor thesis abstract, estimates the most suitable supervisor by integrating textual features, relational graph information, and historical assignment data.


## Development
### Requirements
1. uv

### Project Setup
```bash
# For CPU only machines
make install_deps_cpu

# For GPU enabled machines with CUDA 12.8
make install_deps_cu128
```

### Data
The data folder must contain the following files:
1. `committee.csv`
2. `scholar_details.json`

### Train
```bash
make train
```
