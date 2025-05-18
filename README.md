# Rethinking Reward Model Evaluation Through the Lens of Reward Overoptimization

Official repo for Rethinking Reward Model Evaluation Through the Lens of Reward Overoptimization (ACL 2025 Main Conference)

## 🛠️ Setup

```shell
conda create -n rm_eval python=3.10 -y
conda activate rm_eval
pip install -r requirements.txt
```

To evaluate results, [MARIO EVAL](https://github.com/MARIO-Math-Reasoning/MARIO_EVAL) needs to be installed. 
### Install MARIO EVAL as Python package
```shell
git clone https://github.com/MARIO-Math-Reasoning/MARIO_EVAL.git
cd MARIO_EVAL
cd latex2sympy && pip install . && cd ..
pip install -e .
```


## 🚀 Quick Start
### Inference Classifier-based Reward Models
```bash
bash scripts/run_classifier_rm.sh
```

### Inference Process Reward Mmodels
```bash
bash scripts/run_prm.sh
```


## 👏 Acknowledgements

The underlying codebase for evaluating reward model from [RewardBench](https://github.com/allenai/reward-bench).


## Citation 

```bibtex
```