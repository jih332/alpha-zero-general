# Alpha Gomoku  

## CSE253 Final Project  

This project is folked from the [repo](https://github.com/suragnair/alpha-zero-general).
We reuse/modify some code in the original repo.
To start training a Gomoku agent:
```bash
python main.py
```
To change the arguments:
```bash
vim main.py
```
To change the arguments of model:
```bash
cd gomoku/tensorflow
vim GomokuNNet.py
```
To pit against our model:
```bash
python pit.py
```
Due to the size limit, we deleted the latest training examples file, hence you may not be able to load the latest training data to resume training.