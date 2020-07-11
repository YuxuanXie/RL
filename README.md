# My code for practicing

## Environment setting
```shell
  #Create gym_env for gym 
  conda create -n gym_env -c hcc -c conda-forge python=3.6 gym

  #Install pytorch
  conda install pytorch torchvision cudatoolkit=10.1 -c pytorchs

  #install tensorflow
  pip install tensorflow==2.1.0 -i https://mirrors.cloud.tencent.com/pypi/simple
  pip install tensorflow-probability==0.9.0


```
## First one -- Use DQN to solve CartPole problem 
```shell
python dqn_CartPole.py
```
