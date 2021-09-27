# My code for practicing

## Environment setting

insall conda

```shell
  #Create gym_env for gym 
  conda create -n gym_env python=3.8
  conda activate gym_env

  #Install gym
  conda install gym

  #Install pytorch
  conda install pytorch torchvision 

  ##stop here
  ------

  #install tensorflow
  pip install tensorflow==2.1.0 -i https://mirrors.cloud.tencent.com/pypi/simple
  pip install tensorflow-probability==0.9.0


```
## First one -- Use DQN to solve CartPole problem 
```shell
python dqn_CartPole.py
```
