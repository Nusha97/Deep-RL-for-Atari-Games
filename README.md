# Deep-RL-for-Atari-Games
Investigating different reinforcement algorithms on openai baseline datasets


We actually use two platforms according to our accessible resources:

`Notebooks` is used for training CartPole with DQN and both games with A2C (run on colab)

`baselines` is used for training Acrobot with DQN (run on server with GPU)









## Note for `baselines`
results stored in `./logs` folder; pics stored in `./pics` folder 


### Changed files: 
setup.py: cloudpickle==1.2.0    //I have this version problem. If you don't have then it can be ignored.

train_xxx.py: logger.configure("logs/xxx")     //to set the result saving path

run.sh    //to run the training

plot.py   //to plot results


### To run:
setup environment as baselines readme file (setup.py has to be changed before running `pip install -e .` to install right version)

tuning parameters in `./baselines/deepq/defaults.py` or build a file like `./baselines/deepq/experiments/train_cartpole.py`

`sh ./run.sh`

`python plot.py`


### Games to test:
CartPole-v0, Acrobot-v1


### Parameters to be tuned:
learning rate, batch size, buffer_size
