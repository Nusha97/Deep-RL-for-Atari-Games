#games: CartPole-v0, MaintainCar-v0, Acrobot-v1, 
python -m baselines.run --alg=deepq --env=CartPole-v0 --log_path=logs/CartPole --save_path=models/CartPole.pkl --num_timesteps=1e5
python -m baselines.run --alg=deepq --env=MaintainCar-v0 --log_path=logs/MaintainCar --num_timesteps=1e5
python -m baselines.run --alg=deepq --env=Acrobot-v1 --log_path=logs/Acrobot --num_timesteps=1e5


#python -m baselines.deepq.experiments.train_cartpole
#python -m baselines.deepq.experiments.train_mountaincar