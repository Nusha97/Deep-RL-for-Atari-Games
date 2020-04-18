#games: CartPole-v0, MaintainCar-v0, Acrobot-v1
#tune parameters in deepq/defaluts.py first
#python -m baselines.run --alg=deepq --env=CartPole-v0 --log_path=logs/CartPole --save_path=models/CartPole.pkl --num_timesteps=1e5
#python -m baselines.run --alg=deepq --env=MaintainCar-v0 --log_path=logs/MaintainCar --save_path=models/MaintainCar.pkl --num_timesteps=1e5
time python -m baselines.run --alg=deepq --env=Acrobot-v1 --log_path=logs/Acrobot/lr1e-4_bch256_bf1e4_FF --save_path=models/Acrobot_lr1e-4_bch256_bf1e4_FF.pkl --num_timesteps=1e5

#another way to call deepq training. more convenient to tune parameters.
#python -m baselines.deepq.experiments.train_cartpole
#python -m baselines.deepq.experiments.train_mountaincar
# python -m baselines.deepq.experiments.train_acrobot
python plot.py