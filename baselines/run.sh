#games: CartPole-v0, MaintainCar-v0, Acrobot-v1
#tune parameters in deepq/defaluts.py first
#python -m baselines.run --alg=deepq --env=CartPole-v0 --log_path=logs/CartPole --save_path=models/CartPole.pkl --num_timesteps=1e5
#python -m baselines.run --alg=deepq --env=MaintainCar-v0 --log_path=logs/MaintainCar --save_path=models/MaintainCar.pkl --num_timesteps=1e5
time python -m baselines.run --alg=deepq --env=Acrobot-v1 --log_path=logs/Acrobot/lr5e-3_bch32_bf1e4_FT --save_path=models/Acrobot_lr5e-3_bch32_bf1e4_FT.pkl --num_timesteps=1e5

#another way to call deepq training. more convenient to tune parameters.
#python -m baselines.deepq.experiments.train_cartpole
#python -m baselines.deepq.experiments.train_mountaincar
# python -m baselines.deepq.experiments.train_acrobot
# python plot.py

## Sarah's experiments (for CartPole):
# atari (DONE): time python -m baselines.run --alg=a2c --env=CartPole-v0 --log_path=logs/CartPole_0001_32 --save_path=models/CartPole_0001_32.pkl --num_timesteps=1e7 --lr=0.0001 --nsteps=32 # --num_env= 
# atari_two (DONE): time python -m baselines.run --alg=a2c --env=CartPole-v0 --log_path=logs/CartPole_0005_32 --save_path=models/CartPole_0005_32.pkl --num_timesteps=1e7 --lr=0.0005 --nsteps=32
# atari_three (REDO): CUDA_VISIBLE_DEVICES=1 time python -m baselines.run --alg=a2c --env=CartPole-v0 --log_path=logs/CartPole_001_32 --save_path=models/CartPole_001_32.pkl --num_timesteps=1e7 --lr=0.001 --nsteps=32
# atari_four: CUDA_VISIBLE_DEVICES=0 time python -m baselines.run --alg=a2c --env=CartPole-v0 --log_path=logs/CartPole_005_32 --save_path=models/CartPole_005_32.pkl --num_timesteps=1e7 --lr=0.005 --nsteps=32
# atari_five: CUDA_VISIBLE_DEVICES=0 time python -m baselines.run --alg=a2c --env=CartPole-v0 --log_path=logs/CartPole_0001_256 --save_path=models/CartPole_0001_256.pkl --num_timesteps=1e7 --lr=0.0001 --nsteps=256
# atari_six: CUDA_VISIBLE_DEVICES=1 time python -m baselines.run --alg=a2c --env=CartPole-v0 --log_path=logs/CartPole_0005_256 --save_path=models/CartPole_0005_256.pkl --num_timesteps=1e7 --lr=0.0005 --nsteps=256
# atari_seven: CUDA_VISIBLE_DEVICES=1 time python -m baselines.run --alg=a2c --env=CartPole-v0 --log_path=logs/CartPole_001_256 --save_path=models/CartPole_001_256.pkl --num_timesteps=1e7 --lr=0.001 --nsteps=256
# atari_eight: CUDA_VISIBLE_DEVICES=0 time python -m baselines.run --alg=a2c --env=CartPole-v0 --log_path=logs/CartPole_005_256 --save_path=models/CartPole_005_256.pkl --num_timesteps=1e7 --lr=0.005 --nsteps=256

# Second set (for Acrobot): 

# atari: CUDA_VISIBLE_DEVICES=1 time python -m baselines.run --alg=a2c --env=Acrobot-v1 --log_path=logs/Acrobot_0001_32 --save_path=models/Acrobot_0001_32.pkl --num_timesteps=1e7 --lr=0.0001 --nsteps=32 # --num_env= 
# atari_two: CUDA_VISIBLE_DEVICES=1 time python -m baselines.run --alg=a2c --env=Acrobot-v1 --log_path=logs/Acrobot_0005_32 --save_path=models/Acrobot_0005_32.pkl --num_timesteps=1e7 --lr=0.0005 --nsteps=32
# atari_three: CUDA_VISIBLE_DEVICES=1 time python -m baselines.run --alg=a2c --env=Acrobot-v1 --log_path=logs/Acrobot_001_32 --save_path=models/Acrobot_001_32.pkl --num_timesteps=1e7 --lr=0.001 --nsteps=32
# atari_four: CUDA_VISIBLE_DEVICES=1 time python -m baselines.run --alg=a2c --env=Acrobot-v1 --log_path=logs/Acrobot_005_32 --save_path=models/Acrobot_005_32.pkl --num_timesteps=1e7 --lr=0.005 --nsteps=32
# atari_five: CUDA_VISIBLE_DEVICES=1 time python -m baselines.run --alg=a2c --env=Acrobot-v1 --log_path=logs/Acrobot_0001_256 --save_path=models/Acrobot_0001_256.pkl --num_timesteps=1e7 --lr=0.0001 --nsteps=256
# atari_six: CUDA_VISIBLE_DEVICES=1 time python -m baselines.run --alg=a2c --env=Acrobot-v1 --log_path=logs/Acrobot_0005_256 --save_path=models/Acrobot_0005_256.pkl --num_timesteps=1e7 --lr=0.0005 --nsteps=256
# atari_seven: CUDA_VISIBLE_DEVICES=1 time python -m baselines.run --alg=a2c --env=Acrobot-v1 --log_path=logs/Acrobot_001_256 --save_path=models/Acrobot_001_256.pkl --num_timesteps=1e7 --lr=0.001 --nsteps=256
# atari_eight: CUDA_VISIBLE_DEVICES=1 time python -m baselines.run --alg=a2c --env=Acrobot-v1 --log_path=logs/Acrobot_005_256 --save_path=models/Acrobot_005_256.pkl --num_timesteps=1e7 --lr=0.005 --nsteps=256