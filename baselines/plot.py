from baselines.common import plot_util as pu
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

results = pu.load_results('logs/CartPole', 1, 0, 0) 
r = results[0]
plt.switch_backend("agg")
plt.plot(r.progress.steps, r.progress["mean 100 episode reward"])
#pu.plot_results(results)
plt.savefig('pics/CartPole.png',  format='png')

results = pu.load_results('logs/MountainCar', 1, 0, 0) 
r = results[0]
plt.switch_backend("agg")
plt.plot(r.progress.steps, r.progress["mean 100 episode reward"])
#pu.plot_results(results)
plt.savefig('pics/MountainCar.png',  format='png')

results = pu.load_results('logs/Acrobot', 1, 0, 0) 
r = results[0]
plt.switch_backend("agg")
plt.plot(r.progress.steps, r.progress["mean 100 episode reward"])
#pu.plot_results(results)
plt.savefig('pics/Acrobot.png',  format='png')