from baselines.common import plot_util as pu
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#plots for sweeping learning rate with batchsize-32, buffersize-1e4, model-TT
results1 = pu.load_results('logs/Acrobot/lr1e-4_bch32_bf1e4_FT', 1, 0, 0) 
results2 = pu.load_results('logs/Acrobot/lr5e-4_bch32_bf1e4_FT', 1, 0, 0)
results3 = pu.load_results('logs/Acrobot/lr1e-3_bch32_bf1e4_FT', 1, 0, 0)
results4 = pu.load_results('logs/Acrobot/lr5e-3_bch32_bf1e4_FT', 1, 0, 0)
r1 = results1[0]
r2 = results2[0]
r3 = results3[0]
r4 = results4[0]
plt.switch_backend("agg")
plt.figure(figsize=(20,3))
plt.subplot(121, title='Batch size-32, Buffer size-10000, Model-FT', xlabel='Timesteps', ylabel='Average reward')
plt.plot(r1.progress.steps, r1.progress["mean 100 episode reward"], label='lr-0.0001')
plt.plot(r2.progress.steps, r2.progress["mean 100 episode reward"], label='lr-0.0005')
plt.plot(r3.progress.steps, r3.progress["mean 100 episode reward"], label='lr-0.001')
plt.plot(r4.progress.steps, r4.progress["mean 100 episode reward"], label='lr-0.005')
plt.legend()
# plt.tight_layout()
# plt.subplots_adjust(bottom=0.1, right=1.3, top=0.3)
plt.subplot(122, xlabel='Learning rate', ylabel='Training time')
x = np.arange(4)
time = [453.7, 454.1, 466.2, 425.7]
plt.bar(x, time)
plt.xticks(x, ('0.0001', '0.0005', '0.001', '0.005'))
plt.savefig('pics/Acrobot_bch32_bf1e4_FT.png',  format='png')

#plots for sweeping learning rate with batchsize-256, buffersize-1e4, model-TT
results5 = pu.load_results('logs/Acrobot/lr1e-4_bch256_bf1e4_FT', 1, 0, 0)
results6 = pu.load_results('logs/Acrobot/lr5e-4_bch256_bf1e4_FT', 1, 0, 0)
results7 = pu.load_results('logs/Acrobot/lr1e-3_bch256_bf1e4_FT', 1, 0, 0)
results8 = pu.load_results('logs/Acrobot/lr5e-3_bch256_bf1e4_FT', 1, 0, 0)
r5 = results5[0]
r6 = results6[0]
r7 = results7[0]
r8 = results8[0]
plt.switch_backend("agg")
plt.figure(figsize=(20,3))
plt.subplot(121, title='Batch size-256, Buffer size-10000, Model-FT', xlabel='Timesteps', ylabel='Average reward')
plt.plot(r5.progress.steps, r5.progress["mean 100 episode reward"], label='lr-0.0001')
plt.plot(r6.progress.steps, r6.progress["mean 100 episode reward"], label='lr-0.0005')
plt.plot(r7.progress.steps, r7.progress["mean 100 episode reward"], label='lr-0.001')
plt.plot(r8.progress.steps, r8.progress["mean 100 episode reward"], label='lr-0.005')
plt.legend()
# plt.tight_layout()
# plt.subplots_adjust(bottom=0.1, right=1.3, top=0.3)
plt.subplot(122, xlabel='Learning rate', ylabel='Training time')
x = np.arange(4)
time = [453.6, 448.4, 450.5, 452.4]
plt.bar(x, time)
plt.xticks(x, ('0.0001', '0.0005', '0.001', '0.005'))
plt.savefig('pics/Acrobot_bch256_bf1e4_FT.png',  format='png')

#plots for sweeping learning rate with batchsize-32, buffersize-1e4, model-FF
results1 = pu.load_results('logs/Acrobot/lr1e-4_bch32_bf1e4_TF', 1, 0, 0) 
results2 = pu.load_results('logs/Acrobot/lr5e-4_bch32_bf1e4_TF', 1, 0, 0)
results3 = pu.load_results('logs/Acrobot/lr1e-3_bch32_bf1e4_TF', 1, 0, 0)
results4 = pu.load_results('logs/Acrobot/lr5e-3_bch32_bf1e4_TF', 1, 0, 0)
r1 = results1[0]
r2 = results2[0]
r3 = results3[0]
r4 = results4[0]
plt.switch_backend("agg")
plt.figure(figsize=(20,3))
plt.subplot(121, title='Batch size-32, Buffer size-10000, Model-TF', xlabel='Timesteps', ylabel='Average reward')
plt.plot(r1.progress.steps, r1.progress["mean 100 episode reward"], label='lr-0.0001')
plt.plot(r2.progress.steps, r2.progress["mean 100 episode reward"], label='lr-0.0005')
plt.plot(r3.progress.steps, r3.progress["mean 100 episode reward"], label='lr-0.001')
plt.plot(r4.progress.steps, r4.progress["mean 100 episode reward"], label='lr-0.005')
plt.legend()
# plt.legend(loc='upper left', bbox_to_anchor=(1.0, 0.7, 0.3, 0.2))
# plt.tight_layout()
# plt.subplots_adjust(bottom=0.1, right=1.3, top=0.3)
plt.subplot(122, xlabel='Learning rate', ylabel='Training time')
x = np.arange(4)
time = [451.6, 453.7, 455.1, 455.7]
plt.bar(x, time)
plt.xticks(x, ('0.0001', '0.0005', '0.001', '0.005'))
plt.savefig('pics/Acrobot_bch32_bf1e4_TF.png',  format='png')

#plots for sweeping learning rate with batchsize-256, buffersize-1e4, model-FF
results5 = pu.load_results('logs/Acrobot/lr1e-4_bch256_bf1e4_TF', 1, 0, 0)
results6 = pu.load_results('logs/Acrobot/lr5e-4_bch256_bf1e4_TF', 1, 0, 0)
results7 = pu.load_results('logs/Acrobot/lr1e-3_bch256_bf1e4_TF', 1, 0, 0)
results8 = pu.load_results('logs/Acrobot/lr5e-3_bch256_bf1e4_TF', 1, 0, 0)
r5 = results5[0]
r6 = results6[0]
r7 = results7[0]
r8 = results8[0]
plt.switch_backend("agg")
plt.figure(figsize=(20,3))
plt.subplot(121, title='Batch size-256, Buffer size-10000, Model-TF', xlabel='Timesteps', ylabel='Average reward')
plt.plot(r5.progress.steps, r5.progress["mean 100 episode reward"], label='lr-0.0001')
plt.plot(r6.progress.steps, r6.progress["mean 100 episode reward"], label='lr-0.0005')
plt.plot(r7.progress.steps, r7.progress["mean 100 episode reward"], label='lr-0.001')
plt.plot(r8.progress.steps, r8.progress["mean 100 episode reward"], label='lr-0.005')
plt.legend()
# plt.tight_layout()
# plt.subplots_adjust(bottom=0.1, right=1.3, top=0.3)
plt.subplot(122, xlabel='Learning rate', ylabel='Training time')
x = np.arange(4)
time = [453, 452.8, 453, 458.7]
plt.bar(x, time)
plt.xticks(x, ('0.0001', '0.0005', '0.001', '0.005'))
plt.savefig('pics/Acrobot_bch256_bf1e4_TF.png',  format='png')

# #plots for sweeping buffersize with learningrate-1e-4, batchsize-256, model-TT
# results1 = pu.load_results('logs/Acrobot/lr1e-4_bch256_bf1e2_TT', 1, 0, 0) 
# results2 = pu.load_results('logs/Acrobot/lr1e-4_bch256_bf1e3_TT', 1, 0, 0)
# results3 = pu.load_results('logs/Acrobot/lr1e-4_bch256_bf1e4_TT', 1, 0, 0)
# results4 = pu.load_results('logs/Acrobot/lr1e-4_bch256_bf1e5_TT', 1, 0, 0)
# r1 = results1[0]
# r2 = results2[0]
# r3 = results3[0]
# r4 = results4[0]
# plt.switch_backend("agg")
# plt.figure(figsize=(20,3))
# plt.subplot(121, title='Learning rate-0.0001, Batch size-256, Model-TT', xlabel='Timesteps', ylabel='Average reward')
# plt.plot(r1.progress.steps, r1.progress["mean 100 episode reward"], label='buffersize-100')
# plt.plot(r2.progress.steps, r2.progress["mean 100 episode reward"], label='buffersize-1000')
# plt.plot(r3.progress.steps, r3.progress["mean 100 episode reward"], label='buffersize-10000')
# plt.plot(r4.progress.steps, r4.progress["mean 100 episode reward"], label='buffersize-100000')
# plt.legend()
# plt.subplot(122, xlabel='Buffer size', ylabel='Training time')
# x = np.arange(4)
# time = [453.5, 453.3, 453.6, 453]
# plt.bar(x, time)
# plt.xticks(x, ('100', '1000', '10000', '100000'))
# plt.savefig('pics/Acrobot_lr1e-4_bch256_TT.png',  format='png')