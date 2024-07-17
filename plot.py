#%%
import matplotlib.pyplot as plt
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd 
#%%
def avg_err(multitask, singletask):
    total_task = len(multitask[0])
    reps = len(multitask)
    avg = []

    for jj in range(total_task):
        for ii in range(reps):
            if ii == 0:
                sum = 1-np.array(multitask[ii][jj])
            else:
                sum += 1-np.array(multitask[ii][jj])
        avg.append(sum/reps)

    for ii in range(reps):
        if ii == 0:
            avgsingle = 1-np.array(singletask[ii])
        else:
            avgsingle += 1-np.array(singletask[ii])
    avgsingle /= 10

    return avg, avgsingle
# %%
with open('result/simulation1_no_replay.pickle', 'rb') as f:
    multitask, singletask = pickle.load(f)
# %%
avg_multitask, avg_singletask = avg_err(multitask, singletask)
#%%
total_task = len(multitask)
transfer = []
for ii in range(total_task):
    tx = np.zeros(total_task-ii, dtype=float)
    for jj in range(ii, total_task):
        tx[jj-ii] = np.log(avg_singletask[ii]/avg_multitask[jj][ii])
    transfer.append(tx)

#%%
with open('result/simulation1.pickle', 'rb') as f:
    multitask, singletask = pickle.load(f)
# %%
avg_multitask, avg_singletask = avg_err(multitask, singletask)
#%%
total_task = len(multitask)
transfer_replay = []
for ii in range(total_task):
    tx = np.zeros(total_task-ii, dtype=float)
    for jj in range(ii, total_task):
        tx[jj-ii] = np.log(avg_singletask[ii]/avg_multitask[jj][ii])
    transfer_replay.append(tx)
# %%
task_angles = np.linspace(0,np.pi/6, total_task)

sns.set_context('talk')
fig, ax = plt.subplots(1,2, figsize=(16,8), sharex=True, sharey=True)
labelsize = 28
ticksize = 24
legendsize = 20

for ii in range(total_task):
    ax[0].plot(task_angles[ii:total_task],transfer[ii], linewidth=4, marker='o', label='Task '+str(ii+1))
    ax[1].plot(task_angles[ii:total_task],transfer_replay[ii], linewidth=4, marker='o')


ax[0].set_title(r'No replay', fontsize=labelsize+2)
ax[1].set_title(r'With replay', fontsize=labelsize+2)

leg = fig.legend(
    fontsize=legendsize,
    frameon=False,
    bbox_to_anchor=(0.53,-.25),
    bbox_transform=plt.gcf().transFigure,
    ncol=5,
    loc="lower center",
)

for ii in range(2):
    ax[ii].tick_params(labelsize=ticksize)
    right_side = ax[ii].spines["right"]
    right_side.set_visible(False)
    top_side = ax[ii].spines["top"]
    top_side.set_visible(False)

fig.text(0.4,0,'Task angle (radian)', fontsize=labelsize)
ax[0].set_ylabel('Transfer', fontsize=labelsize)
ax[0].set_xticks([0,.3,.5])

plt.savefig('plots/xor_tasks.png',bbox_inches='tight')
# %% plot SNR
SNR = [-40,-30,-20,-10,0,10,20,30,40]
transfer = []

for snr in SNR:
    print('Doing ', snr)
    with open('result/simulation_SNR_'+str(snr)+'.pickle', 'rb') as f:
        multitask = pickle.load(f)
    
    multitask = multitask[-10:]
    print(len(multitask))
    avg_multitask, avg_singletask = avg_err(multitask, singletask.copy())

    total_task = len(multitask)

    tx = []
    for ii in range(total_task):
        tx.append(np.log(avg_singletask[ii]/avg_multitask[-1][ii]))
    transfer.append(tx)

# %%
 te = {}

for ii,snr in enumerate(SNR):
    te[snr] = transfer[ii]

df = pd.DataFrame.from_dict(te)
df = pd.melt(df,var_name='SNR', value_name='Transfer')

# %%
sns.set_context('talk')
fig, ax = plt.subplots(1,1, figsize=(8,8))

sns.stripplot(x='SNR', y='Transfer', data=df, ax=ax, size=25, legend=None)

ax.set_ylabel('Transfer', fontsize=labelsize)
ax.set_xlabel('Replay SNR (dB)', fontsize=labelsize)

ax.set_yticks([0,.6,1.2])
ax.tick_params(labelsize=ticksize)
right_side = ax.spines["right"]
right_side.set_visible(False)
top_side = ax.spines["top"]
top_side.set_visible(False)

plt.savefig('plots/noisy_replay.png',bbox_inches='tight')
# %%
