#%%
import matplotlib.pyplot as plt
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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
with open('result/simulation1.pickle', 'rb') as f:
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

# %%
task_angles = np.linspace(0,np.pi/6, total_task)

sns.set_context('talk')
fig, ax = plt.subplots(1,1, figsize=(8,8))
labelsize = 28
ticksize = 24
legendsize = 20

for ii in range(total_task):
    ax.plot(task_angles[ii:total_task],transfer[ii], linewidth=4, marker='o', label='Task '+str(ii+1))

ax.set_xlabel('Tasks (angle)', fontsize=labelsize)
ax.set_ylabel('Transfer', fontsize=labelsize)

ax.set_title(r'$\theta$-xor tasks', fontsize=labelsize+2)
leg = fig.legend(
    fontsize=legendsize,
    frameon=False,
    bbox_to_anchor=(0.53,-.25),
    bbox_transform=plt.gcf().transFigure,
    ncol=3,
    loc="lower center",
)
ax.tick_params(labelsize=ticksize)
right_side = ax.spines["right"]
right_side.set_visible(False)
top_side = ax.spines["top"]
top_side.set_visible(False)

plt.savefig('plots/xor_tasks.pdf')
# %%
