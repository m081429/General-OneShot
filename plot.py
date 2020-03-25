import matplotlib.pyplot as plt
epoch_lst=[1,2,3,4,5,6,7,8,9,10]
all_epoch_loss=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.9,0.1,0.2]
all_epoch_val_loss=[0.2,0.3,0.3,0.4,0.5,0.7,0.7,0.9,0.95,0.98]
#fig, axes = plt.subplots(frameon=True, figsize=(12, 8))
#fig.suptitle('Training Metrics')
#axes.set_ylabel("Loss", fontsize=14)
#axes.set_xlabel("Epoch", fontsize=14)
#axes.plot(epoch_lst,all_epoch_loss,color='k',label="train")
#axes.plot(epoch_lst,all_epoch_val_loss,color='g',label="val")
#axes.legend(loc="upper left")
#plt.ylim(-1.5, 2.0)
#plt.show()
#fig.savefig('Loss.pdf', bbox_inches='tight')

fig, axes = plt.subplots(2, sharex=True, figsize=(12, 8))
fig.suptitle('Training Metrics')
axes[0].set_ylabel("Loss", fontsize=14)
axes[0].plot(epoch_lst,all_epoch_loss,color='k',label="train")
axes[0].plot(epoch_lst,all_epoch_val_loss,color='g',label="val")
axes[0].legend(loc="upper left")
axes[1].set_ylabel("Accuracy", fontsize=14)
axes[1].set_xlabel("Epoch", fontsize=14)
axes[1].plot(epoch_lst,all_epoch_loss,color='k',label="train")
axes[1].plot(epoch_lst,all_epoch_val_loss,color='g',label="val")
axes[1].legend(loc="upper left")
plt.show()
fig.savefig('Loss.pdf', bbox_inches='tight')
