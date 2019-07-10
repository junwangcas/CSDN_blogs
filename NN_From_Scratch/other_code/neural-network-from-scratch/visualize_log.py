import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.cm as cm
file_w = open("data/w.txt",'rb')
file_b = open("data/b.txt",'rb')
file_forward = open("data/forward.txt",'rb')
## read log
ws = pickle.load(file_w)
bs = pickle.load(file_b)
forwards = pickle.load(file_forward)
## visualize
# fig, ax = plt.subplots()
# ws_1 = ws[0][0]
# ax.matshow(ws_1, cmap = cm.Greys_r)
# for (i, j), z in np.ndenumerate(ws_1):
#     ax.text(j, i, '{:0.1f}'.format(z), ha='center', va='center',
#             bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))
# plt.show()

# show forward layers
plt.subplot(2,2,1)
a_1 = forwards[0][0][2] # (200, 2)
plt.hist(x=a_1[:,0])
plt.subplot(2,2,2)
plt.hist(x=a_1[:,1])
plt.show()

