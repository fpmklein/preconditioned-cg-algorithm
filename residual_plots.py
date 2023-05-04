import numpy as np
import matplotlib.pyplot as plt

n32_pre = np.genfromtxt("n32_pre.csv", delimiter='')
n32_cg = np.genfromtxt("n32_cg.csv",delimiter='')

#n32_pre = n32_pre[:-1]
#n32_cg = n32_cg[:-1] 

n128_pre = np.genfromtxt("n128_pre.csv", delimiter='')
n128_cg = np.genfromtxt("n128_cg.csv",delimiter='')

#n128_pre = n128_pre[:-1]
#n128_cg = n128_cg[:-1] 


# Create the first plot
fig, ax = plt.subplots(2, 1) # 2 rows, 1 column, plot 1
ax[0].plot(range(len(n32_cg)), n32_cg, 'k-' , label='cg')
ax[0].plot(range(len(n32_pre)), n32_pre, 'r--', label='pre')
#ax[0].set_xscale('log', base=10)
ax[0].set_yscale('log', base=10)
ax[0].set_title('n=32')
ax[0].legend()


ax[1].plot(range(len(n128_cg)), n128_cg, 'k-' , label='cg')
ax[1].plot(range(len(n128_pre)), n128_pre, 'r--', label='pre')
#ax[1].set_xscale('log', base=10)
ax[1].set_yscale('log', base=10)
ax[1].set_title('n=128')
ax[1].legend()

plt.show()
