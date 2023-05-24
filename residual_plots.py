import numpy as np
import matplotlib.pyplot as plt

n1_jacobi = np.genfromtxt("n4_j500_x1.csv", delimiter='')
n1_cg = np.genfromtxt("n4_cg_x1.csv",delimiter='')

#n32_pre = n32_pre[:-1]
#n32_cg = n32_cg[:-1] 

n2_jacobi = np.genfromtxt("n16_j500_x1.csv", delimiter='')
n2_cg = np.genfromtxt("n16_cg_x1.csv",delimiter='')

#n128_pre = n128_pre[:-1]
#n128_cg = n128_cg[:-1] 

#n1_jacobi = np.sqrt(n1_jacobi)
#n2_jacobi = np.sqrt(n2_jacobi)
#n1_cg = np.sqrt(n1_cg)
#n2_cg = np.sqrt(n2_cg)

plt.ion()

# Create the first plot
fig, ax = plt.subplots(2, 1) # 2 rows, 1 column, plot 1
ax[0].plot(range(len(n1_cg)), n1_cg, 'k-' , label='cg')
ax[0].plot(range(len(n1_jacobi)), n1_jacobi, 'r--', label='jac')
ax[0].set_xlabel("iterations")
ax[0].set_xlim([0,max(len(n1_cg),len(n1_jacobi))])
ax[0].set_ylabel(r'$\parallel x \parallel_2$')
ax[0].set_ylim([max(n1_cg[0],n1_jacobi[0]),max(n1_cg[-1],n1_jacobi[-1])+1])
#ax[0].set_xscale('log', base=10)
#ax[0].set_yscale('log', base=10)
ax[0].set_title('n=4')
ax[0].legend()


ax[1].plot(range(len(n2_cg)), n2_cg, 'k-' , label='cg')
ax[1].plot(range(len(n2_jacobi)), n2_jacobi, 'r--', label='jac')
#ax[1].set_xscale('log', base=10)
ax[1].set_xlabel("iterations")
ax[1].set_xlim([0,max(len(n2_cg),len(n2_jacobi))])
ax[1].set_ylabel(r'$\parallel x \parallel_2$')
ax[1].set_ylim([max(n2_cg[0],n2_jacobi[0]),max(n2_cg[-1],n2_jacobi[-1])+5])
#ax[1].set_yscale('log', base=10)
ax[1].set_title('n=16')
ax[1].legend()

fig.tight_layout()

fig.savefig('n4n16_x1.png')

plt.show()
