import numpy as np
import matplotlib.pyplot as plt

cg = np.genfromtxt("cg_32.txt")
jac_1 = np.genfromtxt("jac_32_1.txt")
jac_2 = np.genfromtxt("jac_32_2.txt")
jac_5 = np.genfromtxt("jac_32_5.txt")
jac_10 = np.genfromtxt("jac_32_10.txt")


cheb_1 = np.genfromtxt("cheb_32_1.txt")
cheb_2 = np.genfromtxt("cheb_32_2.txt")
cheb_5 = np.genfromtxt("cheb_32_5.txt")
cheb_10 = np.genfromtxt("cheb_32_10.txt")

r = cg[:,1]
x = cg[:,2]
r_jac_1 = jac_1[:,1]
x_jac_1 = jac_1[:,2]
r_jac_2 = jac_2[:,1]
x_jac_2 = jac_2[:,2]
r_jac_5 = jac_5[:,1]
x_jac_5 = jac_5[:,2]
r_jac_10 = jac_10[:,1]
x_jac_10 = jac_10[:,2]

r_cheb_1 = cheb_1[:,1]
x_cheb_1 = cheb_1[:,2]
r_cheb_2 = cheb_2[:,1]
x_cheb_2 = cheb_2[:,2]
r_cheb_5 = cheb_5[:,1]
x_cheb_5 = cheb_5[:,2]
r_cheb_10 = cheb_10[:,1]
x_cheb_10 = cheb_10[:,2]


plt.ion()

fig, ax = plt.subplots(2, 1) # 2 rows, 1 column, plot 1
ax[0].plot(range(len(x)), x, 'k-' , label='cg')
ax[0].plot(range(len(x_jac_1)), x_jac_1, 'r--', label='jac_1')
ax[0].plot(range(len(x_jac_2)), x_jac_2, 'b--', label='jac_2')
ax[0].plot(range(len(x_jac_5)), x_jac_5, 'm--', label='jac_5')
ax[0].plot(range(len(x_jac_10)), x_jac_10, 'g--', label='jac_10')
ax[0].set_xlabel("iterations")
ax[0].set_xlim([-15,max(len(x),len(x_jac_1),len(x_jac_2),len(x_jac_5), len(x_jac_10))])
ax[0].set_ylabel(r'$\parallel x \parallel_2$')
ax[0].set_ylim([max(x[0],x_jac_1[0],x_jac_2[0], x_jac_5[0], x_jac_10[0]),max(x[-1],x_jac_1[-1],x_jac_2[-1], x_jac_5[-1], x_jac_10[-1])+1])
ax[0].set_title('n=32')
ax[0].legend()

ax[1].plot(range(len(r)), r, 'k-' , label='cg')
ax[1].plot(range(len(r_jac_1)), r_jac_1, 'r--', label='jac_1')
ax[1].plot(range(len(r_jac_2)), r_jac_2, 'b--', label='jac_2')
ax[1].plot(range(len(r_jac_5)), r_jac_5, 'm--', label='jac_5')
ax[1].plot(range(len(r_jac_10)), r_jac_10, 'g--', label='jac_10')
ax[1].set_xlabel("iterations")
ax[1].set_xlim([-15,max(len(r),len(r_jac_1),len(r_jac_2),len(r_jac_5),len(r_jac_10))])
ax[1].set_ylabel(r'$\parallel r \parallel_2$')
#ax[1].set_xscale('log', base=10)
ax[1].set_yscale('log', base=10)
ax[1].set_title('n=32')
ax[1].legend()

fig.tight_layout()

fig.savefig('jac32.png')

fig1, ax1 = plt.subplots(2, 1)

fig1, ax1 = plt.subplots(2, 1) # 2 rows, 1 column, plot 1
ax1[0].plot(range(len(x)), x, 'k-' , label='cg')
ax1[0].plot(range(len(x_cheb_1)), x_cheb_1, 'r--', label='cheb_1')
ax1[0].plot(range(len(x_cheb_2)), x_cheb_2, 'b--', label='cheb_2')
ax1[0].plot(range(len(x_cheb_5)), x_cheb_5, 'm--', label='cheb_5')
ax1[0].plot(range(len(x_cheb_10)), x_cheb_10, 'g--', label='cheb_10')
ax1[0].set_xlabel("iterations")
ax1[0].set_xlim([-15,max(len(x),len(x_cheb_1),len(x_cheb_2),len(x_cheb_5), len(x_cheb_10))])
ax1[0].set_ylabel(r'$\parallel x \parallel_2$')
ax1[0].set_ylim([max(x[0],x_cheb_1[0],x_cheb_2[0], x_cheb_5[0], x_cheb_10[0]),max(x[-1],x_cheb_1[-1],x_cheb_2[-1], x_cheb_5[-1], x_cheb_10[-1])+1])
ax1[0].set_title('n=32')
ax1[0].legend()

ax1[1].plot(range(len(r)), r, 'k-' , label='cg')
ax1[1].plot(range(len(r_cheb_1)), r_cheb_1, 'r--', label='cheb_1')
ax1[1].plot(range(len(r_cheb_2)), r_cheb_2, 'b--', label='cheb_2')
ax1[1].plot(range(len(r_cheb_5)), r_cheb_5, 'm--', label='cheb_5')
ax1[1].plot(range(len(r_cheb_10)), r_cheb_10, 'g--', label='cheb_10')
ax1[1].set_xlabel("iterations")
ax1[1].set_xlim([-15,max(len(r),len(r_cheb_1),len(r_cheb_2),len(r_cheb_5),len(r_cheb_10))])
ax1[1].set_ylabel(r'$\parallel r \parallel_2$')
#ax[1].set_xscale('log', base=10)
ax1[1].set_yscale('log', base=10)
ax1[1].set_title('n=32')
ax1[1].legend()

fig1.tight_layout()

fig1.savefig('cheb32.png')

plt.show()
