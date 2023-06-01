import numpy as np
import matplotlib.pyplot as plt

cg_3248 = np.genfromtxt("cg_poisson_3248.txt")
cg_jac_3248 = np.genfromtxt("cg_poisson_jacobi_500it_3248.txt")
cg_cheb_3248 = np.genfromtxt("cg_poisson_cheb_500it_exp_3248.txt")
cg_cheb_3248_100 = np.genfromtxt("cg_poisson_cheb_100it_exp_3248.txt")
cg_cheb_3248_100pm = np.genfromtxt("cg_poisson_cheb_100it_pm_3248.txt")

res_cg = cg_3248[:,1]
x_cg = cg_3248[:,2]
res_jac = cg_jac_3248[:,1]
x_jac = cg_jac_3248[:,2]
res_cheb = cg_cheb_3248[:,1]
x_cheb = cg_cheb_3248[:,2]

res_cheb_100 = cg_cheb_3248_100[:,1]
x_cheb_100 = cg_cheb_3248_100[:,2]
res_cheb_100pm = cg_cheb_3248_100pm[:,1]
x_cheb_100pm = cg_cheb_3248_100pm[:,2]

plt.ion()

fig, ax = plt.subplots(2, 1) # 2 rows, 1 column, plot 1
ax[0].plot(range(len(x_cg)), x_cg, 'k-' , label='cg')
ax[0].plot(range(len(x_jac)), x_jac, 'r--', label='jac')
ax[0].plot(range(len(x_cheb)), x_cheb, 'b--', label='cheb')
ax[0].set_xlabel("iterations")
ax[0].set_xlim([-15,max(len(x_cg),len(x_jac),len(x_cheb))])
ax[0].set_ylabel(r'$\parallel x \parallel_2$')
ax[0].set_ylim([max(x_cg[0],x_jac[0],x_cheb[0]),max(x_cg[-1],x_jac[-1],x_cheb[-1])+1])
#ax[0].set_xscale('log', base=10)
#ax[0].set_yscale('log', base=10)
ax[0].set_title('n=32')
ax[0].legend()

ax[1].plot(range(len(res_cg)), res_cg, 'k-' , label='cg')
ax[1].plot(range(len(res_jac)), res_jac, 'r--', label='jac')
ax[1].plot(range(len(res_cheb)), res_cheb, 'b--', label='cheb')
ax[1].set_xlabel("iterations")
ax[1].set_xlim([-15,max(len(res_cg),len(res_jac),len(res_cheb))])
ax[1].set_ylabel(r'$\parallel r \parallel_2$')
#ax[1].set_xscale('log', base=10)
#ax[1].set_yscale('log', base=10)
ax[1].set_title('n=32')
ax[1].legend()

fig.tight_layout()

fig.savefig('log_n32.png')


fig1, ax1 = plt.subplots(2, 1)

ax1[0].plot(range(len(x_cheb_100)), x_cheb_100, 'r-x', label='cheb_exp_500it')
ax1[0].plot(range(len(x_cheb)), x_cheb, 'k-' , label='cheb_exp_100it')
ax1[0].plot(range(len(x_cheb_100pm)), x_cheb_100pm, 'b--', label='cheb_pm_100it')
ax1[0].set_xlabel("iterations")
ax1[0].set_xlim([-15,max(len(x_cheb),len(x_cheb_100),len(x_cheb_100pm))])
ax1[0].set_ylabel(r'$\parallel x \parallel_2$')
ax1[0].set_ylim([max(x_cheb[0],x_cheb_100[0],x_cheb_100pm[0]),max(x_cheb[-1],x_cheb_100[-1],x_cheb_100pm[-1])+1])
#ax1[0].set_xscale('log', base=10)
#ax1[0].set_yscale('log', base=10)
ax1[0].set_title('n=32')
ax1[0].legend()

ax1[1].plot(range(len(res_cheb_100)), res_cheb_100, 'r-x', label='cheb_exp_500it')
ax1[1].plot(range(len(res_cheb)), res_cheb, 'k-' , label='cheb_exp_100it')
ax1[1].plot(range(len(res_cheb_100pm)), res_cheb_100pm, 'b--', label='cheb_pm_100it')
ax1[1].set_xlabel("iterations")
ax1[1].set_xlim([-15,max(len(res_cheb),len(res_cheb_100),len(res_cheb_100pm))])
ax1[1].set_ylabel(r'$\parallel r \parallel_2$')
#ax1[0].set_xscale('log', base=10)
#ax1[0].set_yscale('log', base=10)
ax1[1].set_title('n=32')
ax1[1].legend()

fig1.tight_layout()

fig1.savefig('cheb_n32.png')

plt.show()
