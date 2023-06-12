import numpy as np
import matplotlib.pyplot as plt

title = "n=128"

cores = np.array([1,2,4,8,16,32,48])

t_cg = np.array([32.45, 17.84, 10.61, 6.207, 4.233, 3.061, 3.024])

t_jac_1 = np.array([31.21, 16.9, 9.903, 5.518, 3.535, 2.318, 2.102])
t_jac_2 = np.array([50.59, 27.35, 15.58, 8.532, 5.225, 3.467, 2.897])
t_jac_5 = np.array([48.18, 25.49, 14.05, 7.376, 4.25, 2.66, 2.245])
t_jac_10 = np.array([87.59, 46.59, 25.26, 12.96, 7.138, 4.307, 3.681])
t_jac_20 = np.array([117.06, 62.27, 33.6, 16.77, 9.062, 5.299, 4.82])

t_cheb_1 = np.array([31.91, 17.52, 10.26, 5.747, 3.877, 2.672, 2.426])
t_cheb_2 = np.array([67.24, 36.37, 21.17, 11.45, 7.446, 5.18, 4.636])
t_cheb_5 = np.array([67.62, 36.31, 20.94, 11.24, 6.51, 4.205, 3.572])
t_cheb_10 = np.array([64.72, 34.8, 19.97, 10.48, 5.615, 3.523, 2.937])
t_cheb_20 = np.array([62.93, 33.63, 19.26, 9.794, 5.291, 3.102, 2.573])



fig, ax = plt.subplots(1, 2, figsize=(10, 4)) # 1 row, 2 columns, plot 1
ax[0].plot(cores, t_jac_20, color='darkorange' , linestyle = '--' , marker = 'o', label='jac_20')
ax[0].plot(cores, t_jac_10, color = 'g', linestyle = '--' , marker = 'o', label='jac_10')
ax[0].plot(cores, t_jac_5, color = 'm', linestyle = '--' , marker = 'o', label='jac_5')
ax[0].plot(cores, t_jac_2, color = 'b', linestyle = '--' , marker = 'o', label='jac_2')
ax[0].plot(cores, t_jac_1, color ='r', linestyle = '--' , marker = 'o', label='jac_1')
ax[0].plot(cores, t_cg, color = 'k', linestyle = '--' , marker = 'o', label='cg')
ax[0].set_xlabel("cores",fontsize = 14)
ax[0].set_xscale('log', base=2)
ax[0].set_xticks(cores, cores)
ax[0].set_ylabel(r'time (sec)',fontsize = 14)
ax[0].set_yticks(np.arange(0, 121,10),np.arange(0, 121,10))
ax[0].set_title(r'Jacobi preconditioner',fontsize = 18)
ax[0].legend()

ax[1].plot(cores, t_cheb_20, color = 'darkorange', linestyle = '--' , marker = 'o', label='cheb_20')
ax[1].plot(cores, t_cheb_10, color = 'g', linestyle = '--' , marker = 'o', label='cheb_10')
ax[1].plot(cores, t_cheb_5, color = 'm', linestyle = '--' , marker = 'o', label='cheb_5')
ax[1].plot(cores, t_cheb_2, color = 'b', linestyle = '--' , marker = 'o', label='cheb_2')
ax[1].plot(cores, t_cheb_1, color = 'r', linestyle = '--' , marker = 'o', label='cheb_1')
ax[1].plot(cores, t_cg, color ='k', linestyle = '--' , marker = 'o', label='cg')
ax[1].set_xlabel("cores",fontsize = 14)
ax[1].set_xscale('log', base=2)
ax[1].set_xticks(cores, cores)
ax[1].set_ylabel(r'time (sec)',fontsize = 14)
ax[1].set_title(r'Chebyshev preconditioner',fontsize = 18)
ax[1].legend()

fig.suptitle(r"Computation time of (P)CG using",fontsize = 20)
plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.4, hspace=0.4)

fig.tight_layout()

plt.show()

fig.savefig('small_times_final.png')


