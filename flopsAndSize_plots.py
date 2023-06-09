import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors as mcolors

def flops(function_name = "", nx = 32, ny = 32, nz = 32, cg_it = 1000, prec_it = 0):
    n = nx*ny*nz
    flp_init = 0.0
    flp_stencil3d = (6 + 7)*(nx - 2)*(ny - 2)*(nz - 2) + (5 + 6)*2*((nx - 2)*(ny - 2) +(nx - 2)*(nz - 2)+(nz - 2)*(ny - 2)) + (4 + 5)*4*((nx - 2)+ (ny - 2) + (nz - 2)) + (3 + 4)*8
    flp_axy = 1.0 * n
    flp_axpby = 3.0 * n
    flp_dot = 2.0 * n
    flp_copy = 0.0
    flp_explEV = 2.0 + 3.0*2.0 + 5.0 + 31.0 + 1.0
    flp_jac = 2.0 + (1.0 + flp_axy) + prec_it * (flp_stencil3d + 3.0*n)
    flp_cheb = 2.0 + 2.0 + 2.0 + 1.0 + 1.0 + (1.0 + flp_axy) + flp_stencil3d + (5.0 + flp_axpby) + prec_it * (1.0 + 3.0 + 3.0 + 4.0 * flp_axpby + flp_stencil3d + flp_copy)
    flp_cg = 1.0 + flp_dot + 1.0 + 3.0*flp_axpby + flp_stencil3d + flp_dot + 1.0
    flp_cgIntro = flp_stencil3d + flp_axpby + 2.0 * flp_init

    tot_flp_init = 2.0*flp_init
    tot_flp_stencil3d = (1.0 + 1.0*cg_it)*flp_stencil3d
    tot_flp_axy = cg_it*flp_axy
    tot_flp_axpby = (1.0 + 3.0*cg_it)*flp_axpby
    tot_flp_dot = 2.0*cg_it*flp_dot
    tot_flp_copy = 1.0*cg_it*flp_copy
    tot_flp_jac = 1.0*cg_it*flp_jac
    tot_flp_cheb = 1.0*cg_it*flp_cheb
    tot_flp_cg = flp_cgIntro + 1.0*cg_it*flp_cg
    tot_flp_jac = flp_cgIntro + 1.0*flp_init + cg_it*(flp_cg + flp_jac)
    tot_flp_cheb = flp_cgIntro + 1.0*flp_init + flp_explEV + cg_it*(flp_cg + flp_cheb)
    if function_name == "init":
        flp = tot_flp_init
    elif function_name == "apply_stencil3d":
        flp = tot_flp_stencil3d
    elif function_name == "axy":
        flp = tot_flp_axy
    elif function_name == "axpby":
        flp = tot_flp_axpby
    elif function_name == "dot":
        flp = tot_flp_dot 
    elif function_name == "copy":
        flp = tot_flp_copy
    elif function_name == "explicit_ev":
        flp = tot_flp_explEV
    elif function_name == "jac_method":
        flp = tot_flp_jac
    elif function_name == "cheb_method":
        flp = tot_flp_cheb
    elif function_name == "cg":
        flp = tot_flp_cg
    elif function_name == "pcg_jac":
        flp = tot_flp_jac
    elif function_name == "pcg_cheb":
        flp = tot_flp_cheb
    else:
        flp = 0
        print("no such a function, flops = 0 taken")
    Gflp = flp * 10**(-9)
    print(f'Gflp_{function_name}_{prec_it} = ', Gflp)
    return Gflp

def rate_per_core(lst, time_per_core):
    rate = np.zeros(len(time_per_core))
    for i in range(len(time_per_core)):
        rate[i] = lst[i] / time_per_core[i]
    return rate

def plot_flps(cores, cg, jac, cheb, title = ""):
    fig, ax = plt.subplots(1, 2) # 2 rows, 1 column, plot 1
    ax[0].plot(cores, jac[4], color = "darkorange", linestyle = '--', marker = 'o', label='jac_20')
    ax[0].plot(cores, jac[3], 'g--', marker = 'o', label='jac_10')
    ax[0].plot(cores, jac[2], 'm--', marker = 'o', label='jac_5')
    ax[0].plot(cores, jac[1], 'b--', marker = 'o', label='jac_2')
    ax[0].plot(cores, jac[0], 'r--', marker = 'o', label='jac_1')
    ax[0].plot(cores, cg, 'k-' , marker = 'o', label='cg')
    
    ax[0].set_xlabel("cores", fontsize = 14)
    #ax[0].set_xlim([-15,max(len(x),len(x_jac_1),len(x_jac_2),len(x_jac_5), len(x_jac_10))])
    ax[0].set_ylabel("Gflop Rate (Gflops/sec)", fontsize = 14)
    #ax[0].set_ylim([max(x[0],x_jac_1[0],x_jac_2[0], x_jac_5[0], x_jac_10[0]),max(x[-1],x_jac_1[-1],x_jac_2[-1], x_jac_5[-1], x_jac_10[-1])+1])
    #ax[0].set_yscale('log', base=10)
    ax[0].set_xscale('log', base=2)
    ax[0].set_xticks(cores, cores)
    ax[0].set_title(f'Jacobi', fontsize = 18)
    ax[0].legend()

    ax[1].plot(cores, cheb[4], color = "darkorange", linestyle = '--', marker = 'o', label='cheb_20')
    ax[1].plot(cores, cheb[3], 'g--', marker = 'o', label='cheb_10')
    ax[1].plot(cores, cheb[2], 'm--', marker = 'o', label='cheb_5')
    ax[1].plot(cores, cheb[1], 'b--', marker = 'o', label='cheb_2')
    ax[1].plot(cores, cheb[0], 'r--', marker = 'o', label='cheb_1')
    ax[1].plot(cores, cg, 'k-' , marker = 'o', label='cg')

    ax[1].set_xlabel("cores", fontsize = 14)
    #ax[1].set_xlim([-15,max(len(r),len(r_jac_1),len(r_jac_2),len(r_jac_5),len(r_jac_10))])
    ax[1].set_ylabel("Gflop Rate (Gflops/sec)", fontsize = 14)
    #ax[1].set_yscale('log', base=10)
    ax[1].set_xscale('log', base=2)
    ax[1].set_xticks(cores, cores)
    ax[1].set_title('Chebyshev', fontsize = 18)
    ax[1].legend()

    #plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.4, hspace=0.4)
    fig.suptitle(f'{title}\n\nGflop Rate of (P)CG using', fontsize = 20)
    fig.tight_layout()
    
    plt.show()
    fig.savefig('flopsRate_n=128^3.png')
    
def plot_20it_flps(cores, flpRate_cg, flpRate_jac, flpRate_cheb):
    fig, ax = plt.subplots(1, 1) # 2 rows, 1 column, plot 1
    legenda_cg = [r"$n=32^3$: cg ", r"$n=64^3$: cg ", r"$n=128^3$: cg ", r"$n=256^3$: cg ", r"$n=512^3$: cg "]
    legenda_jac = [r"$n=32^3$: jac_20", r"$n=64^3$: jac_20", r"$n=128^3$: jac_20", r"$n=256^3$: jac_20", r"$n=512^3$: jac_20"]
    legenda_cheb = [r"$n=32^3$: cheb_20", r"$n=64^3$: cheb_20", r"$n=128^3$: cheb_20", r"$n=256^3$: cheb_20", r"$n=512^3$: cheb_20"]
    lineStyles = ["-", "--", ":"]
    
    col = ['darkgreen', 'limegreen', 'lawngreen', 'firebrick', 'red', 'darkorange', 'mediumblue', 'cornflowerblue', 'darkturquoise', 'purple', 'mediumorchid','magenta', 'black', 'rosybrown', 'gray']

    for i in range(len(flpRate_cg))[::-1]:
        ax.plot(cores, flpRate_cg[i,:], color = col[3*i], linestyle = lineStyles[0], marker = 'o', label=legenda_cg[i])
        ax.plot(cores, flpRate_jac[i,:], color = col[3*i+1], linestyle = lineStyles[1], marker = 'o', label=legenda_jac[i])
        ax.plot(cores, flpRate_cheb[i,:], color = col[3*i+2], linestyle = lineStyles[2], marker = 'o', label=legenda_cheb[i])
        ax.set_xlabel("cores", fontsize = 14)
        ax.set_ylabel("Gflop Rate (Gflops/sec)", fontsize = 14)
        ax.set_title("Gflop Rate of (P)CG using 20 Jacoib or Chebyshev iterations", fontsize = 20)
    ax.set_xscale('log', base=2)
    ax.set_xticks(cores, cores)
    #ax.set_yscale('log', base=10)
    ax.legend(loc = "upper left", ncol=2)

    fig.tight_layout()

    plt.show()

    fig.savefig('flopsRate_jaccheb20it.png')

def plot_size(size, cg, jac, cheb, title = "", y_title = ""):
    fig, ax = plt.subplots(1, 1) # 2 rows, 1 column, plot 1
    #legenda_cg = ["c=1: cg ", "c=2: cg ", "c=4: cg ", "c=8: cg ", "c=16: cg ", "c=32: cg ", "c=48: cg "]
    #legenda_jac = ["c=1: pcg using 20 jac it", "c=2: pcg using 20 jac it", "c=4: pcg using 20 jac it",
    #               "c=8: pcg using 20 jac it", "c=16: pcg using 20 jac it", "c=32: pcg using 20 jac it", "c=48: pcg using 20 jac it"]
    #legenda_cheb = ["c=1: pcg using 20 cheb it", "c=2: pcg using 20 cheb it", "c=4: pcg using 20 cheb it",
    #                "c=8: pcg using 20 cheb it", "c=16: pcg using 20 cheb it", "c=32: pcg using 20 cheb it", "c=48: pcg using 20 cheb it"]
    legenda = ["cg", "jac_20", "cheb_20"]
    lineStyles = ["-", "--", "--"]

    #col = ['darkgreen', 'limegreen', 'lawngreen', 'firebrick', 'red', 'coral', 'mediumblue', 'cornflowerblue', 'darkturquoise',
    #       'purple', 'mediumorchid','magenta', 'black', 'rosybrown', 'gray', 'saddlebrown', 'sandybrown', 'darkorange',
    #       'olive','gold', 'yellow']
    col = ['k', 'r', 'b']
    ax.plot(size, cg, color = col[0], linestyle = lineStyles[0], marker = 'o', label=legenda[0])
    ax.plot(size, jac, color = col[1], linestyle = lineStyles[1], marker = 'o', label=legenda[1])
    ax.plot(size, cheb, color = col[2], linestyle = lineStyles[2], marker = 'o', label=legenda[2])
    ax.set_xlabel(r'Matrix size ($n$)', fontsize = 14)
    ax.set_ylabel(y_title, fontsize = 14)
    ax.set_title(title, fontsize = 20)
    ax.set_xscale('log', base=2)
    ax.set_yscale('log', base=10)
    ax.set_xticks(size, size)
    #ax.legend(loc = "upper left", ncol=2)
    ax.legend(loc = "upper left")
    fig.tight_layout()

    plt.show()

    fig.savefig(title)
    

#-------------------------------------------start of main---------------------------------------------------------------
if __name__ == "__main__":
    #cores and matrix sizes n
    cores = [1,2,4,8,16,32,48]
    size = [32, 64, 128, 256, 512]
    
    ############### dim(A)=128^3 x 128^3 ####################################################
    #convergence number of cg iterations for cores = [1,2,4,8,16,32,48], len(cores)=7
    n128_it_cg = 598 * np.ones(7)

    n128_it_jac_1 = 299 * np.ones(7)
    n128_it_jac_2 = 345 * np.ones(7)
    n128_it_jac_5 = 173 * np.ones(7)
    n128_it_jac_10 = 179 * np.ones(7)
    n128_it_jac_20 = 128 * np.ones(7)

    n128_it_cheb_1 = 299 * np.ones(7)
    n128_it_cheb_2 = 418 * np.ones(7)
    n128_it_cheb_5 = 217 * np.ones(7)
    n128_it_cheb_10 = 115 * np.ones(7)
    n128_it_cheb_20 = 59 * np.ones(7)

    #Total computation time cg for cores = [1,2,4,8,16,32,48]
    n128_time_cg = np.array([32.45, 17.84, 10.61, 6.207, 4.233, 3.061, 3.024])

    n128_time_jac_1 = np.array([31.21, 16.9, 9.903, 5.518, 3.535, 2.318, 2.102])
    n128_time_jac_2 = np.array([50.59, 27.35, 15.58, 8.532, 5.225, 3.467, 2.897])
    n128_time_jac_5 = np.array([48.18, 25.49, 14.05, 7.376, 4.25, 2.66, 2.245])
    n128_time_jac_10 = np.array([87.59, 46.59, 25.26, 12.96, 7.138, 4.307, 3.681])
    n128_time_jac_20 = np.array([117.06, 62.27, 33.6, 16.77, 9.062, 5.299, 4.82])

    n128_time_cheb_1 = np.array([31.91, 17.52, 10.26, 5.747, 3.877, 2.672, 2.426])
    n128_time_cheb_2 = np.array([67.24,	36.37, 21.17, 11.45, 7.446, 5.18, 4.636])
    n128_time_cheb_5 = np.array([67.62, 36.31, 20.94, 11.24, 6.51, 4.205, 3.572])
    n128_time_cheb_10 = np.array([64.72, 34.8, 19.97, 10.48, 5.615, 3.523, 2.937])
    n128_time_cheb_20 = np.array([62.93, 33.63, 19.26, 9.794, 5.291, 3.102, 2.573])

    #flops
    n128_flp_cg = flops("cg", nx = 128, ny = 128, nz = 128, cg_it = n128_it_cg)

    n128_flp_jac_1 = flops("pcg_jac", nx = 128, ny = 128, nz = 128, cg_it = n128_it_jac_1, prec_it = 1)
    n128_flp_jac_2 = flops("pcg_jac", nx = 128, ny = 128, nz = 128, cg_it = n128_it_jac_2, prec_it = 2)
    n128_flp_jac_5 = flops("pcg_jac", nx = 128, ny = 128, nz = 128, cg_it = n128_it_jac_5, prec_it = 5)
    n128_flp_jac_10 = flops("pcg_jac", nx = 128, ny = 128, nz = 128, cg_it = n128_it_jac_10, prec_it = 10)
    n128_flp_jac_20 = flops("pcg_jac", nx = 128, ny = 128, nz = 128, cg_it = n128_it_jac_20, prec_it = 20)


    n128_flp_cheb_1 = flops("pcg_cheb", nx = 128, ny = 128, nz = 128, cg_it = n128_it_cheb_1, prec_it = 1)
    n128_flp_cheb_2 = flops("pcg_cheb", nx = 128, ny = 128, nz = 128, cg_it = n128_it_cheb_2, prec_it = 2)
    n128_flp_cheb_5 = flops("pcg_cheb", nx = 128, ny = 128, nz = 128, cg_it = n128_it_cheb_5, prec_it = 5)
    n128_flp_cheb_10 = flops("pcg_cheb", nx = 128, ny = 128, nz = 128, cg_it = n128_it_cheb_10, prec_it = 10)
    n128_flp_cheb_20 = flops("pcg_cheb", nx = 128, ny = 128, nz = 128, cg_it = n128_it_cheb_20, prec_it = 20)

    #testing flops
##    flops("dot", nx = 128, ny = 128, nz = 128, cg_it=598)
##    flops("axpby", nx = 128, ny = 128, nz = 128, cg_it=598)
##    flops("apply_stencil3d", nx = 128, ny = 128, nz = 128, cg_it=598)
##    flops("init", nx = 128, ny = 128, nz = 128, cg_it=598)
    
    #flopRate
    n128_flpRate_cg = rate_per_core(n128_flp_cg, n128_time_cg)

    n128_flpRate_jac_1 = rate_per_core(n128_flp_jac_1, n128_time_jac_1)
    n128_flpRate_jac_2 = rate_per_core(n128_flp_jac_2, n128_time_jac_2)
    n128_flpRate_jac_5 = rate_per_core(n128_flp_jac_5, n128_time_jac_5)
    n128_flpRate_jac_10 = rate_per_core(n128_flp_jac_10, n128_time_jac_10)
    n128_flpRate_jac_20 = rate_per_core(n128_flp_jac_20, n128_time_jac_20)

    n128_flpRate_cheb_1 = rate_per_core(n128_flp_cheb_1, n128_time_cheb_1)
    n128_flpRate_cheb_2 = rate_per_core(n128_flp_cheb_2, n128_time_cheb_2)
    n128_flpRate_cheb_5 = rate_per_core(n128_flp_cheb_5, n128_time_cheb_5)
    n128_flpRate_cheb_10 = rate_per_core(n128_flp_cheb_10, n128_time_cheb_10)
    n128_flpRate_cheb_20 = rate_per_core(n128_flp_cheb_20, n128_time_cheb_20)

    n128_flpRate_jac = [n128_flpRate_jac_1, n128_flpRate_jac_2, n128_flpRate_jac_5, n128_flpRate_jac_10, n128_flpRate_jac_20]
    n128_flpRate_cheb = [n128_flpRate_cheb_1, n128_flpRate_cheb_2, n128_flpRate_cheb_5, n128_flpRate_cheb_10, n128_flpRate_cheb_20]

########## plot#########################################
    plot_flps(cores, n128_flpRate_cg, n128_flpRate_jac, n128_flpRate_cheb, r'$n = 128^3$')

####################prec only 20 it ###############################################################
    ############### dim(A)=32^3 x 32^3 ####################################################
    #convergence number of cg iterations for cores = [1,2,4,8,16,32,48]
    n32_it_cg = 143 * np.ones(7)
    n32_it_jac_20 = 28 * np.ones(7)
    n32_it_cheb_20 = 13 * np.ones(7)

    #Total computation time cg for cores = [1,2,4,8,16,32,48]
    n32_time_cg = np.array([0.1251, 0.0665, 0.04042,  0.02718, 0.02325, 0.04069, 0.04426])
    n32_time_jac_20 = np.array([0.3985, 0.2211, 0.1246, 0.07325, 0.05554, 0.09248, 0.119])
    n32_time_cheb_20 = np.array([0.2049, 0.108,  0.06296, 0.03929, 0.03154, 0.05583, 0.06894])

    #flops
    n32_flp_cg = flops("cg", nx = 32, ny = 32, nz = 32, cg_it = n32_it_cg)
    n32_flp_jac_20 = flops("pcg_jac", nx = 32, ny = 32, nz = 32, cg_it = n32_it_jac_20, prec_it = 20)
    n32_flp_cheb_20 = flops("pcg_cheb", nx = 32, ny = 32, nz = 32, cg_it = n32_it_cheb_20, prec_it = 20)


    #flopRate
    n32_flpRate_cg = rate_per_core(n32_flp_cg, n32_time_cg)
    n32_flpRate_jac_20 = rate_per_core(n32_flp_jac_20, n32_time_jac_20)
    n32_flpRate_cheb_20 = rate_per_core(n32_flp_cheb_20, n32_time_cheb_20)

    ############### dim(A)=32^3 x 32^3 ####################################################
    #convergence number of cg iterations for cores = [1,2,4,8,16,32,48]
    n64_it_cg = 292 * np.ones(7)
    n64_it_jac_20 = np.array([61, 61, 60, 60, 61, 60, 60])
    n64_it_cheb_20 = 28 * np.ones(7)

    #Total computation time cg for cores = [1,2,4,8,16,32,48]
    n64_time_cg = np.array([1.906, 1.049, 0.5947, 0.3622, 0.2526, 0.2397, 0.2663])
    n64_time_jac_20 = np.array([6.914, 3.625, 1.924, 0.9741, 0.596, 0.4614, 0.5335])
    n64_time_cheb_20 = np.array([3.527, 1.852, 0.9948, 0.5454, 0.3112, 0.2712, 0.3197])

    #flops
    n64_flp_cg = flops("cg", nx = 64, ny = 64, nz = 64, cg_it = n64_it_cg)
    n64_flp_jac_20 = flops("pcg_jac", nx = 64, ny = 64, nz = 64, cg_it = n64_it_jac_20, prec_it = 20)
    n64_flp_cheb_20 = flops("pcg_cheb", nx = 64, ny = 64, nz = 64, cg_it = n64_it_cheb_20, prec_it = 20)


    #flopRate
    n64_flpRate_cg = rate_per_core(n64_flp_cg, n64_time_cg)
    n64_flpRate_jac_20 = rate_per_core(n64_flp_jac_20, n64_time_jac_20)
    n64_flpRate_cheb_20 = rate_per_core(n64_flp_cheb_20, n64_time_cheb_20)

    
    ############### dim(A)=256^3 x 256^3 ####################################################
    #convergence number of cg iterations for cores = [1,2,4,8,16,32,48]
    n256_it_cg = np.array([1231, 1228, 1227, 1227, 1227, 1227, 1227])
    n256_it_jac_20 = 268 * np.ones(7)
    n256_it_cheb_20 = 126 * np.ones(7)

    #Total computation time cg for cores = [1,2,4,8,16,32,48]
    n256_time_cg = np.array([536.3, 293.8, 176.7, 103.9, 73.73, 58.39, 51.03])
    n256_time_jac_20 = np.array([2036, 1074, 583.9, 303.5, 171.6, 107.5, 75.55])
    n256_time_cheb_20 = np.array([1117, 607.7, 356.5, 193, 124.7, 86.48, 56.68])

    #flops
    n256_flp_cg = flops("cg", nx = 256, ny = 256, nz = 256, cg_it = n256_it_cg)
    n256_flp_jac_20 = flops("pcg_jac", nx = 256, ny = 256, nz = 256, cg_it = n256_it_jac_20, prec_it = 20)
    n256_flp_cheb_20 = flops("pcg_cheb", nx = 256, ny = 256, nz = 256, cg_it = n256_it_cheb_20, prec_it = 20)

    #flopRate
    n256_flpRate_cg = rate_per_core(n256_flp_cg, n256_time_cg)
    n256_flpRate_jac_20 = rate_per_core(n256_flp_jac_20, n256_time_jac_20)
    n256_flpRate_cheb_20 = rate_per_core(n256_flp_cheb_20, n256_time_cheb_20)

    ############### dim(A)=512^3 x 512^3 ####################################################
    #convergence number of cg iterations for cores = [1,2,4,8,16,32,48]
    n512_it_cg = np.array([2781, 2774, 2767, 2765, 2764, 2541, 2538])
    n512_it_jac_20 = np.array([607, 607, 607, 585, 556, 556, 556])
    n512_it_cheb_20 = np.array([266, 266, 266, 266, 266, 266, 266]) 

    #Total computation time cg for cores = [1,2,4,8,16,32,48]
    n512_time_cg = np.array([9753, 5380, 3221, 1903, 1372, 907.3, 782.7])
    n512_time_jac_20 = np.array([36170, 19060, 10400, 5222, 2912, 1721, 1213])
    n512_time_cheb_20 = np.array([18770, 10110, 5980, 3230, 2229, 1355, 928.2]) 

    #flops
    n512_flp_cg = flops("cg", nx = 512, ny = 512, nz = 512, cg_it = n512_it_cg)
    n512_flp_jac_20 = flops("pcg_jac", nx = 512, ny = 512, nz = 512, cg_it = n512_it_jac_20, prec_it = 20)
    n512_flp_cheb_20 = flops("pcg_cheb", nx = 512, ny = 512, nz = 512, cg_it = n512_it_cheb_20, prec_it = 20)

    #flopRate
    n512_flpRate_cg = rate_per_core(n512_flp_cg, n512_time_cg)
    n512_flpRate_jac_20 = rate_per_core(n512_flp_jac_20, n512_time_jac_20)
    n512_flpRate_cheb_20 = rate_per_core(n512_flp_cheb_20, n512_time_cheb_20)

#########################more plots##############################################3
    #plot
    flpRate_cg = np.array([n32_flpRate_cg, n64_flpRate_cg, n128_flpRate_cg, n256_flpRate_cg, n512_flpRate_cg])
    flpRate_jac_20 = np.array([n32_flpRate_jac_20, n64_flpRate_jac_20, n128_flpRate_jac_20, n256_flpRate_jac_20, n512_flpRate_jac_20])
    flpRate_cheb_20 = np.array([n32_flpRate_cheb_20, n64_flpRate_cheb_20, n128_flpRate_cheb_20, n256_flpRate_cheb_20, n512_flpRate_cheb_20])
    plot_20it_flps(cores, flpRate_cg, flpRate_jac_20, flpRate_cheb_20)

    it_cg = np.array([n32_it_cg, n64_it_cg, n128_it_cg, n256_it_cg, n512_it_cg])
    it_jac_20 = np.array([n32_it_jac_20, n64_it_jac_20, n128_it_jac_20, n256_it_jac_20, n512_it_jac_20])
    it_cheb_20 = np.array([n32_it_cheb_20, n64_it_cheb_20, n128_it_cheb_20, n256_it_cheb_20, n512_it_cheb_20])
    it_cg = np.sum(it_cg, axis=1)//np.shape(it_cg)[1]
    it_jac_20 = np.sum(it_jac_20, axis=1)//np.shape(it_jac_20)[1]
    it_cheb_20 = np.sum(it_cheb_20, axis=1)//np.shape(it_cheb_20)[1]
    plot_size(size, it_cg, it_jac_20, it_cheb_20, "Matrix vs Convergence of (P)CG using 20 Jacobi or Chebyshev iterations", "Number of CG iterations needed for convergence")
 
    flp_cg = np.array([n32_flp_cg, n64_flp_cg, n128_flp_cg, n256_flp_cg, n512_flp_cg])
    flp_jac_20 = np.array([n32_flp_jac_20, n64_flp_jac_20, n128_flp_jac_20, n256_flp_jac_20, n512_flp_jac_20])
    flp_cheb_20 = np.array([n32_flp_cheb_20, n64_flp_cheb_20, n128_flp_cheb_20, n256_flp_cheb_20, n512_flp_cheb_20])

    flp_cg = np.sum(flp_cg, axis=1)/np.shape(flp_cg)[1]
    flp_jac_20 = np.sum(flp_jac_20, axis=1)/np.shape(flp_jac_20)[1]
    flp_cheb_20 = np.sum(flp_cheb_20, axis=1)/np.shape(flp_cheb_20)[1]
    
    plot_size(size, flp_cg, flp_jac_20, flp_cheb_20, "Matrix vs Gflops of (P)CG using 20 Jacobi or Chebyshev iterations", "Number of Gflops needed for convergence")
