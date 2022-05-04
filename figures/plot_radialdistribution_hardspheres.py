import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.markers as mmark
import seaborn as sns
from matplotlib import cm
import pandas as pd

plt.style.use(['science'])
plt.rcParams["lines.markersize"] = 4.0
plt.rcParams["axes.labelsize"] =  10.0
plt.rcParams["legend.fontsize"] =  8.0

widthcircle = 1.5

pts_per_inch = 72.27
# column_width_in_pts = 240.71
# text_width_in_inches = column_width_in_pts / pts_per_inch
text_width_in_pts = 504.46
text_width_in_inches = text_width_in_pts / pts_per_inch
aspect_ratio = 0.6
inverse_latex_scale = 1
csize = inverse_latex_scale * text_width_in_inches
fig_size = (1.0 * csize,aspect_ratio * csize)

plt.rcParams['figure.figsize'] = fig_size


#######################################################################
# figsize accepts only inches.
fig, ax = plt.subplots(2,4, sharex=True, sharey=False)
fig.subplots_adjust(left=0.13, right=0.98, top=0.85, bottom=0.08,
                    hspace=0.1, wspace=0.1)

df = pd.read_excel('../MCdata/MCdata-hardsphere-Barker1971.xls',sheet_name='radialdistributionfunction') 

ax[0,0].scatter(df['r'],df['rhob=0.2'],marker='o',edgecolors='C0',facecolors='none',linewidth=widthcircle,label='MC')

N = 256
rhob = 0.2
n1 = np.load('../densityfield-fmt-wbi-rhob0.2-N256-delta0.05.npy')
r1 = np.arange(N//2) * 0.05
ax[0,0].plot(r1,n1[N//2:,N//2,N//2]/rhob,'-',color='k',label='$256^3, 0.05\sigma$')

N = 128
rhob = 0.2
n1 = np.load('../densityfield-fmt-wbi-rhob0.2-N128-delta0.1.npy')
r1 = np.arange(N//2) * 0.1
ax[0,0].plot(r1,n1[N//2:,N//2,N//2]/rhob,'--',color='C3',label='$128^3, 0.1\sigma$')

N = 64
rhob = 0.2
n1 = np.load('../densityfield-fmt-wbi-rhob0.2-N64-delta0.2.npy')
r1 = np.arange(N//2) * 0.2
ax[0,0].plot(r1,n1[N//2:,N//2,N//2]/rhob,':',color='C2',label='$64^3,0.2\sigma$')


# ax[0,0].set_yscale('log')
ax[0,0].set_ylabel(r'$g(r)$')
ax[0,0].set_xlim(1.0,3)
ax[0,0].set_ylim(0.5,2.5)
ax[0,0].legend(loc='upper right',ncol=1)
ax[0,0].tick_params(labelbottom=False)  
ax[0,0].text(1.5,0.65,r'$\rho_b \sigma^3 = 0.2$')


ax[0,1].scatter(df['r'],df['rhob=0.3'],marker='o',edgecolors='C0',facecolors='none',linewidth=widthcircle,label='MC')

N = 256
rhob = 0.3
n1 = np.load('../densityfield-fmt-wbi-rhob0.3-N256-delta0.05.npy')
r1 = np.arange(N//2) * 0.05
ax[0,1].plot(r1,n1[N//2:,N//2,N//2]/rhob,'-',color='k')

N = 128
rhob = 0.3
n1 = np.load('../densityfield-fmt-wbi-rhob0.3-N128-delta0.1.npy')
r1 = np.arange(N//2) * 0.1
ax[0,1].plot(r1,n1[N//2:,N//2,N//2]/rhob,'--',color='C3',label='$\Delta=0.1\sigma$')

N = 64
rhob = 0.3
n1 = np.load('../densityfield-fmt-wbi-rhob0.3-N64-delta0.2.npy')
r1 = np.arange(N//2) * 0.2
ax[0,1].plot(r1,n1[N//2:,N//2,N//2]/rhob,':',color='C2')

ax[0,1].set_ylim(0.5,2.5)
ax[0,1].tick_params(labelbottom=False,labelleft=False)  
ax[0,1].text(1.5,0.65,r'$\rho_b \sigma^3 = 0.3$')

###

ax[0,2].scatter(df['r'],df['rhob=0.4'],marker='o',edgecolors='C0',facecolors='none',linewidth=widthcircle,label='MC')

N = 256
rhob = 0.4
n1 = np.load('../densityfield-fmt-wbi-rhob0.4-N256-delta0.05.npy')
r1 = np.arange(N//2) * 0.05
ax[0,2].plot(r1,n1[N//2:,N//2,N//2]/rhob,'-',color='k')

N = 128
rhob = 0.4
n1 = np.load('../densityfield-fmt-wbi-rhob0.4-N128-delta0.1.npy')
r1 = np.arange(N//2) * 0.1
ax[0,2].plot(r1,n1[N//2:,N//2,N//2]/rhob,'--',color='C3',label='$\Delta=0.1\sigma$')

N = 64
rhob = 0.4
n1 = np.load('../densityfield-fmt-wbi-rhob0.4-N64-delta0.2.npy')
r1 = np.arange(N//2) * 0.2
ax[0,2].plot(r1,n1[N//2:,N//2,N//2]/rhob,':',color='C2')

ax[0,2].set_ylim(0.5,2.5)
ax[0,2].tick_params(labelbottom=False,labelleft=False)  
ax[0,2].text(1.5,0.65,r'$\rho_b \sigma^3 = 0.4$')

###

ax[0,3].scatter(df['r'],df['rhob=0.5'],marker='o',edgecolors='C0',facecolors='none',linewidth=widthcircle,label='MC')

N = 256
rhob = 0.5
n1 = np.load('../densityfield-fmt-wbi-rhob0.5-N256-delta0.05.npy')
r1 = np.arange(N//2) * 0.05
ax[0,3].plot(r1,n1[N//2:,N//2,N//2]/rhob,'-',color='k')

N = 128
rhob = 0.5
n1 = np.load('../densityfield-fmt-wbi-rhob0.5-N128-delta0.1.npy')
r1 = np.arange(N//2) * 0.1
ax[0,3].plot(r1,n1[N//2:,N//2,N//2]/rhob,'--',color='C3',label='$\Delta=0.1\sigma$')

N = 64
rhob = 0.5
n1 = np.load('../densityfield-fmt-wbi-rhob0.5-N64-delta0.2.npy')
r1 = np.arange(N//2) * 0.2
ax[0,3].plot(r1,n1[N//2:,N//2,N//2]/rhob,':',color='C2')

ax[0,3].set_ylim(0.5,2.5)
ax[0,3].tick_params(labelbottom=False,labelleft=False)  
ax[0,3].text(1.5,0.65,r'$\rho_b \sigma^3 = 0.5$')

###

ax[1,0].scatter(df['r'],df['rhob=0.6'],marker='o',edgecolors='C0',facecolors='none',linewidth=widthcircle,label='MC')

N = 256
rhob = 0.6
n1 = np.load('../densityfield-fmt-wbi-rhob0.6-N256-delta0.05.npy')
r1 = np.arange(N//2) * 0.05
ax[1,0].plot(r1,n1[N//2:,N//2,N//2]/rhob,'-',color='k')

N = 128
rhob = 0.6
n1 = np.load('../densityfield-fmt-wbi-rhob0.6-N128-delta0.1.npy')
r1 = np.arange(N//2) * 0.1
ax[1,0].plot(r1,n1[N//2:,N//2,N//2]/rhob,'--',color='C3',label='$\Delta=0.1\sigma$')

N = 64
rhob = 0.6
n1 = np.load('../densityfield-fmt-wbi-rhob0.6-N64-delta0.2.npy')
r1 = np.arange(N//2) * 0.2
ax[1,0].plot(r1,n1[N//2:,N//2,N//2]/rhob,':',color='C2')


# ax[0,0].set_yscale('log')
ax[1,0].set_ylabel(r'$g(r)$')
ax[1,0].set_xlim(1.0,3)
ax[1,0].set_ylim(0.5,6)
ax[1,0].set_xlabel(r'$r/\sigma$')
ax[1,0].text(1.5,5,r'$\rho_b \sigma^3 = 0.6$')


###

ax[1,1].scatter(df['r'],df['rhob=0.7'],marker='o',edgecolors='C0',facecolors='none',linewidth=widthcircle,label='MC')

N = 256
rhob = 0.7
n1 = np.load('../densityfield-fmt-wbi-rhob0.7-N256-delta0.05.npy')
r1 = np.arange(N//2) * 0.05
ax[1,1].plot(r1,n1[N//2:,N//2,N//2]/rhob,'-',color='k')

N = 128
rhob = 0.7
n1 = np.load('../densityfield-fmt-wbi-rhob0.7-N128-delta0.1.npy')
r1 = np.arange(N//2) * 0.1
ax[1,1].plot(r1,n1[N//2:,N//2,N//2]/rhob,'--',color='C3',label='$\Delta=0.1\sigma$')

N = 64
rhob = 0.7
n1 = np.load('../densityfield-fmt-wbi-rhob0.7-N64-delta0.2.npy')
r1 = np.arange(N//2) * 0.2
ax[1,1].plot(r1,n1[N//2:,N//2,N//2]/rhob,':',color='C2')

ax[1,1].set_ylim(0.5,6)
ax[1,1].tick_params(labelleft=False) 
ax[1,1].set_xlabel(r'$r/\sigma$')
ax[1,1].text(1.5,5,r'$\rho_b \sigma^3 = 0.7$')


###

ax[1,2].scatter(df['r'],df['rhob=0.8'],marker='o',edgecolors='C0',facecolors='none',linewidth=widthcircle,label='MC')

N = 256
rhob = 0.8
n1 = np.load('../densityfield-fmt-wbi-rhob0.8-N256-delta0.05.npy')
r1 = np.arange(N//2) * 0.05
ax[1,2].plot(r1,n1[N//2:,N//2,N//2]/rhob,'-',color='k')

N = 128
rhob = 0.8
n1 = np.load('../densityfield-fmt-wbi-rhob0.8-N128-delta0.1.npy')
r1 = np.arange(N//2) * 0.1
ax[1,2].plot(r1,n1[N//2:,N//2,N//2]/rhob,'--',color='C3',label='$\Delta=0.1$')

N = 64
rhob = 0.8
n1 = np.load('../densityfield-fmt-wbi-rhob0.8-N64-delta0.2.npy')
r1 = np.arange(N//2) * 0.2
ax[1,2].plot(r1,n1[N//2:,N//2,N//2]/rhob,':',color='C2')

ax[1,2].set_ylim(0.5,6)
ax[1,2].tick_params(labelleft=False) 
ax[1,2].set_xlabel(r'$r/\sigma$')
ax[1,2].text(1.5,5,r'$\rho_b \sigma^3 = 0.8$')


###

ax[1,3].scatter(df['r'],df['rhob=0.9'],marker='o',edgecolors='C0',facecolors='none',linewidth=widthcircle,label='MC')


N = 256
n1 = np.load('../densityfield-fmt-wbi-rhob0.9-N256-delta0.05.npy')
r1 = np.arange(N//2) * 0.05
ax[1,3].plot(r1,n1[N//2:,N//2,N//2]/rhob,'-',color='k',label='$\Delta=0.05$')

N = 128
rhob = 0.9
n1 = np.load('../densityfield-fmt-wbi-rhob0.9-N128-delta0.1.npy')
r1 = np.arange(N//2) * 0.1
ax[1,3].plot(r1,n1[N//2:,N//2,N//2]/rhob,'--',color='C3',label='$\Delta=0.1$')

# N = 86
# rhob = 0.9
# n1 = np.load('../densityfield-fmt-wbi-rhob0.9-N86-delta0.15.npy')
# r1 = np.arange(N//2) * 0.15
# ax[1,3].plot(r1,n1[N//2:,N//2,N//2]/rhob,'-.',color='C1')

N = 64
rhob = 0.9
n1 = np.load('../densityfield-fmt-wbi-rhob0.9-N64-delta0.2.npy')
r1 = np.arange(N//2) * 0.2
ax[1,3].plot(r1,n1[N//2:,N//2,N//2]/rhob,':',color='C2')

ax[1,3].set_ylim(0.5,6)
ax[1,3].tick_params(labelleft=False) 
ax[1,3].set_xlabel(r'$r/\sigma$')
ax[1,3].text(1.5,5,r'$\rho_b \sigma^3 = 0.9$')

fig.savefig('radialdistribution_hardspheres.pdf')
fig.savefig('radialdistribution_hardspheres.png', bbox_inches='tight')
plt.close()