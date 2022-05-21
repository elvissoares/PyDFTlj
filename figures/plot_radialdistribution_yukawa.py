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

widthcircle = 1.0

# pts_per_inch = 72.27
# column_width_in_pts = 240.71
# text_width_in_inches = column_width_in_pts / pts_per_inch
# # text_width_in_pts = 504.46
# # text_width_in_inches = text_width_in_pts / pts_per_inch
# aspect_ratio = 0.6
# inverse_latex_scale = 1
# csize = inverse_latex_scale * text_width_in_inches
# fig_size = (1.0 * csize,aspect_ratio * csize)

# plt.rcParams['figure.figsize'] = fig_size


#######################################################################
# figsize accepts only inches.
fig, ax = plt.subplots(1,1)
# fig.subplots_adjust(left=0.13, right=0.98, top=0.85, bottom=0.08,
#                     hspace=0.1, wspace=0.1)

df = pd.read_excel('../MCdata/MCdata-radialdistribution-yukawa-Shukla2000.xls',sheet_name='radialdistributionfunction-l=1.8') # Shukla2000

ax.scatter(df['r.1'],df['rhob=0.8-T=2.0'],marker='o',edgecolors='C0',facecolors='none',linewidth=widthcircle,label='MC')

rhob = 0.8

N = 256
delta = 0.05
z = np.arange(0,N*delta,delta)-N*delta/2
n = np.load('../results/densityfield-yukawa-fmsa-rhob0.8-N256-delta0.05.npy')
ax.plot(z,n[N//2,N//2,:]/rhob,'-',color='k',label='$256^3, 0.05\sigma$')
print(4.202,n.max()/rhob)

N = 128
delta = 0.1
z = np.arange(0,N*delta,delta)-N*delta/2
n = np.load('../results/densityfield-yukawa-fmsa-rhob0.8-N128-delta0.1.npy')
ax.plot(z,n[N//2,N//2,:]/rhob,'-.',color='C3',label='$128^3,0.1\sigma$')
print(4.202,n.max()/rhob)

N = 64
delta = 0.2
z = np.arange(0,N*delta,delta)-N*delta/2
n = np.load('../results/densityfield-yukawa-fmsa-rhob0.8-N64-delta0.2.npy')
ax.plot(z,n[N//2,N//2,:]/rhob,'--',color='C2',label='$64^3,0.2\sigma$')
print(4.202,n.max()/rhob)

N = 32
delta = 0.4
z = np.arange(0,N*delta,delta)-N*delta/2
n = np.load('../results/densityfield-yukawa-fmsa-rhob0.8-N32-delta0.4.npy')
ax.plot(z,n[N//2,N//2,:]/rhob,':',color='grey',label='$32^3,0.4\sigma$')
print(4.202,n.max()/rhob)
# ax.plot(rhob,gsigma,':',color='grey',label='$32^3,0.4\sigma$')

# ax[0,0].set_yscale('log')
ax.set_ylabel(r'$g(r)$')
ax.set_xlabel(r'$r/\sigma$')
ax.text(1.5,2.5,r'$k_B T/\epsilon = 2.0$')
ax.text(1.55,2.0,r'$\rho_b \sigma^3 = 0.8$')
ax.set_xlim(1.0,3.0)
ax.set_ylim(0.5,5)
ax.legend(loc='upper right',ncol=1)

fig.savefig('radialdistribution_yukawa-rhob=0.8-T=2.0-l=1.8.pdf')
fig.savefig('radialdistribution_yukawa-rhob=0.8-T=2.0-l=1.8.png', bbox_inches='tight')
plt.close()

#######################################################################
# figsize accepts only inches.
fig, ax = plt.subplots(1,1)
# fig.subplots_adjust(left=0.13, right=0.98, top=0.85, bottom=0.08,
#                     hspace=0.1, wspace=0.1)

df = pd.read_excel('../MCdata/MCdata-radialdistribution-yukawa-Shukla2000.xls',sheet_name='radialdistributionfunction-l=1.8') # Shukla2000

ax.scatter(df['r'],df['rhob=0.3-T=2.0'],marker='o',edgecolors='C0',facecolors='none',linewidth=widthcircle,label='MC')

rhob = 0.3

N = 256
delta = 0.05
z = np.arange(0,N*delta,delta)-N*delta/2
n = np.load('../results/densityfield-yukawa-fmsa-rhob0.3-N256-delta0.05.npy')
ax.plot(z,n[N//2,N//2,:]/rhob,'-',color='k',label='$256^3, 0.05\sigma$')
print(4.202,n.max()/rhob)

N = 128
delta = 0.1
z = np.arange(0,N*delta,delta)-N*delta/2
n = np.load('../results/densityfield-yukawa-fmsa-rhob0.3-N128-delta0.1.npy')
ax.plot(z,n[N//2,N//2,:]/rhob,'-.',color='C3',label='$128^3,0.1\sigma$')
print(4.202,n.max()/rhob)

N = 64
delta = 0.2
z = np.arange(0,N*delta,delta)-N*delta/2
n = np.load('../results/densityfield-yukawa-fmsa-rhob0.3-N64-delta0.2.npy')
ax.plot(z,n[N//2,N//2,:]/rhob,'--',color='C2',label='$64^3,0.2\sigma$')
print(4.202,n.max()/rhob)

N = 32
delta = 0.4
z = np.arange(0,N*delta,delta)-N*delta/2
n = np.load('../results/densityfield-yukawa-fmsa-rhob0.3-N32-delta0.4.npy')
ax.plot(z,n[N//2,N//2,:]/rhob,':',color='grey',label='$32^3,0.4\sigma$')
print(4.202,n.max()/rhob)
# ax.plot(rhob,gsigma,':',color='grey',label='$32^3,0.4\sigma$')

# ax[0,0].set_yscale('log')
ax.set_ylabel(r'$g(r)$')
ax.set_xlabel(r'$r/\sigma$')
ax.text(1.5,1.5,r'$k_B T/\epsilon = 2.0$')
ax.text(1.55,1.3,r'$\rho_b \sigma^3 = 0.3$')
ax.set_xlim(1.0,3.0)
ax.set_ylim(0.5,2.5)
ax.legend(loc='upper right',ncol=1)

fig.savefig('radialdistribution_yukawa-rhob=0.3-T=2.0-l=1.8.pdf')
fig.savefig('radialdistribution_yukawa-rhob=0.3-T=2.0-l=1.8.png', bbox_inches='tight')
plt.close()