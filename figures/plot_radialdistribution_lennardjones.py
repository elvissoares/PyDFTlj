import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.markers as mmark
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

df = pd.read_excel('../MCdata/MCdata-radialdistribution-lennardjones-Reatto1986.xls',sheet_name='rhob=0.84-kT=0.75') # Shukla2000

ax.scatter(df['r/sigma'],df['g(r)'],marker='o',edgecolors='C0',facecolors='none',linewidth=widthcircle,label='MD')

[x,n] = np.load('../results/radialdistribution-lennardjones-rhob=0.84-T=0.75-Elvisparameters.npy')
ax.plot(x,n,linestyle=(0, (1, 1)),color='r',label='1D - $0.01\sigma$')

rhob = 0.84
kT = 0.75

sigma = 1.0
d = sigma*(1+0.2977*kT)/(1+0.33163*kT+0.0010477*kT**2)

rhob *= d**3

# N = 256
# delta = 0.05*d
# z = np.arange(0,N*delta,delta)-N*delta/2
# n = np.load('../results/densityfield-lj-fmsa-rhostar=0.84-Tstar=0.71-N256-delta0.05.npy')
# ax.plot(z,n[N//2,N//2,:]/rhob,'-',color='k',label='$256^3, 0.05\sigma$')

N = 128
delta = 0.1*d
z = np.arange(0,N*delta,delta)-N*delta/2
n = np.load('../results/densityfield-lj-fmsa-rhostar=0.84-Tstar=0.75-N128-delta0.1.npy')
ax.plot(z,n[N//2,N//2,:]/rhob,'-.',color='C3',label='$128^3,0.1\sigma$')

# N = 64
# delta = 0.2*d
# z = np.arange(0,N*delta,delta)-N*delta/2
# n = np.load('../results/densityfield-lj-fmsa-rhostar=0.84-Tstar=0.71-N64-delta0.2.npy')
# ax.plot(z,n[N//2,N//2,:]/rhob,'--',color='C2',label='$64^3,0.2\sigma$')


# N = 32
# delta = 0.4*d
# z = np.arange(0,N*delta,delta)-N*delta/2
# n = np.load('../results/densityfield-lj-fmsa-rhostar=0.84-Tstar=0.71-N32-delta0.4.npy')
# ax.plot(z,n[N//2,N//2,:]/rhob,':',color='grey',label='$32^3,0.4\sigma$')


# ax[0,0].set_yscale('log')
ax.set_ylabel(r'$g(r)$')
ax.set_xlabel(r'$r/\sigma$')
ax.text(1.7,2.0,r'$k_B T/\epsilon = 0.75$')
ax.text(1.75,1.7,r'$\rho_b \sigma^3 = 0.84$')
ax.set_xlim(0.0,5.0)
ax.set_ylim(0,3.5)
ax.legend(loc='upper right',ncol=1)

fig.savefig('radialdistribution_lennardjones-rhob=0.84-T=0.75.pdf')
fig.savefig('radialdistribution_lennardjones-rhob=0.84-T=0.75.png', bbox_inches='tight')
plt.close()


#######################################################################
# figsize accepts only inches.
fig, ax = plt.subplots(1,1)
# fig.subplots_adjust(left=0.13, right=0.98, top=0.85, bottom=0.08,
#                     hspace=0.1, wspace=0.1)

df = pd.read_excel('../MCdata/MCdata-radialdistribution-lennardjones-Verlet1968.xls',sheet_name='Argon') # Shukla2000

ax.scatter(df['r']/3.405,df['KT=0.71-rhob=0.84'],marker='o',edgecolors='C0',facecolors='none',linewidth=widthcircle,label=r'${}^{36}$Ar @ 85 K')

rhob = 0.84
kT = 0.71

sigma = 1.0
d = sigma*(1+0.2977*kT)/(1+0.33163*kT+0.0010477*kT**2)

rhob *= d**3

N = 256
delta = 0.05*d
z = np.arange(0,N*delta,delta)-N*delta/2
n = np.load('../results/densityfield-lj-fmsa-rhostar=0.84-Tstar=0.71-N256-delta0.05.npy')
ax.plot(z,n[N//2,N//2,:]/rhob,'-',color='k',label='$256^3, 0.05\sigma$')

N = 128
delta = 0.1*d
z = np.arange(0,N*delta,delta)-N*delta/2
n = np.load('../results/densityfield-lj-fmsa-rhostar=0.84-Tstar=0.71-N128-delta0.1.npy')
ax.plot(z,n[N//2,N//2,:]/rhob,'-.',color='C3',label='$128^3,0.1\sigma$')

# N = 128
# delta = 0.1
# z = np.arange(0,N*delta,delta)-N*delta/2
# n = np.load('../results/densityfield-hs-wbi-rhostar=0.84-Tstar=0.71-N128-delta0.1.npy')
# ax.plot(z,n[N//2,N//2,:]/rhob,'-.',color='grey')

N = 64
delta = 0.2*d
z = np.arange(0,N*delta,delta)-N*delta/2
n = np.load('../results/densityfield-lj-fmsa-rhostar=0.84-Tstar=0.71-N64-delta0.2.npy')
ax.plot(z,n[N//2,N//2,:]/rhob,'--',color='C2',label='$64^3,0.2\sigma$')


N = 32
delta = 0.4*d
z = np.arange(0,N*delta,delta)-N*delta/2
n = np.load('../results/densityfield-lj-fmsa-rhostar=0.84-Tstar=0.71-N32-delta0.4.npy')
ax.plot(z,n[N//2,N//2,:]/rhob,':',color='grey',label='$32^3,0.4\sigma$')


# ax[0,0].set_yscale('log')
ax.set_ylabel(r'$g(r)$')
ax.set_xlabel(r'$r/\sigma$')
ax.text(1.7,2.0,r'$k_B T/\epsilon = 0.71$')
ax.text(1.75,1.7,r'$\rho_b \sigma^3 = 0.84$')
ax.set_xlim(0.5,5.0)
ax.set_ylim(-0.2,3.2)
ax.legend(loc='upper right',ncol=1)

fig.savefig('radialdistribution_lennardjones-rhob=0.84-T=0.71.pdf')
fig.savefig('radialdistribution_lennardjones-rhob=0.84-T=0.71.png', bbox_inches='tight')
plt.close()

#######################################################################
# figsize accepts only inches.
fig, ax = plt.subplots(1,1)
# fig.subplots_adjust(left=0.13, right=0.98, top=0.85, bottom=0.08,
#                     hspace=0.1, wspace=0.1)

df = pd.read_excel('../MCdata/MCdata-radialdistribution-lennardjones-Verlet1968.xls',sheet_name='rhob=0.650') 

ax.scatter(df['r'],df['KT=3.669'],marker='o',edgecolors='C0',facecolors='none',linewidth=widthcircle,label='MD')

rhob = 0.65
kT = 3.669

sigma = 1.0
d = sigma*(1+0.2977*kT)/(1+0.33163*kT+0.0010477*kT**2)

rhob *= d**3

N = 256
delta = 0.05*d
z = np.arange(0,N*delta,delta)-N*delta/2
n = np.load('../results/densityfield-lj-fmsa-rhostar=0.65-Tstar=3.669-N256-delta0.05.npy')
ax.plot(z,n[N//2,N//2,:]/rhob,'-',color='k',label='$256^3, 0.05\sigma$')

N = 128
delta = 0.1*d
z = np.arange(0,N*delta,delta)-N*delta/2
n = np.load('../results/densityfield-lj-fmsa-rhostar=0.65-Tstar=3.669-N128-delta0.1.npy')
ax.plot(z,n[N//2,N//2,:]/rhob,'-.',color='C3',label='$128^3,0.1\sigma$')

# N = 128
# delta = 0.1
# z = np.arange(0,N*delta,delta)-N*delta/2
# n = np.load('../results/densityfield-hs-wbi-rhostar=0.84-Tstar=0.71-N128-delta0.1.npy')
# ax.plot(z,n[N//2,N//2,:]/rhob,'-.',color='grey')

N = 64
delta = 0.2*d
z = np.arange(0,N*delta,delta)-N*delta/2
n = np.load('../results/densityfield-lj-fmsa-rhostar=0.65-Tstar=3.669-N64-delta0.2.npy')
ax.plot(z,n[N//2,N//2,:]/rhob,'--',color='C2',label='$64^3,0.2\sigma$')


# N = 32
# delta = 0.4
# z = np.arange(0,N*delta,delta)-N*delta/2
# n = np.load('../results/densityfield-lj-fmsa-rhostar=0.84-Tstar=0.71-N32-delta0.4.npy')
# ax.plot(z,n[N//2,N//2,:]/rhob,':',color='grey',label='$32^3,0.4\sigma$')


# ax[0,0].set_yscale('log')
ax.set_ylabel(r'$g(r)$')
ax.set_xlabel(r'$r/\sigma$')
ax.text(1.7,1.7,r'$k_B T/\epsilon = 3.669$')
ax.text(1.75,1.5,r'$\rho_b \sigma^3 = 0.65$')
ax.set_xlim(0.5,5.0)
ax.set_ylim(-0.2,2.2)
ax.legend(loc='upper right',ncol=1)

fig.savefig('radialdistribution_lennardjones-rhob=0.65-T=0.9.pdf')
fig.savefig('radialdistribution_lennardjones-rhob=0.65-T=0.9.png', bbox_inches='tight')
plt.close()
