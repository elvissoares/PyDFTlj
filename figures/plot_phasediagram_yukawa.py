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

widthcircle = 1.2

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

df = pd.read_excel('../MCdata/MCdata-yukawa-phasediagram.xls',sheet_name='l=1.8') 
ax.scatter(df['rho1'],df['T'],marker='o',edgecolors='C0',facecolors='none',linewidth=widthcircle,label='MC')
ax.scatter(df['rho2'],df['T'],marker='o',edgecolors='C0',facecolors='none',linewidth=widthcircle)

df = pd.read_excel('../MCdata/MCdata-yukawa-phasediagram.xls',sheet_name='l=3.0') 
ax.scatter(df['rho1'],df['T'],marker='o',edgecolors='C1',facecolors='none',linewidth=widthcircle)
ax.scatter(df['rho2'],df['T'],marker='o',edgecolors='C1',facecolors='none',linewidth=widthcircle)

df = pd.read_excel('../MCdata/MCdata-yukawa-phasediagram.xls',sheet_name='l=4.0') 
ax.scatter(df['rho1'],df['T'],marker='o',edgecolors='C2',facecolors='none',linewidth=widthcircle)
ax.scatter(df['rho2'],df['T'],marker='o',edgecolors='C2',facecolors='none',linewidth=widthcircle)

df = pd.read_excel('../MCdata/MCdata-yukawa-phasediagram.xls',sheet_name='l=8.0') 
ax.scatter(df['rho1'],df['T'],marker='o',edgecolors='C3',facecolors='none',linewidth=widthcircle)
ax.scatter(df['rho2'],df['T'],marker='o',edgecolors='C3',facecolors='none',linewidth=widthcircle)

data = np.loadtxt('../results/phasediagram_yukawa_l1.8_FMSA.dat')
[kT,rhov,rhol,mu,omega] = data.T
ax.plot(np.hstack((rhov,rhol[::-1])),np.hstack((kT,kT[::-1])),linestyle='-',color='k',label='FMSA')

data = np.loadtxt('../results/phasediagram_yukawa_l3.0_FMSA.dat')
[kT,rhov,rhol,mu,omega] = data.T
ax.plot(np.hstack((rhov,rhol[::-1])),np.hstack((kT,kT[::-1])),linestyle='-',color='k')

data = np.loadtxt('../results/phasediagram_yukawa_l4.0_FMSA.dat')
[kT,rhov,rhol,mu,omega] = data.T
ax.plot(np.hstack((rhov,rhol[::-1])),np.hstack((kT,kT[::-1])),linestyle='-',color='k')

# ax.set_yscale('log')
ax.set_ylabel(r'$k_B T /\epsilon$')
ax.set_xlabel(r'$\rho_b \sigma^3$')
ax.set_xlim(0.0,0.9)
ax.set_ylim(0.2,1.4)
ax.text(0.4,1.25,r'$\lambda \sigma = 1.8$',fontsize=10)
ax.text(0.4,0.8,r'$\lambda \sigma = 3.0$',fontsize=10)
ax.text(0.4,0.65,r'$\lambda \sigma = 4.0$',fontsize=10)
ax.legend(loc='upper right',ncol=1)

fig.savefig('phasediagram_yukawa.pdf')
fig.savefig('phasediagram_yukawa.png', bbox_inches='tight')
plt.close()

#######################################################################
# figsize accepts only inches.
fig, ax = plt.subplots(1,1)
# fig.subplots_adjust(left=0.13, right=0.98, top=0.85, bottom=0.08,
#                     hspace=0.1, wspace=0.1)

df = pd.read_excel('../MCdata/MCdata-yukawa-phasediagram.xls',sheet_name='l=1.8') 
ax.scatter(1.0/df['T'],df['P'],marker='o',edgecolors='C0',facecolors='none',linewidth=widthcircle,label='MC')

df = pd.read_excel('../MCdata/MCdata-yukawa-phasediagram.xls',sheet_name='l=3.0') 
ax.scatter(1.0/df['T'],df['P'],marker='o',edgecolors='C1',facecolors='none',linewidth=widthcircle)

df = pd.read_excel('../MCdata/MCdata-yukawa-phasediagram.xls',sheet_name='l=4.0') 
ax.scatter(1.0/df['T'],df['P'],marker='o',edgecolors='C2',facecolors='none',linewidth=widthcircle)

df = pd.read_excel('../MCdata/MCdata-yukawa-phasediagram.xls',sheet_name='l=8.0') 
ax.scatter(1.0/df['T'],df['P'],marker='o',edgecolors='C3',facecolors='none',linewidth=widthcircle)

data = np.loadtxt('../results/phasediagram_yukawa_l1.8_FMSA.dat')
[kT,rhov,rhol,mu,omega] = data.T
ax.plot(1.0/kT,-omega,linestyle='-',color='k',label='FMSA')

data = np.loadtxt('../results/phasediagram_yukawa_l3.0_FMSA.dat')
[kT,rhov,rhol,mu,omega] = data.T
ax.plot(1.0/kT,-omega,linestyle='-',color='k')

data = np.loadtxt('../results/phasediagram_yukawa_l4.0_FMSA.dat')
[kT,rhov,rhol,mu,omega] = data.T
ax.plot(1.0/kT,-omega,linestyle='-',color='k')

# data = np.loadtxt('../results/phasediagram_yukawa_l8.0_FMSA.dat')
# [kT,rhov,rhol,mu,omega] = data.T
# ax.plot(1.0/kT,-omega,linestyle='-',color='k')


ax.set_yscale('log')
ax.set_xlabel(r'$\epsilon/k_B T$')
ax.set_ylabel(r'$p \sigma^3/\epsilon$')
ax.set_xlim(0.5,3)
ax.set_ylim(5e-3,0.2)
ax.legend(loc='upper right',ncol=1)

fig.savefig('pressure_yukawa.pdf')
fig.savefig('pressure_yukawa.png', bbox_inches='tight')
plt.close()