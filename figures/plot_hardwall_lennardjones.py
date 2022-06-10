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

MCdata = np.loadtxt('../MCdata/lj-hardwall-rhob0.82-T1.35.dat')
xMC,rhoMC = MCdata[:,0], MCdata[:,1]
ax.scatter(xMC+0.5,rhoMC,label='MC')

[x,n] = np.load('../results/profiles-lennardjones-hardwall-rhob=0.82-T=1.35.npy')
ax.plot(x,n,'--',color='grey',label='Tang2004')

[x,n] = np.load('../results/profiles-lennardjones-hardwall-rhob=0.82-T=1.35-Elvisparameters.npy')
ax.plot(x,n,'-',color='k',label='this work')

# ax[0,0].set_yscale('log')
ax.set_ylabel(r'$\rho(z) \sigma^3$')
ax.set_xlabel(r'$z/\sigma$')
ax.text(1.7,2.0,r'$k_B T/\epsilon = 1.35$')
ax.text(1.75,1.7,r'$\rho_b \sigma^3 = 0.82$')
ax.set_xlim(0.5,5.0)
ax.set_ylim(0,3)
ax.legend(loc='upper right',ncol=1)

fig.savefig('hardwall_lennardjones-rhob=0.82-T=1.35.pdf')
fig.savefig('hardwall_lennardjones-rhob=0.82-T=1.35.png', bbox_inches='tight')
plt.close()


#######################################################################
# figsize accepts only inches.
fig, ax = plt.subplots(1,1)
# fig.subplots_adjust(left=0.13, right=0.98, top=0.85, bottom=0.08,
#                     hspace=0.1, wspace=0.1)

MCdata = np.loadtxt('../MCdata/lj-hardwall-rhob0.5-T1.35.dat')
xMC,rhoMC = MCdata[:,0], MCdata[:,1]
ax.scatter(xMC+0.5,rhoMC,label='MC')

[x,n] = np.load('../results/profiles-lennardjones-hardwall-rhob=0.5-T=1.35.npy')
ax.plot(x,n,'--',color='grey',label='Tang2004')

[x,n] = np.load('../results/profiles-lennardjones-hardwall-rhob=0.5-T=1.35-Elvisparameters.npy')
ax.plot(x,n,'-',color='k',label='this work')

# ax[0,0].set_yscale('log')
ax.set_ylabel(r'$\rho(z) \sigma^3$')
ax.set_xlabel(r'$z/\sigma$')
ax.text(1.0,0.6,r'$k_B T/\epsilon = 1.35$')
ax.text(1.0,0.5,r'$\rho_b \sigma^3 = 0.5$')
ax.set_xlim(0.5,5.0)
ax.set_ylim(0,1.0)
ax.legend(loc='upper right',ncol=1)

fig.savefig('hardwall_lennardjones-rhob=0.5-T=1.35.pdf')
fig.savefig('hardwall_lennardjones-rhob=0.5-T=1.35.png', bbox_inches='tight')
plt.close()

#######################################################################
# figsize accepts only inches.
fig, ax = plt.subplots(1,1)
# fig.subplots_adjust(left=0.13, right=0.98, top=0.85, bottom=0.08,
#                     hspace=0.1, wspace=0.1)

MCdata = np.loadtxt('../MCdata/lj-hardwall-rhob0.65-T1.35.dat')
xMC,rhoMC = MCdata[:,0], MCdata[:,1]
ax.scatter(xMC+0.5,rhoMC,label='MC')

[x,n] = np.load('../results/profiles-lennardjones-hardwall-rhob=0.65-T=1.35.npy')
ax.plot(x,n,'--',color='grey',label='Tang2004')

[x,n] = np.load('../results/profiles-lennardjones-hardwall-rhob=0.65-T=1.35-Elvisparameters.npy')
ax.plot(x,n,'-',color='k',label='this work')

# ax[0,0].set_yscale('log')
ax.set_ylabel(r'$\rho(z) \sigma^3$')
ax.set_xlabel(r'$z/\sigma$')
ax.text(2.0,0.1,r'$k_B T/\epsilon = 1.35$')
ax.text(2.0,0.15,r'$\rho_b \sigma^3 = 0.65$')
ax.set_xlim(0.5,5.0)
ax.set_ylim(0,1.5)
ax.legend(loc='upper right',ncol=1)

fig.savefig('hardwall_lennardjones-rhob=0.65-T=1.35.pdf')
fig.savefig('hardwall_lennardjones-rhob=0.65-T=1.35.png', bbox_inches='tight')
plt.close()

#######################################################################
# figsize accepts only inches.
fig, ax = plt.subplots(1,1)
# fig.subplots_adjust(left=0.13, right=0.98, top=0.85, bottom=0.08,
#                     hspace=0.1, wspace=0.1)

MCdata = np.loadtxt('../MCdata/lj-slitpore-steele-T1.2-rhob0.5925-H1.8-GEMC.dat')
xMC,rhoMC = MCdata[:,0], MCdata[:,1]
ax.scatter(xMC,rhoMC,label='MC')

[x,n] = np.load('../results/profiles-lennardjones-slitpore-rhob=0.5925-T=1.2-H=1.8.npy')
ax.plot(x,n,'--',color='grey',label='Tang2004')

[x,n] = np.load('../results/profiles-lennardjones-slitpore-rhob=0.5925-T=1.2-H=1.8-Elvisparameters.npy')
ax.plot(x,n,'-',color='k',label='this work')

# ax[0,0].set_yscale('log')
ax.set_ylabel(r'$\rho(z) \sigma^3$')
ax.set_xlabel(r'$z/\sigma$')
ax.text(1.3,2.5,r'$H/\sigma= 1.8$')
ax.text(1.25,3.1,r'$k_B T/\epsilon = 1.2$')
ax.text(1.2,3.7,r'$\rho_b \sigma^3 = 0.5925$')
ax.set_xlim(0,1.8)
ax.set_ylim(0,10)
ax.legend(loc='upper right',ncol=1)

fig.savefig('lennardjones-slitpore-rhob=0.5925-T=1.2-H=1.8.pdf')
fig.savefig('lennardjones-slitpore-rhob=0.5925-T=1.2-H=1.8.png', bbox_inches='tight')
plt.close()

#######################################################################
# figsize accepts only inches.
fig, ax = plt.subplots(1,1)
# fig.subplots_adjust(left=0.13, right=0.98, top=0.85, bottom=0.08,
#                     hspace=0.1, wspace=0.1)

MCdata = np.loadtxt('../MCdata/lj-slitpore-steele-T1.2-rhob0.5925-H2-GEMC.dat')
xMC,rhoMC = MCdata[:,0], MCdata[:,1]
ax.scatter(xMC,rhoMC,label='MC')

[x,n] = np.load('../results/profiles-lennardjones-slitpore-rhob=0.5925-T=1.2-H=2.0.npy')
ax.plot(x,n,'--',color='grey',label='Tang2004')

[x,n] = np.load('../results/profiles-lennardjones-slitpore-rhob=0.5925-T=1.2-H=2.0-Elvisparameters.npy')
ax.plot(x,n,'-',color='k',label='this work')

# ax[0,0].set_yscale('log')
ax.set_ylabel(r'$\rho(z) \sigma^3$')
ax.set_xlabel(r'$z/\sigma$')
ax.text(1.3,2.5,r'$H/\sigma= 2.0$')
ax.text(1.25,3.1,r'$k_B T/\epsilon = 1.2$')
ax.text(1.2,3.7,r'$\rho_b \sigma^3 = 0.5925$')
ax.set_xlim(0,2)
ax.set_ylim(0,8)
ax.legend(loc='upper right',ncol=1)

fig.savefig('lennardjones-slitpore-rhob=0.5925-T=1.2-H=2.0.pdf')
fig.savefig('lennardjones-slitpore-rhob=0.5925-T=1.2-H=2.0.png', bbox_inches='tight')
plt.close()

#######################################################################
# figsize accepts only inches.
fig, ax = plt.subplots(1,1)
# fig.subplots_adjust(left=0.13, right=0.98, top=0.85, bottom=0.08,
#                     hspace=0.1, wspace=0.1)

MCdata = np.loadtxt('../MCdata/lj-slitpore-steele-T1.2-rhob0.5925-H3-GEMC.dat')
xMC,rhoMC = MCdata[:,0], MCdata[:,1]
ax.scatter(xMC,rhoMC,label='MC')

[x,n] = np.load('../results/profiles-lennardjones-slitpore-rhob=0.5925-T=1.2-H=3.0.npy')
ax.plot(x,n,'--',color='grey',label='Tang2004')

[x,n] = np.load('../results/profiles-lennardjones-slitpore-rhob=0.5925-T=1.2-H=3.0-Elvisparameters.npy')
ax.plot(x,n,'-',color='k',label='this work')

# ax[0,0].set_yscale('log')
ax.set_ylabel(r'$\rho(z) \sigma^3$')
ax.set_xlabel(r'$z/\sigma$')
ax.text(1.1,3.3,r'$k_B T/\epsilon = 1.2$')
ax.text(1.05,3.6,r'$\rho_b \sigma^3 = 0.5925$')
ax.set_xlim(0,3)
ax.set_ylim(0,4)
ax.legend(loc='upper right',ncol=1)

fig.savefig('lennardjones-slitpore-rhob=0.5925-T=1.2-H=3.0.pdf')
fig.savefig('lennardjones-slitpore-rhob=0.5925-T=1.2-H=3.0.png', bbox_inches='tight')
plt.close()

#######################################################################
# figsize accepts only inches.
fig, ax = plt.subplots(1,1)
# fig.subplots_adjust(left=0.13, right=0.98, top=0.85, bottom=0.08,
#                     hspace=0.1, wspace=0.1)

MCdata = np.loadtxt('../MCdata/lj-slitpore-steele-T1.2-rhob0.5925-H4-GEMC.dat')
xMC,rhoMC = MCdata[:,0], MCdata[:,1]
ax.scatter(xMC,rhoMC,label='MC')

[x,n] = np.load('../results/profiles-lennardjones-slitpore-rhob=0.5925-T=1.2-H=4.0.npy')
ax.plot(x,n,'--',color='grey',label='Tang2004')

[x,n] = np.load('../results/profiles-lennardjones-slitpore-rhob=0.5925-T=1.2-H=4.0-Elvisparameters.npy')
ax.plot(x,n,'-',color='k',label='this work')

# ax[0,0].set_yscale('log')
ax.set_ylabel(r'$\rho(z) \sigma^3$')
ax.set_xlabel(r'$z/\sigma$')
ax.text(1.1,3.3,r'$k_B T/\epsilon = 1.2$')
ax.text(1.05,3.6,r'$\rho_b \sigma^3 = 0.5925$')
ax.set_xlim(0,4)
ax.set_ylim(0,4)
ax.legend(loc='upper right',ncol=1)

fig.savefig('lennardjones-slitpore-rhob=0.5925-T=1.2-H=4.0.pdf')
fig.savefig('lennardjones-slitpore-rhob=0.5925-T=1.2-H=4.0.png', bbox_inches='tight')
plt.close()

#######################################################################
# figsize accepts only inches.
fig, ax = plt.subplots(1,1)
# fig.subplots_adjust(left=0.13, right=0.98, top=0.85, bottom=0.08,
#                     hspace=0.1, wspace=0.1)

MCdata = np.loadtxt('../MCdata/lj-slitpore-steele-T1.2-rhob0.5925-H7.5-GEMC.dat')
xMC,rhoMC = MCdata[:,0], MCdata[:,1]
ax.scatter(xMC,rhoMC,label='MC')

[x,n] = np.load('../results/profiles-lennardjones-slitpore-rhob=0.5925-T=1.2-H=7.5.npy')
ax.plot(x,n,'--',color='grey',label='Tang2004')

[x,n] = np.load('../results/profiles-lennardjones-slitpore-rhob=0.5925-T=1.2-H=7.5-Elvisparameters.npy')
ax.plot(x,n,'-',color='k',label='this work')

# ax[0,0].set_yscale('log')
ax.set_ylabel(r'$\rho(z) \sigma^3$')
ax.set_xlabel(r'$z/\sigma$')
ax.text(2.5,2.0,r'$k_B T/\epsilon = 1.2$')
ax.text(2.4,2.3,r'$\rho_b \sigma^3 = 0.5925$')
ax.set_xlim(0,7.5)
ax.set_ylim(0,4)
ax.legend(loc='upper right',ncol=1)

fig.savefig('lennardjones-slitpore-rhob=0.5925-T=1.2-H=7.5.pdf')
fig.savefig('lennardjones-slitpore-rhob=0.5925-T=1.2-H=7.5.png', bbox_inches='tight')
plt.close()