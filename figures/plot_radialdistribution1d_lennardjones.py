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

df = pd.read_excel('../MCdata/MCdata-radialdistribution-lennardjones-Verlet1968.xls',sheet_name='rhob=0.850') 

ax.scatter(df['r'],df['KT=0.658'],marker='o',edgecolors='C0',facecolors='none',linewidth=widthcircle,label='MD')

[x,n] = np.load('../results/radialdistribution-lennardjones-rhob=0.85-T=0.658.npy')
ax.plot(x,n,'--',color='grey',label='Tang2004')

[x,n] = np.load('../results/radialdistribution-lennardjones-rhob=0.85-T=0.658-Elvisparameters.npy')
ax.plot(x,n,'-',color='k',label='this work')

# ax[0,0].set_yscale('log')
ax.set_ylabel(r'$g(r)$')
ax.set_xlabel(r'$r/\sigma$')
ax.text(1.7,2.0,r'$k_B T/\epsilon = 0.658$')
ax.text(1.75,1.7,r'$\rho_b \sigma^3 = 0.85$')
ax.set_xlim(0.0,4.0)
ax.set_ylim(0,3.5)
ax.legend(loc='upper right',ncol=1)

fig.savefig('lennardjones-radialdistribution-rhob=0.85-T=0.658.pdf')
fig.savefig('lennardjones-radialdistribution-rhob=0.85-T=0.658.png', bbox_inches='tight')
plt.close()

#######################################################################
# figsize accepts only inches.
fig, ax = plt.subplots(1,1)
# fig.subplots_adjust(left=0.13, right=0.98, top=0.85, bottom=0.08,
#                     hspace=0.1, wspace=0.1)

df = pd.read_excel('../MCdata/MCdata-radialdistribution-lennardjones-Verlet1968.xls',sheet_name='Argon') # Shukla2000

ax.scatter(df['r']/3.405,df['KT=0.71-rhob=0.84'],marker='o',edgecolors='C0',facecolors='none',linewidth=widthcircle,label=r'${}^{36}$Ar @ 85 K')

[x,n] = np.load('../results/radialdistribution-lennardjones-rhob=0.84-T=0.71.npy')
ax.plot(x,n,'--',color='grey',label='Tang2004')

[x,n] = np.load('../results/radialdistribution-lennardjones-rhob=0.84-T=0.71-Elvisparameters.npy')
ax.plot(x,n,'-',color='k',label='this work')

# ax[0,0].set_yscale('log')
ax.set_ylabel(r'$g(r)$')
ax.set_xlabel(r'$r/\sigma$')
ax.text(1.7,2.0,r'$k_B T/\epsilon = 0.71$')
ax.text(1.75,1.7,r'$\rho_b \sigma^3 = 0.84$')
ax.set_xlim(0.0,8.0)
ax.set_ylim(0,3.5)
ax.legend(loc='upper right',ncol=1)

fig.savefig('lennardjones-radialdistribution-rhob=0.84-T=0.71.pdf')
fig.savefig('lennardjones-radialdistribution-rhob=0.84-T=0.71.png', bbox_inches='tight')
plt.close()

#######################################################################
# figsize accepts only inches.
fig, ax = plt.subplots(1,1)
# fig.subplots_adjust(left=0.13, right=0.98, top=0.85, bottom=0.08,
#                     hspace=0.1, wspace=0.1)

df = pd.read_excel('../MCdata/MCdata-radialdistribution-lennardjones-Reatto1986.xls',sheet_name='rhob=0.84-kT=0.75') # Shukla2000

ax.scatter(df['r/sigma'],df['g(r)'],marker='o',edgecolors='C0',facecolors='none',linewidth=widthcircle,label='MD')

[x,n] = np.load('../results/radialdistribution-lennardjones-rhob=0.84-T=0.75.npy')
ax.plot(x,n,'--',color='grey',label='Tang2004')

[x,n] = np.load('../results/radialdistribution-lennardjones-rhob=0.84-T=0.75-Elvisparameters.npy')
ax.plot(x,n,'-',color='k',label='this work')

# ax[0,0].set_yscale('log')
ax.set_ylabel(r'$g(r)$')
ax.set_xlabel(r'$r/\sigma$')
ax.text(1.7,2.0,r'$k_B T/\epsilon = 0.75$')
ax.text(1.75,1.7,r'$\rho_b \sigma^3 = 0.84$')
ax.set_xlim(0.0,3.0)
ax.set_ylim(0,3.5)
ax.legend(loc='upper right',ncol=1)

fig.savefig('lennardjones-radialdistribution-rhob=0.84-T=0.75.pdf')
fig.savefig('lennardjones-radialdistribution-rhob=0.84-T=0.75.png', bbox_inches='tight')
plt.close()