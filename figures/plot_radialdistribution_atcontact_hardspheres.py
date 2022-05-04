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

df = pd.read_excel('../MCdata/MCdata-hardsphere-Barker1971.xls',sheet_name='radialdistributionfunction_atsigma') 

ax.scatter(df['rhob'],df['g(sigma)'],marker='o',edgecolors='C0',facecolors='none',linewidth=widthcircle,label='MC')

rhob = np.array([0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9])

N = 256
gsigma = np.empty_like(rhob)
for i in range(rhob.size):
    n1 = np.load('../densityfield-fmt-wbi-rhob'+str(rhob[i])+'-N'+str(N)+'-delta0.05.npy')
    gsigma[i] = n1.max()/rhob[i]
ax.plot(rhob,gsigma,'-',color='k',label='$256^3, 0.05\sigma$')

N = 128
gsigma = np.empty_like(rhob)
for i in range(rhob.size):
    n1 = np.load('../densityfield-fmt-wbi-rhob'+str(rhob[i])+'-N'+str(N)+'-delta0.1.npy')
    gsigma[i] = n1.max()/rhob[i]
ax.plot(rhob,gsigma,'--',color='C3',label='$128^3,0.1\sigma$')

N = 64
gsigma = np.empty_like(rhob)
for i in range(rhob.size):
    n1 = np.load('../densityfield-fmt-wbi-rhob'+str(rhob[i])+'-N'+str(N)+'-delta0.2.npy')
    gsigma[i] = n1.max()/rhob[i]
ax.plot(rhob,gsigma,':',color='C2',label='$64^3,0.2\sigma$')

# ax[0,0].set_yscale('log')
ax.set_ylabel(r'$g(\sigma)$')
ax.set_xlabel(r'$\rho_b \sigma^3$')
ax.set_xlim(0.2,0.9)
ax.set_ylim(1,6)
ax.legend(loc='upper left',ncol=1)

fig.savefig('radialdistribution_atcontact_hardspheres.pdf')
fig.savefig('radialdistribution_atcontact_hardspheres.png', bbox_inches='tight')
plt.close()