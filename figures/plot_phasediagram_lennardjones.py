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

df = pd.read_excel('../MCdata/MCdata-lennardjones-phasediagram.xls',sheet_name='NIST') 

ax.scatter(df['rho1'],df['T'],marker='o',edgecolors='C0',facecolors='none',linewidth=widthcircle,label='MC')
ax.scatter(df['rho2'],df['T'],marker='o',edgecolors='C0',facecolors='none',linewidth=widthcircle)


# ax.set_yscale('log')
ax.set_ylabel(r'$k_B T /\epsilon$')
ax.set_xlabel(r'$\rho_b \sigma^3$')
ax.set_xlim(0.0,0.9)
ax.set_ylim(0.6,1.4)
ax.legend(loc='upper right',ncol=1)

fig.savefig('phasediagram_lennardjones.pdf')
fig.savefig('phasediagram_lennardjones.png', bbox_inches='tight')
plt.close()