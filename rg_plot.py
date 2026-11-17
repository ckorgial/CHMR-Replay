"""Regenerate signed paired comparisons and the RG architecture without archived KITTI substitution.

Usage: python rg_plot.py                # MR/PA confirmation and RG method diagram
       python rg_plot.py --require-kitti # additionally demand a rerun KITTI metrics file
"""
from pathlib import Path
import argparse
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
mpl.rcParams.update({'pdf.fonttype':42,'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
folder=Path('rg_results');out=Path('rg_results')
def export(fig,name):
    for ext in ['pdf','png']:fig.savefig(out/f'{name}.{ext}',dpi=300,bbox_inches='tight')
    plt.close(fig)

def paired(require_kitti=False):
    m=pd.read_csv(folder/'main_metrics.csv')
    datasets=[('MR',m,'confirmation'),('PA',m,'confirmation')]
    if (folder/'kitti_metrics.csv').exists():
        k=pd.read_csv(folder/'kitti_metrics.csv')
        datasets.append(('KITTI',k,'masked'))
    elif require_kitti:raise FileNotFoundError('KITTI curves require fresh kitti_metrics.csv from raw KITTI inputs')
    fig,axes=plt.subplots(1,len(datasets),figsize=(2.55*len(datasets),2.24),squeeze=False,layout='constrained')
    for ax,(suite,data,split) in zip(axes[0],datasets):
        selected=data[(data.suite==suite)&(data.comparator=='CA')&(data.split==split)].set_index('method')
        for i,(method,color) in enumerate([('Fixed mix','#657182'),('G','#16776d'),('PCG','#183d68')]):
            r=selected.loc[method]
            ax.plot([1000*r.ci_low,1000*r.ci_high],[i,i],color=color,lw=2,solid_capstyle='round')
            ax.scatter(1000*r.difference,i,color=color,s=21,zorder=5)
        ax.axvline(0,color='#888888',lw=.8)
        ax.set(title=f'{suite} ({split})',yticks=[0,1,2],ylim=(-.6,2.6),xlabel=r'RMSE $-$ CA (mm)')
        ax.set_yticklabels(['Fixed mix','Gate only','Full'])
        ax.grid(axis='x',lw=.4,alpha=.4)
    export(fig,'increment')

def diagram():
    fig,ax=plt.subplots(figsize=(3.5,3.05));ax.set_xlim(0,100);ax.set_ylim(0,100);ax.axis('off')
    def bx(x,y,w,h,txt,fill,edge,size=8.0):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.28,rounding_size=2',facecolor=fill,edgecolor=edge,lw=1.0))
        ax.text(x+w/2,y+h/2,txt,ha='center',va='center',fontsize=size,linespacing=1.25,color='#172B43')
    def arrow(x,y,x2,y2):ax.annotate('',xy=(x2,y2),xytext=(x,y),arrowprops={'arrowstyle':'-|>','color':'#415569','lw':1,'mutation_scale':10})
    bx(5,87,90,12,'Observed depths $y_0,\\ldots,y_J$\nLoss origin $J$ (no future updates)','#e7edf4','#406588')
    arrow(50,87,50,83)
    bx(5,69,90,14,'Common pre-loss origins $o$, $o+h\\leq J$\nFIR linear and quadratic; CA prefix replay','#edf4ed','#55826a',7.6)
    arrow(50,69,50,65)
    bx(5,51,90,14,'Horizon proxy scores $A_h,B_h$\nAR(1) correction $s_h$ and pooled anchor $p$','#fff5e8','#bf7c39',7.6)
    arrow(28,51,28,47);arrow(73,51,73,47)
    bx(3,30,52,17,'Corrected, shrunk FIR\nweights $v_1,\\ldots,v_H$','#edf2fa','#41638b',8)
    bx(59,30,38,17,'CA gate $g_h$\nfrom fixed FIR anchor','#edf4ed','#55826a',7.7)
    arrow(28,30,28,26);arrow(78,30,78,26)
    bx(3,9,94,17, r'Track-loss forecast, $h=1,\ldots,H$'+'\n'+r'$(1-g_h)\,f_{\mathrm{FIR}}(v_h)+g_h\,f_{\mathrm{CA}}$', '#f1ebf8','#795ca3',8.2)
    ax.text(50,3,'Past: replay scores    |    Future: forecasts only',ha='center',fontsize=7.3,color='#42586d')
    fig.subplots_adjust(left=.02,right=.98,top=.99,bottom=.01)
    export(fig,'architecture_rg')
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--require-kitti',action='store_true');args=parser.parse_args()
    paired(args.require_kitti);diagram()
