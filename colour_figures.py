"""Render the paper figures from the archived full-precision results."""
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import pandas as pd
import numpy as np
from models import bank
out=Path(__file__).resolve().parent/'results'
mpl.rcParams.update({'pdf.fonttype':42,'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
blue,teal,orange,purple='#24568A','#00786D','#BA5912','#6B4596'
def export(fig,name):
    for ext in ('pdf','png'):
        fig.savefig(out/f'{name}.{ext}',dpi=360,facecolor='white')
    plt.close(fig)

def architecture():
    fig=plt.figure(figsize=(3.45,3.95))
    ax=fig.add_axes([.009,.007,.982,.986],xlim=(0,100),ylim=(0,100))
    ax.axis('off')
    def box(x,y,w,h,face,edge):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.25,rounding_size=1.8',facecolor=face,edgecolor=edge,lw=1.05))
    def label(x,y,s,**kw):
        ax.text(x,y,s,ha='center',va='center',fontsize=9,**kw)
    def arrow(x,y,X,Y):
        ax.annotate('',xy=(X,Y),xytext=(x,y),arrowprops={'arrowstyle':'-|>','color':'#465566','lw':.9,'shrinkA':0,'shrinkB':0,'mutation_scale':8})
    box(2,88,96,11,'#E5EFF9',blue)
    label(50,95.4,'Pre-loss range measurements',weight='bold',color=blue)
    label(50,90.9,r'Common origins $o$ and loss origin $J$')
    ax.plot([25,75],[85.6,85.6],color='#465566',lw=.9);ax.plot([50,50],[88,85.6],color='#465566',lw=.9)
    for x,bg,edge,title,p in [(2,'#E3F3EE',teal,'Linear FIR',1),(52,'#FEF0DC',orange,'Quadratic FIR',2)]:
        arrow(x+23,85.6,x+23,83);box(x,73,46,10,bg,edge);label(x+23,79.9,title,weight='bold',color=edge);label(x+23,75.6,rf'$f_{{{p},o,h}},\ f_{{{p},J,h}}$')
        arrow(x+23,73,x+23,70)
    box(2,34,96,36,'#F2EDF8',purple)
    label(50,66.3,'Historical replay · weight estimation',weight='bold',color=purple)
    label(50,60.3,r'$o\in\mathcal{O},\quad o+h\leq J$')
    label(50,54.6,r'$D=f_{2,o,h}-f_{1,o,h},\quad e=y_{o+h}-f_{1,o,h}$')
    label(50,48.7,r'$A_h=\langle D^2\rangle_\alpha,\quad B_h=\langle De\rangle_\alpha$')
    ax.plot([8,92],[45.4,45.4],lw=.6,color='#CDBDDC')
    label(50,41.4,'Separate weights or one shared weight',weight='bold',color=purple)
    label(50,36.5,r'$w_1,\ldots,w_H$ or $w_{\mathrm{pool}}$ from mean $A_h,B_h$')
    arrow(50,34,50,31)
    box(2,17,96,14,'#E5EFF9',blue)
    label(50,27.1,'Unobserved interval after loss',weight='bold',color=blue)
    label(50,20.8,r'$\hat r_{J+h\mid J}= f_{1,J,h}+w(f_{2,J,h}-f_{1,J,h})$')
    arrow(50,17,50,13.6)
    box(2,1,96,12.5,'#FAEAF0','#963B5C')
    label(50,9.8,'Threshold screening',weight='bold',color='#963B5C')
    label(50,4.7,r'$\hat c_h=\mathbf{1}\{\hat r_{J+h\mid J}>g\}$')
    export(fig,'architecture')

def curves(kind):
    weights=kind=='weights'
    file='weight_diagnostic' if weights else 'subgroups'
    data=pd.read_csv(out/f'{file}.csv')
    fig,axs=plt.subplots(1,3,figsize=(6.8,2.06),sharey=True)
    for i,ax in enumerate(axs):
        if weights:
            rho=[0,.4,.85][i];sub=data[data.rho==rho]
            lines=[(sub.h,sub[k],label) for k,label in [('range','Range optimum'),('proxy','Proxy optimum'),('mean_fitted','Mean fitted')]]
            if 'lower' in sub:
                ax.fill_between(sub.h,sub.lower,sub.upper,color=teal,alpha=.22)
            title=rf'$\rho={rho:g}$'
        else:
            lines=[]
            for m in ['One-step','CHMR','Pooled']:
                sub=data[(data.suite=='MR')&(data['subset']==f'group {i}')&(data.method==m)]
                lines.append((sub.horizon,sub.rmse,m))
            title=['Cruise','Queue approach','Cut-in settling'][i]
        for (x,y,label),c,style in zip(lines,[blue,orange,teal],['-','--',':o']):
            ax.plot(x,y,style,color=c,label=label,ms=3,lw=1.65,markerfacecolor='white')
        ax.grid(axis='y',color='#DCE2E6',lw=.5)
        ax.set(title=title,xticks=[1,5,10],xlabel='Horizon (frames)')
    axs[0].set_ylabel('Quadratic weight' if weights else 'RMSE (m)')
    fig.legend(*axs[0].get_legend_handles_labels(),loc='upper center',ncol=3,frameon=False)
    fig.tight_layout(rect=[0,0,1,.85])
    export(fig,kind)
if __name__=='__main__':
    architecture()
    for k in ('weights','regimes'):curves(k)
    fig,ax=plt.subplots(figsize=(3.45,1.92))
    for p,c,style in [(1,teal,'-o'),(2,orange,'--s')]:
        ax.plot(np.arange(1,11),np.sum(bank(15,p)**2,axis=0),style,color=c,label=['Linear','Quadratic'][p-1],ms=3)
    ax.set(xlabel='Horizon (frames)',ylabel='White-noise gain',xticks=[1,5,10]);ax.legend(frameon=False,loc='upper left')
    fig.tight_layout();export(fig,'noise_gain')
