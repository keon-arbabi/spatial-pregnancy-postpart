import polars as pl
import matplotlib.pyplot as plt
import matplotlib.patches as mp
import numpy as np
from matplotlib.patches import FancyBboxPatch as Box,PathPatch
from matplotlib.path import Path
fig,axs=plt.subplots(3,1,figsize=(14,12))
wdir='/home/karbabi/spatial-pregnancy-postpart'
df=pl.read_csv(f'{wdir}/output/data/de_results_sig.csv')
df=df.filter(pl.col('FDR')<0.05)
p1=df.filter(pl.col('contrast')=='PREG_vs_CTRL')
p2=df.filter(pl.col('contrast')=='POSTPART_vs_PREG')
def get_pattern(ct,p1,p2,fc_th=0.3,cnt_th=3):
 d1=p1.filter(pl.col('cell_type')==ct)
 d2=p2.filter(pl.col('cell_type')==ct)
 if d1.height<cnt_th and d2.height<cnt_th:return None
 up1=d1.filter(pl.col('logFC')>fc_th).height
 dn1=d1.filter(pl.col('logFC')<-fc_th).height
 up2=d2.filter(pl.col('logFC')>fc_th).height
 dn2=d2.filter(pl.col('logFC')<-fc_th).height
 if up1>dn1+cnt_th:s1='U'
 elif dn1>up1+cnt_th:s1='D'
 else:s1='='
 if up2>dn2+cnt_th:s2='U'
 elif dn2>up2+cnt_th:s2='D'
 else:s2='-'
 return s1+s2
cts=sorted(set(p1['cell_type'].to_list()+p2['cell_type'].to_list()))
patterns={ct:get_pattern(ct,p1,p2) for ct in cts}
patterns={k:v for k,v in patterns.items() if v}
glut=[c for c in cts if 'Glut' in c and c in patterns]
gaba=[c for c in cts if 'Gaba' in c and c in patterns]
nn=[c for c in cts if 'NN' in c and c in patterns]
def bezier_path(x0,y0,x1,y1,h=0.5):
 verts=[(x0,y0),(x0+h,y0),(x1-h,y1),(x1,y1)]
 codes=[Path.MOVETO,Path.CURVE4,Path.CURVE4,Path.CURVE4]
 return Path(verts,codes)
def draw_flow(ax,cts,pats,title):
 ax.set_xlim(-1,11);ax.set_ylim(-1,11);ax.axis('off')
 ax.text(5,10.5,title,ha='center',fontsize=16,weight='bold')
 cats=['UD','DU','=U','=D','U-','D-']
 cols={'U':'#FF8787','D':'#74A9F2','=':'#E8E8E8','-':'#E8E8E8'}
 grps={c:[k for k,v in pats.items() if v==c and k in cts] for c in cats}
 tot=len([c for c in cts if c in pats])
 if tot==0:return
 y_h={c:len(grps[c])/tot*8 for c in cats}
 y_s={}
 y=9
 for c in cats:
  y_s[c]=y;y-=y_h[c]
 left_insets=[]
 for i,s in enumerate(['D-','U-','-D','-U','DU','UD']):
  ix=ax.inset_axes([0.02,0.85-i*0.14,0.08,0.08])
  ix.set_xlim(0,2);ix.set_ylim(-0.2,1.2);ix.axis('off')
  if s[0]!='-':
   ix.plot([0,0.8],[0.5,0.9 if s[0]=='U' else 0.1],
   color=cols['U' if s[0]=='U' else 'D'],lw=3)
  if s[1]!='-':
   ix.plot([0.8,1.6],[0.9 if s[0]=='U' else 0.1,
   0.5 if s[1]=='-' else (0.9 if s[1]=='U' else 0.1)],
   color=cols[s[1]],lw=3)
  else:
   ix.plot([0.8,1.6],[0.9 if s[0]=='U' else 0.1,0.5],color='gray',lw=3)
  left_insets.append(ix)
 for c in cats:
  if not grps[c]:continue
  ym=y_s[c]-y_h[c]/2
  ax.text(1.5,ym,c,ha='center',va='center',fontsize=12,weight='bold')
  ct_h=y_h[c]/max(len(grps[c]),1)
  for i,ct in enumerate(grps[c]):
   y_ct=y_s[c]-i*ct_h-ct_h/2
   p=bezier_path(2,ym,4,y_ct,0.8)
   pp=PathPatch(p,facecolor='none',edgecolor=cols[c[0]],lw=2,alpha=0.7)
   ax.add_patch(pp)
   lbl=ct.replace(' Glut','').replace(' Gaba','').replace(' NN','')
   if c[0]!='=':bcol=cols[c[0]]
   elif c[1]!='-':bcol=cols[c[1]]
   else:bcol='#F0F0F0'
   bx=Box((4.2,y_ct-ct_h*0.35),1.8,ct_h*0.7,boxstyle="round,pad=0.05",
   facecolor=bcol,edgecolor='gray',lw=0.5,alpha=0.8)
   ax.add_patch(bx)
   ax.text(5.1,y_ct,lbl,ha='center',va='center',fontsize=8)
   if c[1] not in ['-','=']:
    p2=bezier_path(6,y_ct,8,ym,0.8)
    pp2=PathPatch(p2,facecolor='none',edgecolor=cols[c[1]],lw=2,alpha=0.7)
    ax.add_patch(pp2)
 preg_y=5;post_y=5
 for c in cats:
  if not grps[c]:continue
  ym=y_s[c]-y_h[c]/2
  if c[0]!='=':
   ax.add_patch(mp.Rectangle((0.5,ym-y_h[c]*0.4),0.6,y_h[c]*0.8,
   facecolor=cols[c[0]],edgecolor='none',alpha=0.6))
  if c[1] not in ['-','=']:
   ax.add_patch(mp.Rectangle((8.4,ym-y_h[c]*0.4),0.6,y_h[c]*0.8,
   facecolor=cols[c[1]],edgecolor='none',alpha=0.6))
 ax.text(0.8,9.5,'Control',ha='center',fontsize=10,style='italic')
 ax.text(0.8,-0.5,'Pregnancy',ha='center',fontsize=10,style='italic')
 ax.text(8.7,9.5,'Pregnancy',ha='center',fontsize=10,style='italic')
 ax.text(8.7,-0.5,'Postpartum',ha='center',fontsize=10,style='italic')
 uflow=sum(len(grps[c]) for c in cats if 'U' in c[0] and c[0]!='=')
 dflow=sum(len(grps[c]) for c in cats if 'D' in c[0])
 if uflow>0:
  ax.add_patch(mp.FancyArrowPatch((0.3,5+uflow/tot*2),(0.3,5-uflow/tot*2),
  arrowstyle='-',color=cols['U'],lw=uflow/tot*20,alpha=0.6))
 if dflow>0:
  ax.add_patch(mp.FancyArrowPatch((1.1,5-dflow/tot*2),(1.1,5+dflow/tot*2),
  arrowstyle='-',color=cols['D'],lw=dflow/tot*20,alpha=0.6))
 u2flow=sum(len(grps[c]) for c in cats if 'U' in c[1] and c[1]!='=')
 d2flow=sum(len(grps[c]) for c in cats if 'D' in c[1])
 if u2flow>0:
  ax.add_patch(mp.FancyArrowPatch((9.1,5+u2flow/tot*2),(9.1,5-u2flow/tot*2),
  arrowstyle='-',color=cols['U'],lw=u2flow/tot*20,alpha=0.6))
 if d2flow>0:
  ax.add_patch(mp.FancyArrowPatch((8.3,5-d2flow/tot*2),(8.3,5+d2flow/tot*2),
  arrowstyle='-',color=cols['D'],lw=d2flow/tot*20,alpha=0.6))
draw_flow(axs[0],glut,patterns,'Glutamatergic neurons')
draw_flow(axs[1],gaba,patterns,'GABAergic neurons')
draw_flow(axs[2],nn,patterns,'Non-neuronal cells')
plt.tight_layout()
plt.savefig(f'{wdir}/figures/pregnancy_transitions.png',dpi=300,bbox_inches='tight')
plt.savefig(f'{wdir}/figures/pregnancy_transitions.svg',bbox_inches='tight')
plt.close() 