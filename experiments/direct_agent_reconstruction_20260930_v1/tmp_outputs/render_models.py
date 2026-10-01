"""Render these manually built JSON models without looking at source images.
Usage: geosym/bin/python render_models.py RUN_DIRECTORY
"""
import json,math,sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle,Polygon,Arc,FancyArrowPatch
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':16})
def render(path):
 m=json.loads(path.read_text());p={k:np.array(x['coordinates'],float) for k,x in m['points'].items()};entities={x['id']:x for x in m['entities']}
 def arcpoints(e):
  th=np.linspace(e['start_deg'],e['end_deg'],160)*np.pi/180;c=p[e['center']];return c+e['radius']*np.column_stack([np.cos(th),np.sin(th)])
 bounds=list(p.values())
 for e in entities.values():
  if e['kind']=='circle':
   c=p[e['center']];r=e['radius'];bounds.extend([c-r,c+r])
  elif e['kind']=='arc':bounds.extend(arcpoints(e))
 for d in m['render_instructions']:
  if d['kind']=='text':bounds.append(d['at'])
 lo=np.min(bounds,axis=0);hi=np.max(bounds,axis=0);span=max(hi-lo);small=span*.035
 fig,ax=plt.subplots(figsize=(6.5,5),dpi=160);fig.subplots_adjust(0.05,.05,.95,.95);ax.set_aspect('equal');ax.axis('off')
 def line(ps,color='#222222',width=1.7,style='solid',z=3):
  ps=np.array(ps);ax.plot(ps[:,0],ps[:,1],color=color,lw=width,linestyle=style,zorder=z,solid_capstyle='round')
 def text(s,at,size=17,color='#222222',**kw):
  return ax.text(*at,s,fontsize=size,color=color,ha='center',va='center',zorder=10,**kw)
 for d in m['render_instructions']:
  kind=d['kind'];color=d.get('color','#222222')
  if kind=='label':
   ax.annotate(d['text'],p[d['point']],xytext=d['offset'],textcoords='offset points',fontsize=19,fontstyle='italic',ha='center',va='center',zorder=10)
  elif kind=='text':text(d['text'],d['at'],d.get('size',16),color)
  elif kind=='dot':ax.scatter(*p[d['point']],s=30 if d.get('hollow') else 40,facecolors='white' if d.get('hollow') else color,edgecolors=color,linewidths=1.2,zorder=8)
  elif kind=='stroke':
   e=entities[d['entity']];a,b=[p[k] for k in e['points']];ext=d['extension'];style=d['style']
   if e['kind']=='segment':line([a,b],color,d['width'],style)
   else:
    if e['kind']=='ray':end=b+(ext or 0)*(b-a);start=a;ar='-|>'
    else:start=a-(ext or 0)*(b-a);end=b+(ext or 0)*(b-a);ar='<|-|>'
    ax.add_patch(FancyArrowPatch(start,end,arrowstyle=ar,mutation_scale=16,lw=d['width'],color=color,linestyle=style,zorder=3,shrinkA=0,shrinkB=0))
    bounds.extend([start,end])
  elif kind=='circle':
   e=entities[d['entity']];ax.add_patch(Circle(p[e['center']],e['radius'],fill=False,color=color,lw=1.7,zorder=3))
  elif kind=='arc':line(arcpoints(entities[d['entity']]),color)
  elif kind=='fill_polygon':
   e=entities[d['entity']];ax.add_patch(Polygon([p[k] for k in e['vertices']],closed=True,facecolor=color,edgecolor='none',zorder=0))
  elif kind=='fill_curvilinear':
   e=entities[d['entity']];pts=np.concatenate([arcpoints(part) for part in e['boundary']]);ax.add_patch(Polygon(pts,closed=True,facecolor=color,edgecolor='none',zorder=0))
  elif kind=='right_angle':
   a,o,b=[p[k] for k in d['points']];u=(a-o)/np.linalg.norm(a-o);w=(b-o)/np.linalg.norm(b-o);s=d['size'] or small;line([o+s*u,o+s*(u+w),o+s*w],color,1.2)
  elif kind=='ticks':
   a,b=[p[k] for k in d['points']];u=(b-a)/np.linalg.norm(b-a);n=np.array([-u[1],u[0]]);mid=(a+b)/2
   for i in range(d['count']):
    c=mid+(i-(d['count']-1)/2)*small*.4*u;line([c-small*.6*n,c+small*.6*n],color,1.35,z=5)
  elif kind=='angle_mark':
   a,o,b=[p[k] for k in d['points']];a0=math.atan2(*(a-o)[::-1]);a1=math.atan2(*(b-o)[::-1]);diff=(a1-a0+math.pi)%(2*math.pi)-math.pi;th=np.linspace(a0,a0+diff,60);r=d['radius'] or small*2;line(o+r*np.column_stack([np.cos(th),np.sin(th)]),color,1.2,z=5)
  elif kind=='dimension':
   a,b=[p[k] for k in d['points']];u=(b-a)/np.linalg.norm(b-a);n=np.array([-u[1],u[0]]);off=d['offset'];c=a+off*n;f=b+off*n;line([c,f],color,1.0)
   for q in [c,f]:line([q-small*.6*n,q+small*.6*n],color,1.0)
   text(d['text'],(c+f)/2,size=17,bbox=dict(facecolor='white',edgecolor='none',pad=3));bounds.extend([c-small*n,f+small*n])
  elif kind=='parallel_mark':
   a,b=[p[k] for k in d['points']];c=(a+b)/2;u=(b-a)/np.linalg.norm(b-a);n=np.array([-u[1],u[0]]);pts=[c-small*.7*u,c+small*.7*u+small*.45*n,c+small*.7*u-small*.45*n];ax.add_patch(Polygon(pts,closed=True,facecolor=color,edgecolor=color,zorder=6))
  elif kind=='leader':
   ax.add_patch(FancyArrowPatch(d['start'],d['end'],arrowstyle='->',mutation_scale=12,color=color,lw=1.2,connectionstyle='arc3,rad=-0.35' if d.get('curved') else 'arc3,rad=0',zorder=7))
  else:raise ValueError(kind)
 lo=np.min(bounds,axis=0);hi=np.max(bounds,axis=0);pad=span*.10;ax.set_xlim(lo[0]-pad,hi[0]+pad);ax.set_ylim(lo[1]-pad,hi[1]+pad)
 fig.savefig(path.parent/'redraw.png',facecolor='white');plt.close(fig)
 return str(path.parent/'redraw.png')
if __name__=='__main__':
 run=Path(sys.argv[1]);paths=sorted(run.glob('*/model.json'))
 if len(paths)!=20:raise ValueError(f'Expected 20 models, found {len(paths)}')
 for path in paths:print(render(path))
