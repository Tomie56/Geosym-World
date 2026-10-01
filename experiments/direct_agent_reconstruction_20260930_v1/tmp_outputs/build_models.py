"""One-off, manually reasoned models. Reads only isolated image/question records.
No parser/library/project imports; no automated image parsing.
"""
from pathlib import Path
import json, math, numpy as np
ROOT=Path('/data/Geosym127K')
RUN=ROOT/'Geosym-World/artifacts/direct_agent_reconstruction/agent_visual_20260930_v1'
INPUTS=json.loads((RUN/'inputs.json').read_text())
BLUE='#007caf'; PINK='#ff1686'; BLACK='#222222'; ORANGE='#ff5b20'
def v(x): return np.array(x,dtype=float)
def polar(r,a): return v([r*math.cos(math.radians(a)),r*math.sin(math.radians(a))])
def cross(a,b): return float(a[0]*b[1]-a[1]*b[0])
def intersect(a,b,c,d):
 a,b,c,d=map(v,[a,b,c,d]); return a+(b-a)*cross(c-a,d-c)/cross(b-a,d-c)
def distance(a,b): return float(np.linalg.norm(v(a)-v(b)))
def angle(a,b,c):
 u,w=v(a)-v(b),v(c)-v(b); return math.degrees(math.acos(np.clip(np.dot(u,w)/np.linalg.norm(u)/np.linalg.norm(w),-1,1)))
def area(ps): return abs(sum(cross(v(a),v(b)) for a,b in zip(ps,ps[1:]+ps[:1])))/2
class Model:
 def __init__(self,n,summary,color=BLACK):
  self.n=n; self.color=color; self.p={}; self.entities=[]; self.relations=[]; self.q=[]; self.draw=[]; self.notes=[]; self.checks=[]; self.target=None; self.guarantee='exact_euclidean_realization'; self.summary=summary
 def point(self,name,xy,label=True,offset=(0,12),dot=False):
  self.p[name]=list(map(float,xy))
  if label: self.draw.append(dict(kind='label',point=name,text=name,offset=offset))
  if dot:self.draw.append(dict(kind='dot',point=name,color=self.color))
  return v(xy)
 def seg(self,a,b,name=None,style='solid',color=None,width=1.7,kind='segment',end=None):
  name=name or a+b; self.entities.append(dict(id=name,kind=kind,points=[a,b]))
  self.draw.append(dict(kind='stroke',entity=name,style=style,color=color or self.color,width=width,extension=end))
  return name
 def poly(self,ids,name,fill=None,color=None,outline=True):
  self.entities.append(dict(id=name,kind='polygon',vertices=ids))
  if fill:self.draw.append(dict(kind='fill_polygon',entity=name,color=fill))
  if outline:
   for a,b in zip(ids,ids[1:]+ids[:1]):self.seg(a,b,color=color)
 def circle(self,c,r,name='circle',color=None):
  self.entities.append(dict(id=name,kind='circle',center=c,radius=float(r)))
  self.draw.append(dict(kind='circle',entity=name,color=color or self.color))
 def arc(self,c,r,a,b,name,color=None,draw=True):
  self.entities.append(dict(id=name,kind='arc',center=c,radius=float(r),start_deg=float(a),end_deg=float(b),orientation='counterclockwise'))
  if draw:self.draw.append(dict(kind='arc',entity=name,color=color or self.color))
 def rel(self,kind,objects,evidence='image',**kw): self.relations.append(dict(kind=kind,objects=objects,evidence=evidence,**kw))
 def qty(self,id,value,bind,kind='length',evidence='image',unit=None):
  self.q.append(dict(id=id,value=value,bind=bind,kind=kind,evidence=evidence,unit=unit));return id
 def text(self,text,at,bind=None,color=BLACK,size=16):self.draw.append(dict(kind='text',text=text,at=list(map(float,at)),bind=bind,color=color,size=size))
 def right(self,a,o,b,size=None,color=PINK):self.draw.append(dict(kind='right_angle',points=[a,o,b],size=size,color=color))
 def ticks(self,a,b,count=1,color=PINK):self.draw.append(dict(kind='ticks',points=[a,b],count=count,color=color))
 def anglemark(self,a,o,b,r=None,color=BLACK):self.draw.append(dict(kind='angle_mark',points=[a,o,b],radius=r,color=color))
 def bracket(self,a,b,text,offset):self.draw.append(dict(kind='dimension',points=[a,b],text=text,offset=offset,color=BLACK))
 def check(self,name,actual,expected,tol=1e-8):self.checks.append(dict(name=name,actual=float(actual),expected=float(expected),tolerance=tol,passed=bool(abs(actual-expected)<=tol)))
 def write(self):
  inp=INPUTS[self.n-1]; out=RUN/f'{self.n:02d}_{inp["id"]}';out.mkdir(exist_ok=True)
  data=dict(index=self.n,sample_id=inp['id'],source=dict(image=inp['image'],question=inp['question'],image_sha256=inp['image_sha256']),summary=self.summary,representation=dict(coordinates='Cartesian, x right / y up',units='arbitrary unless specified by quantities',embedding_guarantee=self.guarantee,source='Direct visual reasoning by this agent; hand-built per image, no prior parser output'),points={k:dict(coordinates=p,source_label=k if any(d.get('kind')=='label' and d.get('point')==k for d in self.draw) else None) for k,p in self.p.items()},entities=self.entities,relations=self.relations,quantities=self.q,target=self.target,uncertainties=self.notes,render_instructions=self.draw,coordinate_checks=self.checks)
  if hasattr(self,'extra'):data.update(self.extra)
  (out/'model.json').write_text(json.dumps(data,ensure_ascii=False,indent=2))
  return data
models=[]
# 01: triangles around H, then complete the parallelogram along the two parallel carriers.
m=Model(1,'平行四边形 ADEG 内两条相交线段 CF、BE，交于 H；相似三角形 CBH 与 FEH。',BLUE)
H=v([0,0]);E=v([0,-6]);F=v([math.sqrt(100-1.25**2),-1.25]);C=-.7*F;B=-.7*E;delta=B-C;D=C-.85*delta;A=B+.4*delta;G=A+E-D
for k,p in dict(A=A,B=B,C=C,D=D,E=E,F=F,G=G,H=H).items():m.point(k,p,offset={'A':(0,12),'B':(-10,8),'C':(-10,7),'D':(-12,0),'E':(0,-14),'F':(10,-10),'G':(12,2),'H':(8,10)}[k])
m.poly(['A','D','E','G'],'ADEG');m.seg('B','E');m.seg('C','F')
for ids in [['D','C','B','A'],['E','F','G'],['C','H','F'],['B','H','E']]:m.rel('collinear_ordered',ids)
m.rel('parallelogram',['A','D','E','G'],'question');m.rel('similar_triangles',[['C','B','H'],['F','E','H']],'question',scale_CBHoverFEH=.7)
for id,val,bind,at in [('CH',7,['C','H'],(C+H)/2+[0,-1]),('FH',10,['F','H'],(F+H)/2+[0,1]),('FE',11,['F','E'],(F+E)/2+[1,-1]),('EH',6,['E','H'],(E+H)/2+[-1,0])]:m.qty(id,val,bind,evidence='image_and_question');m.text(str(val),at,id);m.check(id,distance(m.p[bind[0]],m.p[bind[1]]),val)
m.target=dict(kind='perimeter',object=['C','B','H'],derived_value=18.9,derivation='CB=0.7*FE=7.7; BH=0.7*EH=4.2; CH=7')
m.notes=['A、D 在直线 CB 上的位置不能唯一确定；本模型选择了与原图接近的一组。']
models.append(m.write())
# 02 angle bisector theorem gives SU=10; use exact three side lengths.
m=Model(2,'三角形 VRU 中 RS 平分 R 角，S 在 VU 内部。')
rx=(11.2**2-16**2+17**2)/34
for k,p in dict(V=[0,0],S=[7,0],U=[17,0],R=[rx,math.sqrt(11.2**2-rx**2)]).items():m.point(k,p,offset=(0,-14) if k!='R' else (0,14))
m.poly(['V','R','U'],'VRU');m.seg('R','S');m.rel('between',['V','S','U']);m.rel('angle_bisector',['R','S','V','U'],'question')
for id,val,at in [('VR',11.2,[1.2,5.4]),('RU',16,[11.4,6]),('VS',7,[3.5,-1]),('SU','x',[12,-1])]:m.qty(id,val,list(id));m.text(str(val),at,id)
m.target=dict(kind='solve_variable',variable='x',derived_value=10,equation='7/x=11.2/16')
m.check('VR',distance(m.p['V'],m.p['R']),11.2);m.check('RU',distance(m.p['R'],m.p['U']),16);m.check('bisected angle',angle(m.p['V'],m.p['R'],m.p['S']),angle(m.p['S'],m.p['R'],m.p['U']))
m.notes=['精确边长使三角形比原图高；原图不是按比例绘制。']
models.append(m.write())
# 03 perpendicular from isosceles vertex bisects JL.
m=Model(3,'等腰三角形 JML，MJ=ML；从 K 经 M 的射线垂直于底边 JL，JL=12。',BLUE)
t=math.radians(25); rot=np.array([[math.cos(t),-math.sin(t)],[math.sin(t),math.cos(t)]])
for k,p in dict(J=[0,6],K=[0,0],L=[0,-6],M=[-14,0]).items():m.point(k,rot@v(p),offset={'J':(0,15),'K':(14,2),'L':(12,-6),'M':(0,-16)}[k])
m.poly(['J','M','L'],'JML');m.seg('K','M',kind='ray',end=.1);m.right('J','K','M',size=1);m.ticks('J','M');m.ticks('M','L');m.bracket('J','L','12',2)
m.rel('equal_length',[['M','J'],['M','L']]);m.rel('perpendicular',['KM','JL']);m.rel('between',['J','K','L']);m.qty('JL',12,['J','L']);m.target=dict(kind='length',object=['K','L'],derived_value=6)
m.check('JL',distance(m.p['J'],m.p['L']),12);m.check('KM perpendicular JL',angle(m.p['J'],m.p['K'],m.p['M']),90);m.notes=['原图左端箭头解释为从 K 经 M 延伸的射线；题目只需要其垂线性质。']
models.append(m.write())
# 04 medians; match two supplied centroid lengths.
m=Model(4,'三角形 JKL 的三条中线交于重心 P；R、S、T 分别是三边中点。',BLUE)
P=v([0,0]);J=v([-6,-math.sqrt(45)]);R=v([-3.4,-math.sqrt(3.5**2-3.4**2)]);L=-2*R;K=-J-L;S=(K+L)/2;T=(J+L)/2
for k,p in dict(J=J,K=K,L=L,R=R,S=S,T=T,P=P).items():m.point(k,p,offset={'J':(-10,-10),'K':(-10,10),'L':(12,0),'R':(-14,2),'S':(0,14),'T':(5,-14),'P':(2,16)}[k])
m.poly(['J','K','L'],'JKL')
for a,b in [('J','S'),('K','T'),('L','R')]:m.seg(a,b)
for midpoint,ends,count in [('R',['J','K'],3),('S',['K','L'],1),('T',['J','L'],2)]:
 m.rel('midpoint',[midpoint,*ends]);m.ticks(ends[0],midpoint,count);m.ticks(midpoint,ends[1],count)
m.rel('concurrent',['JS','KT','LR','P']);m.qty('RP',3.5,['R','P'],evidence='question');m.qty('JP',9,['J','P'],evidence='question');m.target=dict(kind='length',object=['P','S'],derived_value=4.5)
m.check('RP',distance(R,P),3.5);m.check('JP',distance(J,P),9);m.check('centroid',float(np.linalg.norm((J+K+L)/3-P)),0);m.notes=['题干长度未额外印到图上；原图的单、双、三道等长刻痕分别保留。']
models.append(m.write())
# 05 Keep literal data; do not pretend the inconsistent source has an exact embedding.
m=Model(5,'两四边形按 BCDE↔RSTU 对应全等；逐项保留表达式，但全部数值约束不相容。',BLUE);m.guarantee='schematic_only_source_constraints_inconsistent'
for k,p in dict(B=[0,4],C=[10,4],D=[6.5,0],E=[3.5,0],R=[12,0],S=[22,0],T=[18.5,4],U=[15.5,4]).items():m.point(k,p,offset=(0,12) if k in 'BCTU' else (0,-12))
m.poly(['B','C','D','E'],'BCDE');m.poly(['R','S','T','U'],'RSTU');m.rel('congruent_polygons',[['B','C','D','E'],['R','S','T','U']],'question',correspondence=dict(B='R',C='S',D='T',E='U'))
for id,value,bind,at in [('BC','4w-7',['B','C'],[5,4.65]),('CD',11,['C','D'],[9.2,1.8]),('DE','3z+10',['D','E'],[5,-.65]),('RU',12,['R','U'],[12.8,2.5]),('UT','z+16',['U','T'],[17,4.65]),('RS','2w+13',['R','S'],[17,-.65])]:m.qty(id,value,bind);m.text(str(value),at,id,size=12)
for id,value,bind,at in [('angle_B','2x+9',['E','B','C'],[2.5,3.35]),('angle_D','2y-31',['C','D','E'],[5.8,1]),('angle_R',49,['U','R','S'],[13.5,.45]),('angle_T','y+11',['S','T','U'],[17.55,2.95])]:m.qty(id,value,bind,'angle_degrees');m.text(f'({value})°' if not isinstance(value,int) else f'{value}°',at,id,size=12)
m.draw.extend([dict(kind='leader',start=[6.1,.8],end=[6.45,.2],color=BLACK),dict(kind='leader',start=[17.8,3.2],end=[18.3,3.8],color=BLACK)])
w,z,x,y=10,3,20,42; diag1=math.sqrt(33**2+12**2-2*33*12*math.cos(math.radians(49)));diag2=math.sqrt(11**2+19**2-2*11*19*math.cos(math.radians(53)))
m.target=dict(kind='solve_variable',variable='y',algebraic_value=42,equation='2y-31=y+11',caveat='仅全等对应角等式的代数解；不是全部条件可实现的证明')
m.extra=dict(consistency_analysis=dict(implied_variables=dict(w=w,z=z,x=x,y=y),diagonal_CE_from_triangle_BCE=diag1,diagonal_CE_from_triangle_CDE=diag2,contradiction='同一对角线 CE 得到两个不同长度，因此文字和图内全部标注不能同时精确成立。'))
m.notes=['不能把看起来平行的边直接升级为平行条件，图中没有平行标记。','本图的坐标只保存外形和标注位置；角和长度公式为独立的题面声明，并非该坐标的真值。','未猜测或修正原题可能存在的笔误。']
models.append(m.write())
# 06 8-15-17 triangle on a diameter.
m=Model(6,'圆内三角形 ABC，以 AB 为直径；AC=8、BC=15 英寸，圆心为无字母实点。')
for k,p in dict(A=[-8.5,0],B=[8.5,0],C=[-8.5+64/17,-120/17],O=[0,0]).items():m.point(k,p,label=k!='O',dot=k=='O',offset={'A':(-15,0),'B':(15,0),'C':(-8,-15),'O':(0,0)}[k])
m.circle('O',8.5);m.poly(['A','B','C'],'ABC');m.rel('diameter',['A','B','circle'],'question');m.rel('on_circle',['A','B','C','circle']);m.qty('AC',8,['A','C'],evidence='question',unit='inch');m.qty('BC',15,['B','C'],evidence='question',unit='inch');m.target=dict(kind='diameter',object='circle',derived_value=17,unit='inch')
m.check('AC',distance(m.p['A'],m.p['C']),8);m.check('BC',distance(m.p['B'],m.p['C']),15);m.check('C on circle',distance(m.p['C'],m.p['O']),8.5);m.notes=['未把推导出的直角标记添加到原本没有直角标记的图上。']
models.append(m.write())
# 07 enforce angles and the diagonal bisection simultaneously.
m=Model(7,'无字母四边形，附两条对角线；求使其成为平行四边形的 y。',BLUE)
u=1/math.tan(math.radians(115));w=1/math.tan(math.radians(55))-u;s=18/math.hypot(w-u,1)
pts=dict(A=v([u,1])*s,B=v([0,0]),C=v([w,0])*s,D=v([w+u,1])*s);pts['I']=(pts['A']+pts['C'])/2
for k,p in pts.items():m.point(k,p,label=False)
m.poly(['A','B','C','D'],'quad');m.seg('A','C');m.seg('B','D');m.rel('parallelogram',['A','B','C','D'],'question_target');m.rel('intersection',['AC','BD','I'])
for id,value,bind,at,kind in [('AI','x+4',['A','I'],(pts['A']+pts['I'])/2+[0,.8],'length'),('IC','3x-6',['I','C'],(pts['I']+pts['C'])/2+[0,-.8],'length'),('ADB',55,['A','D','B'],pts['D']+[-2,-.8],'angle_degrees'),('BDC',60,['B','D','C'],pts['D']+[.1,-2],'angle_degrees'),('ABD','5y',['A','B','D'],pts['B']+[-.1,2],'angle_degrees')]:m.qty(id,value,bind,kind);m.text(str(value)+('°' if kind=='angle_degrees' else ''),at,id)
m.target=dict(kind='solve_variable',variable='y',derived_value=12,auxiliary_solution={'x':5},equations=['5y=60','x+4=3x-6'])
m.check('ADB',angle(pts['A'],pts['D'],pts['B']),55);m.check('BDC',angle(pts['B'],pts['D'],pts['C']),60);m.check('AI',distance(pts['A'],pts['I']),9);m.notes=['A、B、C、D、I 是内部生成的名字，不在重绘中显示。','平行四边形为题目要求成立的条件；不是仅由外观判定的已知事实。']
models.append(m.write())
# 08 chord perpendicular to a radius.
m=Model(8,'圆 A 的弦 CD 垂直于半径 AB，交点 E；半径14、弦长22。',BLUE)
v1=polar(1,-45);v2=polar(1,45);A=v([0,0]);B=14*v1;E=math.sqrt(75)*v1;C=E-11*v2;D=E+11*v2
for k,p in dict(A=A,B=B,C=C,D=D,E=E).items():m.point(k,p,offset={'A':(-17,3),'B':(10,-12),'C':(-6,-16),'D':(14,4),'E':(-15,-3)}[k],dot=k=='A')
m.circle('A',14);m.seg('A','B');m.seg('C','D');m.right('A','E','D',size=2);m.rel('perpendicular',['AB','CD']);m.rel('between',['A','E','B']);m.rel('on_circle',['B','C','D','circle']);m.qty('radius',14,'circle','radius','question');m.qty('CD',22,['C','D'],evidence='question');m.target=dict(kind='length',object=['E','B'],derived_value=round(14-math.sqrt(75),2),exact='14-sqrt(75)')
m.check('CD',distance(C,D),22);m.check('AE perpendicular CD',angle(A,E,D),90);m.check('C on circle',distance(A,C),14);m.notes=['原图顶部有极小裁切残墨，未解释为几何对象。']
models.append(m.write())
# 09 rectangle and external right triangle.
m=Model(9,'矩形 ABCD，AC⊥CE，AE 与 CD 交于 F；求三角形 ACF 面积。')
pts=dict(A=v([0,0]),B=v([0,16]),C=v([12,16]),D=v([12,0]),E=v([24,7]));pts['F']=intersect(pts['A'],pts['E'],pts['C'],pts['D'])
for k,p in pts.items():m.point(k,p,offset={'A':(-12,-5),'B':(-12,5),'C':(9,8),'D':(9,-3),'E':(0,-14),'F':(-12,8)}[k])
m.poly(['A','B','C','D'],'ABCD');m.seg('A','C');m.seg('C','E');m.seg('A','E');m.right('A','C','E',size=1.5,color=BLACK);m.rel('rectangle',['A','B','C','D'],'question');m.rel('perpendicular',['AC','CE'],'image_and_question');m.rel('intersection',['AE','CD','F'],'question')
for id,val,at in [('AB',16,[-1.7,8]),('BC',12,[5.6,14.8]),('CE',15,[19.2,12.7])]:m.qty(id,val,list(id),evidence='image_and_question');m.text(str(val),at,id)
m.target=dict(kind='area',object=['A','C','F'],derived_value=75);m.check('CE',distance(pts['C'],pts['E']),15);m.check('perpendicular',angle(pts['A'],pts['C'],pts['E']),90);m.check('area ACF',area([pts[k].tolist() for k in ['A','C','F']]),75);m.notes=[]
models.append(m.write())
# 10 quarter-disk incircle, named K is the circle, not a shown center.
m=Model(10,'半径6的四分之一圆中内切圆 K，同时与两条半径和大圆弧相切。')
r=6/(1+math.sqrt(2))
for k,p in dict(O=[0,0],X=[6,0],Y=[0,-6],C=[r,-r],T1=[r,0],T2=[0,-r],T3=polar(6,-45)).items():m.point(k,p,label=False)
m.seg('O','X');m.seg('O','Y');m.arc('O',6,-90,0,'outer_quarter');m.circle('C',r,'K');m.text('K',[r+1.1,-r+.9],'K',size=23)
m.rel('perpendicular',['OX','OY'],'image');m.rel('tangent',['K','OX','T1'],'question');m.rel('tangent',['K','OY','T2'],'question');m.rel('internal_tangent',['K','outer_quarter','T3'],'question');m.qty('outer_radius',6,'outer_quarter','radius','question');m.target=dict(kind='radius',object='K',exact='6*(sqrt(2)-1)',derived_value=r)
m.check('internal tangency',distance(m.p['C'],m.p['O'])+r,6);m.notes=['隐含圆心和切点只在数学模型中命名，原图没有的点名、半径线不额外显示。']
models.append(m.write())
# 11 shaded curvilinear region: analytic circular-arc boundary, no raster tracing.
m=Model(11,'三个半径2的上半圆；阴影由上半圆及两个下方四分之一圆弧围成。')
for k,p in dict(E=[0,0],F=[4,0],A=[0,2],B=[4,2],O=[2,2],T=[2,0],L=[-2,0],R=[6,0]).items():m.point(k,p,label=k in 'EFAB',dot=k in 'EFAB',offset={'E':(0,-15),'F':(0,-15),'A':(-12,6),'B':(12,6)}.get(k,(0,0)))
m.arc('E',2,0,180,'left_semicircle');m.arc('F',2,0,180,'right_semicircle');m.arc('O',2,0,180,'top_semicircle');m.seg('L','R')
m.entities.append(dict(id='shaded',kind='curvilinear_region',boundary=[dict(center='O',radius=2,start_deg=180,end_deg=0),dict(center='F',radius=2,start_deg=90,end_deg=180),dict(center='E',radius=2,start_deg=0,end_deg=90)]));m.draw.insert(0,dict(kind='fill_curvilinear',entity='shaded',color='#d0d0d0'))
m.rel('rectangle',['A','B','F','E'],'question_and_image',source_text_order='ABEF');m.rel('centers',[['E','left_semicircle'],['F','right_semicircle']],'question');m.rel('tangent',['left_semicircle','right_semicircle','T'],'image')
for id in ['left_semicircle','right_semicircle','top_semicircle']:m.qty(id+'_radius',2,id,'radius','question','cm')
m.target=dict(kind='area',object='shaded',derived_value=8,unit='cm^2',derivation='rectangle 4*2 + top semicircle 2π - two lower quarter circles 2π')
m.check('AB',distance(m.p['A'],m.p['B']),4);m.notes=['题干写 ABEF 是矩形，但图中字母的边界顺序为 ABFE；保留题干原文，并按图中位置建模。','矩形的 AB、AE、BF 只是计算辅助关系，原图未画出的边不额外画。']
models.append(m.write())
# 12 solve a unit isosceles triangle, then intersect the bisector and a 120° ray from C.
m=Model(12,'AC=BC 的四边形 ABCD，BD 平分 ABC；标注 ACB=20°、BDC=80°。')
B=v([0,0]);C=v([1,0]);A=C+polar(1,160);D=intersect(B,polar(1,40),C,C+polar(1,120));I=intersect(A,C,B,D)
for k,p in dict(A=A,B=B,C=C,D=D,I=I).items():m.point(k,p,label=k!='I',offset={'A':(-12,6),'B':(-8,-12),'C':(12,-5),'D':(0,14),'I':(0,0)}[k])
m.poly(['A','B','C','D'],'ABCD');m.seg('A','C');m.seg('B','D');m.anglemark('A','C','B',.23);m.anglemark('B','D','C',.07);m.text('20°',[.71,.06],'ACB');m.text('80°',D+[.015,-.14],'BDC')
m.rel('equal_length',[['A','C'],['B','C']],'question');m.rel('angle_bisector',['B','D','A','C'],'question');m.rel('intersection',['AC','BD','I'],'image');m.qty('ACB',20,['A','C','B'],'angle_degrees','image_and_question');m.qty('BDC',80,['B','D','C'],'angle_degrees','image_and_question');m.target=dict(kind='angle_degrees',object=['B','A','D'],derived_value=120)
m.check('AC=BC',distance(A,C),distance(B,C));m.check('ACB',angle(A,C,B),20);m.check('BDC',angle(B,D,C),80);m.check('bisector',angle(A,B,D),angle(D,B,C));m.check('BAD',angle(B,A,D),120);m.notes=['未把题干中的等长和角平分条件额外画为原图没有的刻痕。']
models.append(m.write())
# 13 exact affine extensions of an arbitrary triangle of area one.
m=Model(13,'三角形三边向两端各延长一倍，六个端点组成六边形 PQRSTU。')
A=v([.15,10/7]);B=v([.7,0]);C=v([-.7,0]);pts=dict(A=A,B=B,C=C,P=2*A-B,S=2*B-A,Q=2*A-C,T=2*C-A,R=2*B-C,U=2*C-B)
for k,p in pts.items():m.point(k,p,offset={'A':(0,15),'B':(5,13),'C':(-5,13),'P':(-3,14),'Q':(5,14),'R':(14,-4),'S':(4,-14),'T':(-6,-14),'U':(-14,0)}[k])
m.poly(list('PQRSTU'),'hexagon');m.poly(['A','B','C'],'ABC')
for a,b in [('P','A'),('B','S'),('T','C'),('A','Q'),('U','C'),('B','R')]:m.seg(a,b,style='dashed')
for ids in [['P','A','B','S'],['T','C','A','Q'],['U','C','B','R']]:m.rel('collinear_ordered',ids,'question');m.rel('equal_successive_lengths',ids,'question')
m.qty('area_ABC',1,['A','B','C'],'area','question');m.target=dict(kind='area',object='hexagon',derived_value=13);m.check('area ABC',area([pts[k].tolist() for k in 'ABC']),1);m.check('area hexagon',area([pts[k].tolist() for k in 'PQRSTU']),13);m.notes=['三角形形状不唯一；只固定面积1及各条边的延长关系。']
models.append(m.write())
# 14 a genuinely spatial polygon, all 12 cube-edge midpoints; 2D crossings are not vertices.
m=Model(14,'立方体12条棱的中点组成一条闭合空间12边折线；保留透视投影中的交叉。')
verts={f'V{x}{y}{z}':[x,y,z] for x in [0,1] for y in [0,1] for z in [0,1]}
cycle=[[.5,0,1],[1,.5,1],[1,1,.5],[.5,1,1],[0,.5,1],[0,1,.5],[.5,1,0],[0,.5,0],[0,0,.5],[.5,0,0],[1,.5,0],[1,0,.5]]
all3={**verts,**{f'M{i+1}':p for i,p in enumerate(cycle)}}
for k,(x,y,z) in all3.items():m.point(k,[x+.38*y,z+.38*y],label=False)
for k,p in verts.items():
 for axis in range(3):
  if p[axis]==0:
   q=p.copy();q[axis]=1;name='V'+''.join(map(str,q));m.seg(k,name,color='#888888',width=.8)
for i in range(12):
 a=f'M{i+1}';b=f'M{(i+1)%12+1}';m.seg(a,b,width=2)
 m.draw.append(dict(kind='dot',point=a,color=BLACK,hollow=True))
 support=[]
 for k,p in verts.items():
  for l,q in verts.items():
   if k<l and sum(abs(a-b) for a,b in zip(p,q))==1 and np.allclose((v(p)+v(q))/2,v(cycle[i])):support=[k,l]
 m.rel('midpoint',[a,*support],'question_and_image')
angles=[]
for i,b in enumerate(cycle):
 a=v(cycle[i-1])-v(b);c=v(cycle[(i+1)%12])-v(b);angles.append(math.degrees(math.acos(np.dot(a,c)/np.linalg.norm(a)/np.linalg.norm(c))))
m.extra=dict(spatial_model=dict(dimension=3,points=all3,cube_edge_length=1,polygon_cycle=[f'M{i+1}' for i in range(12)],projection=dict(type='oblique_parallel',matrix=[[1,.38,0],[0,.38,1]]),interior_angles_degrees=angles,topology_note='2D edge crossings are projection crossings; no vertex unless explicitly listed in polygon_cycle'))
m.target=dict(kind='sum_spatial_vertex_angles',derived_value=sum(angles),unit='degree');m.check('angle sum',sum(angles),1080);m.guarantee='exact_3d_realization_with_oblique_2d_projection';m.notes=['正方体画法为斜投影，不用图中二维折角求空间角。','原图未对背面折线做虚线遮挡，重绘同样全部实线。','M1…M12 为内部命名，图中只显示空心中点标记。']
models.append(m.write())
# 15 two triangles sharing an 18-inch diagonal, with distinct altitudes 6 and 4.
m=Model(15,'绿色四边形及18英寸对角线；对21英寸底边的高为6，对该对角线的另一顶点高为4。',color='#688b4b')
A=v([0,0]);D=v([21,0]);B=v([21-math.sqrt(18**2-6**2),6]);H=v([B[0],0]);N=B+.72*(D-B);C=N+4*v([6,math.sqrt(288)])/18
for k,p in dict(A=A,B=B,C=C,D=D,H=H,N=N).items():m.point(k,p,label=False)
m.poly(['A','B','C','D'],'quad',fill='#b4d497');m.seg('B','D',style='dashed',color=PINK);m.seg('B','H',style='dashed',color=PINK);m.seg('C','N',style='dashed',color=PINK);m.right('B','H','D',size=.8);m.right('C','N','D',size=.8)
m.rel('perpendicular',['BH','AD']);m.rel('perpendicular',['CN','BD']);m.rel('between',['A','H','D']);m.rel('between',['B','N','D'])
for id,val,bind,at in [('AD',21,['A','D'],[10.5,-1.1]),('BH',6,['B','H'],(B+H)/2+[1.3,0]),('BD',18,['B','D'],(B+D)/2+[0,-1]),('CN',4,['C','N'],(C+N)/2+[-1.5,.2])]:m.qty(id,val,bind,unit='inch');m.text(f'{val} in.',at,id);m.check(id,distance(m.p[bind[0]],m.p[bind[1]]),val)
m.target=dict(kind='unspecified',reason='没有题干，也没有问号；不能确定要求面积还是其他量',available_derived_quantities={'quadrilateral_area_inch2':99});m.notes=['所有字母是内部命名。','C 的高足 N 沿对角线位置无法唯一确定，选择参数0.72保留图形外观。']
models.append(m.write())
# 16 arc labels versus angles at an inscribed/tangent vertex.
m=Model(16,'圆上一条竖直直径、三条弦与右下切线；a,b,c 为角，d,e 为弧度数。')
pts=dict(O=v([0,0]),T=polar(1,90),B=polar(1,270),L=polar(1,210),R=polar(1,326));pts['X']=pts['R']+1.0*polar(1,56);pts['I']=intersect(pts['T'],pts['B'],pts['L'],pts['R'])
for k,p in pts.items():m.point(k,p,label=False,dot=k=='O')
m.circle('O',1,color='#009ed0')
for a,b in [('T','B'),('T','R'),('B','R'),('L','R')]:m.seg(a,b)
m.seg('R','X',kind='ray',end=0)
for name,a,b in [('arc_TL',90,210),('arc_LB',210,270),('arc_BR',270,326),('arc_RT',326,450)]:m.arc('O',1,a,b,name,draw=False)
m.rel('diameter',['T','B','circle'],'image');m.rel('tangent',['RX','circle','R'],'visual_hypothesis',confidence='high',reason='射线在圆周处接触且与半径垂直的常规画法；无题干明确声明');m.rel('intersection',['TB','LR','I'])
for id,value,bind,kind in [('TL',120,'arc_TL','arc_degrees'),('BR',56,'arc_BR','arc_degrees'),('a','a',['L','R','B'],'angle_degrees'),('b','b',['T','R','L'],'angle_degrees'),('c','c',['T','R','X'],'angle_degrees'),('d','d','arc_RT','arc_degrees'),('e','e','arc_LB','arc_degrees')]:m.qty(id,value,bind,kind)
for text,at,bind in [('120°',[-1.13,.67],'TL'),('56°',[.58,-1.12],'BR'),('d°',[.94,.75],'d'),('e°',[-.5,-1.13],'e'),('a°',[.16,-.82],'a'),('b°',[.41,-.25],'b'),('c°',[.88,-.02],'c')]:m.text(text,at,bind,color=ORANGE,size=19)
m.anglemark('T','R','X',r=.53,color=ORANGE);m.draw.append(dict(kind='leader',start=[.35,-.77],end=[.68,-.61],color=ORANGE))
m.target=dict(kind='inferred_symbol_values',variables=dict(a=30,b=60,c=62,d=124,e=60),explicit_question=False,conditional='c assumes ray RX tangent; all others follow circle incidence and marked arcs')
m.check('a',angle(pts['L'],pts['R'],pts['B']),30);m.check('b',angle(pts['T'],pts['R'],pts['L']),60);m.check('c',angle(pts['T'],pts['R'],pts['X']),62);m.notes=['精确使用120°及56°后，LR 弦略微倾斜；原图画成近水平，不能再把水平作为硬约束。','切线为高置信视觉解释，题干为空；保留这一假设的来源。']
models.append(m.write())
# 17 exact segment lengths force a different relative slope than the schematic source.
m=Model(17,'三条平行直线与两条截线；左截线上相邻段等长，右截线上绑定 y 表达式。',BLUE)
h=1.8;dx=math.sqrt(4-h*h);dy=math.sqrt(2.25**2-h*h)
pts=dict(A=[0,h],B=[dx,0],C=[2*dx,-h],D=[5,h],E=[5+dy,0],F=[5+2*dy,-h])
for k,p in pts.items():m.point(k,p,offset=(10,12) if k in 'ABC' else (-12,12))
for a,b in [('A','D'),('B','E'),('C','F')]:m.seg(a,b,kind='line',end=.18);m.draw.append(dict(kind='parallel_mark',points=[a,b],color=PINK))
m.seg('A','C',kind='line',end=.28);m.seg('D','F',kind='line',end=.28);m.ticks('A','B');m.ticks('B','C');m.rel('parallel',['AD','BE','CF']);m.rel('equal_length',[['A','B'],['B','C']]);m.rel('collinear_ordered',['A','B','C']);m.rel('collinear_ordered',['D','E','F'])
for id,val,at in [('AB','3x-4',[-1,.9]),('BC','6-2x',[-.15,-.9]),('DE','(5/3)y+1',[7.8,1]),('EF','3y',[8.1,-.9])]:m.qty(id,val,list(id));m.text(val,at,id,size=15)
m.target=dict(kind='inferred_symbol_values',variables=dict(x=2,y=.75),explicit_question=False,equations=['3x-4=6-2x','(5/3)y+1=3y']);m.check('AB',distance(pts['A'],pts['B']),2);m.check('BC',distance(pts['B'],pts['C']),2);m.check('DE',distance(pts['D'],pts['E']),2.25);m.check('EF',distance(pts['E'],pts['F']),2.25);m.notes=['原图右截线比左截线更接近竖直，但数值要求右侧相邻段更长；精确重绘因此增大右截线倾斜度。','三个粉色单箭头为平行标记，蓝色端箭头表示直线无限延伸。']
models.append(m.write())
# 18 source 21.5 is compatible with rounding, not exact equality.
m=Model(18,'窄三角形的底边5英寸，外部高8英寸；高足距底边左端15英寸。',color='#d9972d');m.guarantee='exact_geometry_with_rounded_source_length'
pts=dict(A=[0,8],H=[0,0],B=[15,0],C=[20,0])
for k,p in pts.items():m.point(k,p,label=False)
m.poly(['A','B','C'],'triangle',fill='#ffeaba');m.seg('A','H',style='dashed',color=PINK);m.seg('H','B',style='dashed',color=PINK);m.right('A','H','B',size=1);m.bracket('B','C','5 in.',-1.2)
for id,val,bind,at in [('AH',8,['A','H'],[-1.7,4]),('HB',15,['H','B'],[7.5,-1]),('AC',21.5,['A','C'],[11,5.3])]:m.qty(id,val,bind,unit='inch');m.text(f'{val} in.',at,id)
m.qty('BC',5,['B','C'],unit='inch');m.rel('collinear_ordered',['H','B','C']);m.rel('perpendicular',['AH','HC']);m.target=dict(kind='unspecified',reason='题干为空；长度均给定，没有显式求解目标',available_derived_quantities={'triangle_area_inch2':20})
m.check('AC rounded to tenth',round(distance(pts['A'],pts['C']),1),21.5);m.notes=['若把21.5视为绝对精确，与8、15、5不相容：sqrt(8²+20²)=21.540659…。本模型明确采用四舍五入到0.1英寸的解释。','15英寸绑定高足到三角形左底点的外部距离，不能绑定整条底边。']
models.append(m.write())
# 19 rhombus side16, altitude15 to the slanted right edge.
m=Model(19,'四边等长的蓝色菱形，边长16；从左下顶点到右边的垂线段长15。')
A=v([0,0]);B=v([16,0]);C=v([16-math.sqrt(31),15]);D=v([-math.sqrt(31),15]);H=B+(C-B)*(math.sqrt(31)/16)
for k,p in dict(A=A,B=B,C=C,D=D,H=H).items():m.point(k,p,label=False)
m.poly(['A','B','C','D'],'rhombus',fill='#acd3eb');m.seg('A','H',width=1.2);m.right('A','H','B',size=1.5,color='#ff233f')
for a,b in [('A','B'),('B','C'),('C','D'),('D','A')]:m.ticks(a,b,color='#ff233f')
m.bracket('B','C','16',-2.8);m.text('15',(A+H)/2+[0,1.2],'AH');m.rel('equal_length',[['A','B'],['B','C'],['C','D'],['D','A']]);m.rel('perpendicular',['AH','BC']);m.rel('between',['B','H','C']);m.qty('BC',16,['B','C']);m.qty('AH',15,['A','H']);m.target=dict(kind='unspecified',reason='无题干，不假定题目一定求面积',available_derived_quantities={'area':240})
m.check('AH',distance(A,H),15);m.check('BC',distance(B,C),16);m.check('right angle',angle(A,H,B),90);m.notes=['原图的15绑定斜置高，不是对角线，也不是两顶点间的长度。']
models.append(m.write())
# 20 distinguish x arc from y angle; exact tangents and secant.
m=Model(20,'圆外切线与割线夹角50°，两切点间左侧弧160°；x 是右上弧，y 是弦切角。')
O=v([0,0]);A=polar(1,70);B=polar(1,230);C=polar(1,10);X=intersect(A,A+polar(1,160),B,C);U=A+polar(.8,160);V=B+polar(.8,140);W=B+polar(.8,320)
for k,p in dict(O=O,A=A,B=B,C=C,X=X,U=U,V=V,W=W).items():m.point(k,p,label=False)
m.circle('O',1,color='#009ed0');m.seg('X','U',kind='ray',end=0);m.seg('V','W',name='tangent_B',kind='line',end=0);m.seg('B','X');m.arc('O',1,70,230,'arc_AB',draw=False);m.arc('O',1,10,70,'arc_CA',draw=False);m.arc('O',1,230,370,'arc_BC',draw=False)
m.rel('tangent',['XU','circle','A'],'visual_hypothesis',confidence='high');m.rel('tangent',['tangent_B','circle','B'],'visual_hypothesis',confidence='high');m.rel('collinear_ordered',['B','C','X']);m.qty('arc_AB',160,'arc_AB','arc_degrees');m.qty('AXC',50,['A','X','C'],'angle_degrees');m.qty('x','x','arc_CA','arc_degrees');m.qty('y','y',['C','B','W'],'angle_degrees')
for text,at,bind in [('160°',[-1.22,.63],'arc_AB'),('x°',[.63,.41],'x'),('50°',[1.35,-.13],'AXC'),('y°',[-.14,-.8],'y')]:m.text(text,at,bind,color=ORANGE,size=19)
m.anglemark('C','B','W',r=.35,color=ORANGE);m.draw.append(dict(kind='leader',start=[1.16,-.08],end=(X+[-.2,-.04]).tolist(),color=ORANGE,curved=True))
m.target=dict(kind='inferred_symbol_values',variables=dict(x=60,y=70),explicit_question=False,equations=['50=(160-x)/2','y=(360-160-x)/2'],conditional='Both visually touching straight strokes are treated as tangents')
m.check('AXC',angle(A,X,C),50);m.check('CBW',angle(C,B,W),70);m.check('radius tangent at A',angle(O,A,X),90);m.notes=['x°位于圆内但没有顶点或角弧，按相邻圆弧度数解释。','无题干明确保证相切；两条切线均为高置信视觉假设。','50°的弯箭头是标注引线，不是新的几何射线。']
models.append(m.write())
(RUN/'model_index.json').write_text(json.dumps([dict(index=m['index'],id=m['sample_id'],summary=m['summary'],embedding=m['representation']['embedding_guarantee']) for m in models],ensure_ascii=False,indent=2))
for m in models:
 failed=[c for c in m['coordinate_checks'] if not c['passed']]
 print(m['index'], m['sample_id'],m['representation']['embedding_guarantee'], 'CHECKS',len(m['coordinate_checks']),'FAILED',failed)
