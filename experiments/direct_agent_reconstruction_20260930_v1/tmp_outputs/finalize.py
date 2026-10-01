from pathlib import Path
import json,hashlib
root=Path('/data/Geosym127K/Geosym-World/artifacts/direct_agent_reconstruction/agent_visual_20260930_v1')
models=sorted(root.glob('*/model.json'))
assert len(models)==20
rows=[]
status={
1:('基本重建成功','平行四边形、交点 H、相似三角形和四个长度均重现；A/D 的共线位置由外形选择。'),
2:('基本重建成功','角平分线结构和 11.2、16、7、x 的比例关系重现。'),
3:('基本重建成功','等腰三角形、K 垂足、12 的底边标注和 M 射线重现。'),
4:('基本重建成功','三条中线、重心 P、中点刻痕和 RP/JP 数据重现。'),
5:('仅相似外形','全等对应和全部表达式保留；独立余弦核查表明同一对角线长度矛盾，不能作精确欧氏重建。'),
6:('基本重建成功','直径 AB、圆、8-15-17 三角形重现。'),
7:('基本重建成功','平行四边形及对角线、55°、60°、5y 与长度表达式重现；内部点名为辅助命名。'),
8:('基本重建成功','圆心 A、半径 AB、弦 CD、E 垂足和 14/22 数据重现。'),
9:('基本重建成功','矩形、外接直角三角形、AE∩CD=F 和 16/12/15 重现。'),
10:('基本重建成功','四分之一圆、半径6和三重相切内圆重现。'),
11:('基本重建成功','三个半径2半圆、E/F 圆心和灰色阴影边界重现。'),
12:('基本重建成功','四边形、两条对角线、20°/80° 标注和 BD 角平分结构重现。'),
13:('基本重建成功','三角形面积1及六个外延点的共线/等长关系重现；输出六边形面积13。'),
14:('基本重建成功','12 条立方体棱中点的空间闭合折线重现；明确区分投影交叉和空间顶点。'),
15:('仅相似外形','绿色四边形、两条虚线高、21/18/6/4 标注重现；C 的垂足位置原图没有足够信息唯一确定。'),
16:('条件性重建','圆、直径、弦、120°/56°弧和 a–e 标注重现；两条黑色射线按视觉解释为切线。'),
17:('基本重建成功','三组平行线、两截线、等长刻痕和表达式重现；坐标按数值约束略改了原图斜率。'),
18:('仅相似外形','三角形、8/15/5/21.5 标注和高线重现；21.5 只能视为 21.540659… 的一位小数近似。'),
19:('基本重建成功','菱形填充、四边等长刻痕、15 的垂线和 16 边长重现。'),
20:('条件性重建','圆、两条切线、割线夹角50°、160°弧及 x/y 标注重现；相切关系来自图像而非题干。')}
for p in models:
 d=json.loads(p.read_text()); assert (p.parent/'redraw.png').exists(); assert (p.parent/'original.png').exists(); assert (p.parent/'comparison.png').exists()
 s,r=status[d['index']]; rows.append(dict(index=d['index'],sample_id=d['sample_id'],status=s,observation=r,model=str(p.relative_to(root)),redraw=str((p.parent/'redraw.png').relative_to(root)),comparison=str((p.parent/'comparison.png').relative_to(root))))
(root/'quality_review.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2))
(root/'README.md').write_text('''# Direct agent reconstruction — 20260930 v1\n\n本目录包含 20 张题图的手工数学模型、模型驱动重绘、原图/重绘逐图对照和总览。模型只根据输入 JSON 的 `parser_input.image` 与 `parser_input.question` 建立；本次实验没有读取正式 Parser、formal state、ground truth、旧 redraw 或历史解析结果。\n\n- `01_*/model.json`：逐图数学模型，含坐标、对象、关系、题面数量、目标、检查和不确定性。\n- `01_*/redraw.png`：由 `render_instructions` 绘制。\n- `01_*/comparison.png`：原图与重绘并排。\n- `contact.png`：20 张并排总览（两列，每行一张样本）。\n- `quality_review.json`：逐图人工质量判断。\n- `model_index.json`、`inputs.json`：本次样本索引与输入图像哈希。\n\n`render_models.py` 和 `build_models.py` 位于工作区 `/data/Geosym127K/tmp/direct_agent_20260930_v1/`，仅为一次性实验脚本。\n''')
print('models',len(rows),'redraws',len(list(root.glob('*/redraw.png'))),'comparisons',len(list(root.glob('*/comparison.png'))),'contact_bytes',(root/'contact.png').stat().st_size)
print('sha256_contact',hashlib.sha256((root/'contact.png').read_bytes()).hexdigest())
