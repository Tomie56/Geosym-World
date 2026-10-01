# 几何题图直接重建实验（2026-09-30）

实验产物：[contact.png](/data/Geosym127K/Geosym-World/artifacts/direct_agent_reconstruction/agent_visual_20260930_v1/contact.png)。逐图模型、重绘和原图/重绘对照位于同一目录；[quality_review.json](/data/Geosym127K/Geosym-World/artifacts/direct_agent_reconstruction/agent_visual_20260930_v1/quality_review.json) 是逐图判断。

共处理20张。15张（1–4、6–14、17、19）基本重建了主要几何对象和题面关系；3张（5、15、18）主要是相似外形或带有无法唯一确定的几何自由度；2张（16、20）是条件性重建，切线关系来自图像解释而非题干。第14张按三维立方体棱中点建模，二维交叉不当作新增顶点。

第5张的两个四边形条件不能同时实现：将角和边代入余弦定理后，同一条对角线得到不同长度。因此模型保留全部题面约束并标记为 `schematic_only_source_constraints_inconsistent`。第18张的 `21.5 in.` 与 8、15、5 对应的精确值为约 `21.540659 in.`，重绘明确记录为一位小数近似。

可见遗漏和限制包括：无字母点由模型内部命名；没有题干的样本无法确定求解目标；原图不按比例时，重绘优先满足数值约束而改变外形斜率；第16、20张的切线解释仍有视觉假设。意外问题是题干、角度/长度标注和图形外观有时彼此不相容，不能靠单一“按图测量”策略解决。
