# GeoSym Relation Extraction Prompt v0

输出 JSON，不要输出解释文字。

字段：

```json
{
  "objects": ["Point A", "Segment AB"],
  "relations_text": [
    "AB is perpendicular to BC",
    "AB has length 3",
    "Find AC"
  ],
  "uncertain_relations_text": []
}
```

优先抽取：

- 点、线段、圆、弧、阴影区域。
- 长度、角度、面积、比例标签。
- 垂直、平行、等长、共线、圆上点、切线。
- 题目要求的 target。

不要把像素坐标当作数学坐标。像素位置只能作为 debug metadata。
