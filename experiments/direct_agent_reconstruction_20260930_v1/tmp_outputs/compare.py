from pathlib import Path
import json,shutil
from PIL import Image,ImageDraw,ImageFont
RUN=Path('/data/Geosym127K/Geosym-World/artifacts/direct_agent_reconstruction/agent_visual_20260930_v1')
TMP=Path('/data/Geosym127K/tmp/direct_agent_20260930_v1')
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',17)
pairs=[]
for mp in sorted(RUN.glob('*/model.json')):
 m=json.loads(mp.read_text());out=mp.parent
 im=Image.open(m['source']['image']).convert('RGB');im.save(out/'original.png')
 pair=Image.new('RGB',(1200,500),'white');d=ImageDraw.Draw(pair);d.text((12,8),f"{m['index']:02d}  {m['sample_id']}",font=font,fill='#222222');d.text((12,36),'ORIGINAL',font=font,fill='#555555');d.text((612,36),'MODEL REDRAW',font=font,fill='#555555');d.line((600,60,600,490),fill='#dddddd')
 for j,name in enumerate(['original.png','redraw.png']):
  pic=Image.open(out/name).convert('RGB');pic.thumbnail((570,420));pair.paste(pic,(j*600+(600-pic.width)//2,65+(420-pic.height)//2))
 pair.save(out/'comparison.png');pairs.append(pair)
contact=Image.new('RGB',(2400,5000),'#dddddd')
for i,pair in enumerate(pairs):contact.paste(pair,((i%2)*1200,(i//2)*500))
contact.save(RUN/'contact.png')
for k in range(5):
 page=Image.new('RGB',(1200,2000),'white')
 for j in range(4):page.paste(pairs[k*4+j],(0,j*500))
 page.save(TMP/f'review_{k+1}.png')
print('contact',contact.size)
