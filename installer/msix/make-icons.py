# Regenerates installer/msix/Assets: python3 installer/msix/make-icons.py installer/msix/Assets
# Needs Pillow (a local tool, not a build dependency: the PNGs are committed).
# Draws the EuLLM app icon at 1024 px and writes the MSIX asset sizes.
import math, sys, os
from PIL import Image, ImageDraw
NAVY=(26,35,50,255); YELLOW=(255,204,0,255); BLUE=(77,143,204,255); WHITE=(255,255,255,255)
N=1024
def star(d,cx,cy,r,col):
    pts=[]
    for i in range(10):
        a=-math.pi/2+i*math.pi/5; rr=r if i%2==0 else r*0.42
        pts.append((cx+rr*math.cos(a),cy+rr*math.sin(a)))
    d.polygon(pts,fill=col)
def icon(bg=True):
    im=Image.new('RGBA',(N,N),(0,0,0,0)); d=ImageDraw.Draw(im)
    if bg: d.rectangle((0,0,N-1,N-1),fill=NAVY)
    c=N/2
    for i in range(12):
        a=-math.pi/2+i*2*math.pi/12
        star(d,c+360*math.cos(a),c+360*math.sin(a),62,YELLOW)
    hexr=210
    hexpts=[(c+hexr*math.cos(math.pi/6+k*math.pi/3),c+hexr*math.sin(math.pi/6+k*math.pi/3)) for k in range(6)]
    d.line(hexpts+[hexpts[0]],fill=WHITE if bg else BLUE,width=34,joint='curve')
    nodes=[(c+130*math.cos(-math.pi/2+k*math.pi/3),c+130*math.sin(-math.pi/2+k*math.pi/3)) for k in range(6)]
    for x,y in nodes:
        d.line((c,c,x,y),fill=BLUE,width=26)
    for x,y in nodes:
        d.ellipse((x-34,y-34,x+34,y+34),fill=BLUE)
    d.ellipse((c-58,c-58,c+58,c+58),fill=YELLOW)
    return im
out=sys.argv[1]; os.makedirs(out,exist_ok=True)
big=icon()
def save(img,name,size): img.resize(size,Image.LANCZOS).save(os.path.join(out,name),optimize=True)
for s in (44,150): save(big,f'Square{s}x{s}Logo.png',(s,s))
save(big,'StoreLogo.png',(50,50))
# Taskbar / terminal size without the plate, as Windows expects for unplated targets.
for s in (16,24,32,48,256):
    save(icon(bg=False),f'Square44x44Logo.targetsize-{s}_altform-unplated.png',(s,s))
wide=Image.new('RGBA',(310,150),NAVY); small=big.resize((130,130),Image.LANCZOS); wide.paste(small,(90,10),small)
wide.save(os.path.join(out,'Wide310x150Logo.png'),optimize=True)

