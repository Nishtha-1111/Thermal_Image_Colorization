import sys
sys.path.insert(0, 'src')

print('='*55)
print('CLAIM 1 -- Edge-guided generator input')
print('='*55)

from dataset import LLVIPDataset
ds = LLVIPDataset('LLVIP/infrared/train','LLVIP/visible/train',image_size=(256,256))
s  = ds[0]
print('  ir shape   :', s['ir'].shape,   'dtype=', s['ir'].dtype)
print('  edge shape :', s['edge'].shape, 'dtype=', s['edge'].dtype)
print('  edge range : [{:.3f}, {:.3f}]'.format(s['edge'].min().item(), s['edge'].max().item()))

import torch
from models import UNetGenerator
netG = UNetGenerator(in_channels=2, out_channels=3, ngf=64)
first_conv = netG.d1[0]
print('  Generator first Conv2d: in_channels={}, out_channels={}'.format(
      first_conv.in_channels, first_conv.out_channels))

ir   = s['ir'].unsqueeze(0)
edge = s['edge'].unsqueeze(0)
inp  = torch.cat([ir, edge], dim=1)
out  = netG(inp)
print('  torch.cat([ir,edge]) shape :', inp.shape)
print('  netG output shape          :', out.shape)
print()

print('='*55)
print('CLAIM 2 -- Sobel-gradient loss')
print('='*55)
from models import EdgePreservingLoss
epl = EdgePreservingLoss()
print('  kx kernel shape :', epl.kx.shape)
print('  kx kernel values:')
print(epl.kx[0,0])
print('  ky kernel values:')
print(epl.ky[0,0])

fake = torch.randn(2,3,256,256)
real = torch.randn(2,3,256,256)
fake.requires_grad_(True)
loss = epl(fake, real)
loss.backward()
print('  Loss value     :', round(loss.item(), 6))
print('  Gradient flows :', fake.grad is not None)
print()

print('='*55)
print('CLAIM 3 -- VOC annotation reading (parse_voc_annotation CALLED)')
print('='*55)
from dataset import parse_voc_annotation
import glob, os
all_xml = glob.glob('LLVIP/Annotations/*.xml')
print('  Total XML annotation files:', len(all_xml))
for xf in sorted(all_xml)[:5]:
    boxes = parse_voc_annotation(xf)
    print('  {} -> {} box(es): {}'.format(os.path.basename(xf), len(boxes), boxes[:2]))
