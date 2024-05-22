#!/usr/bin/env python

import os
import re
import sys

fils = sys.argv[1:]
hdrs = set()
sysi = set()

for path in fils:
  with open(path) as f:
    for line in f:
      line = line.rstrip('\n')
      m = re.search(r'# *include  *"([^"]*)"', line)
      if m:
        hdr = os.path.basename(m.group(1))
        hdrs.add(hdr)
      m = re.search(r'# *include  *<([^>]*)>', line)
      if m:
        hdr = m.group(1)
        if ('cu' not in hdr and
            'hip' not in hdr and
            not os.path.exists(os.path.join('/usr/local/cuda/include', hdr))):
          sysi.add(hdr)

srcs = [f for f in fils if f not in hdrs]
done = set()

def visit(path):
  print()
  print('////////////////////////////////////////////////////////////////////////////////')
  print('//')
  print('// ROLLUP %s' % (path))
  print('//')
  print('////////////////////////////////////////////////////////////////////////////////')
  print()
  with open(path) as f:
    for line in f:
      line = line.rstrip('\n')
      m = re.search(r'# *include  *"([^"]*)"', line)
      if m:
        hdr = os.path.basename(m.group(1))
        if hdr in fils:
          if hdr not in done:
            done.add(hdr)
            visit(hdr)
          continue
      m = re.search(r'# *include  *<([^>]*)>', line)
      if m:
        hdr = m.group(1)
        if hdr in sysi:
          continue
      if '#pragma once' in line:
        continue
      print(line)

for path in sorted(sysi):
  print('#include <%s>' % (path))
print()
for path in srcs:
  visit(path)
