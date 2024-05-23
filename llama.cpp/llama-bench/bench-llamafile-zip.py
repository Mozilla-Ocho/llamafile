#!/usr/bin/env python

import sys

def split(line):
  return [x.strip() for x in line[1:-1].split('|')]

with open('before') as f:
  lines = [split(x) for x in f.read().strip().split('\n') if x.startswith('|')]
  headers = lines[0]
  bars = lines[1]
  before = lines[2:]

with open('after') as f:
  lines = [split(x) for x in f.read().strip().split('\n') if x.startswith('|')]
  after = lines[2:]

res = []

def print_row(A):
  res.append('| ')
  for i, a in enumerate(A):
    if i:
      res.append(' | ')
    res.append(str(a))
  res.append(' |\n')

last_name = headers[-1]
headers = headers[:-1]
headers.append(last_name + ' before')
headers.append(last_name + ' after')
headers.append(last_name + ' speedup')
print_row(headers)
print_row(["---:" for h in headers])

for B, A in zip(before, after):
  print_row(B[:] + [A[-1], ('%.2f' % (float(A[-1]) / float(B[-1]))) + 'x'])

print(''.join(res))
