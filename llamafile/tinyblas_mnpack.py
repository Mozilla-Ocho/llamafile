# Copyright 2024 Mozilla Foundation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# tinyBLAS
MAX_M = 5
MAX_N = 5
EDGE_M = 2
EDGE_N = 2
OVERHEAD = 1

# # tinyBLAS_Q0
# MAX_M = 3
# MAX_N = 3
# EDGE_M = 2
# EDGE_N = 2
# OVERHEAD = 8

def doit(VECTOR_REGISTERS, PRECISE):
  # choose tile size that exploits all vector registers

  specified = {}
  precise = set()

  for mc in range(1, MAX_M + 1):
    for nc in range(1, MAX_N + 1):
      accumulators = mc * nc
      memory_loads = min(mc, nc)

      if (mc > EDGE_M or nc > EDGE_N) and (mc == 1 or nc == 1):
        continue

      # # always use precise if there's enough registers
      # v = accumulators * 2 + memory_loads + OVERHEAD
      # if v <= VECTOR_REGISTERS:
      #   if mc % 8 == 0:
      #     v += 2
      #   if mc % 4 == 0:
      #     v += 1
      #   specified[mc, nc] = v
      #   if PRECISE or (mc == 1 or nc == 1):
      #     precise.add((mc, nc))
      # elif not PRECISE:
      #   v = accumulators + memory_loads + OVERHEAD
      #   if v <= VECTOR_REGISTERS:
      #     if mc % 8 == 0:
      #       v += 2
      #     if mc % 4 == 0:
      #       v += 1
      #     specified[mc, nc] = v

      if PRECISE:
        v = accumulators * 2 + memory_loads + OVERHEAD
        if v <= VECTOR_REGISTERS:
          if mc % 8 == 0:
            v += 2
          if mc % 4 == 0:
            v += 1
          specified[mc, nc] = v
          precise.add((mc, nc))
      else:
        v = accumulators + memory_loads + OVERHEAD
        if v <= VECTOR_REGISTERS:
          if mc % 8 == 0:
            v += 2
          if mc % 4 == 0:
            v += 1
          specified[mc, nc] = v

  # generate code for handling biggest tile (e.g. 5x5)
  # generate code for handling edge tiles (i.e. <=2x2)
  # avoid long compile times to generate tiles between
  (best_mc, best_nc), best_v = list(sorted(specified.items(), key=lambda s: s[1]))[-1]
  for (mc, nc), v in list(specified.items()):
    if v < best_v and (mc > EDGE_M or nc > EDGE_N):
      del specified[mc, nc]

  print("switch ((MIN(m - m0, %d) << 4) | MIN(n - n0, %d)) {" % (best_mc, best_nc))

  a = []
  for (mc, nc), v in specified.items():
    s = ""
    s += "case 0x%x%x:\n" % (mc, nc)
    s += "    mc = %d;\n" % (mc)
    s += "    nc = %d;\n" % (nc)
    s += "    gemm<%d, %d, %s>(m0, m, n0, n);\n" % (mc, nc, "true" if (mc, nc) in precise else "false")
    s += "    break;"
    a.append((v, mc, nc, s, []))

  a = list(reversed(sorted(a)))

  for mc in range(1, best_mc + 1):
    for nc in range(1, best_nc + 1):
      if (mc, nc) in specified:
        continue
      for v_, mc_, nc_, s_, extra in a:
        if mc_ <= mc and nc_ <= nc:
          extra.append("case 0x%x%x:" % (mc, nc))
          break

  for v, mc, nc, s, extra in a:
    for e in list(reversed(sorted(extra))):
      print(e)
    print(s)

  print("default:")
  print("    return;")
  print("}")

for VECTOR_REGISTERS in (32, 16):
  print()
  print("#if VECTOR_REGISTERS == %d" % (VECTOR_REGISTERS))
  print("if (!FLAG_precise) {")
  doit(VECTOR_REGISTERS, False)
  print("} else {")
  doit(VECTOR_REGISTERS, True)
  print("}")
  print("#endif")
