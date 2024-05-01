import re

with open('llama.cpp/ggml-quants.h') as f:
  prototypes = f.read()
prototypes = [line.replace(';', '') for line in prototypes.split('\n')
              if line.endswith(';') and not line.startswith('//')]
FUNCS = [(re.search(r'(?<= )\w+', proto).group(0), proto)
         for proto in prototypes]

# BEGIN SPECIAL FUNCTIONS
FUNCS.append(('ggml_validate_row_data', 'bool ggml_validate_row_data(enum ggml_type type, const void * data, size_t nbytes)'))
# END SPECIAL FUNCTIONS

ARCHS = (
  ('amd_avx512', '__x86_64__', ('FMA', 'F16C', 'AVX2', 'AVX512F')),
  ('amd_avx2', '__x86_64__', ('FMA', 'F16C', 'AVX2')),
  ('amd_avx', '__x86_64__', ()),
  ('arm80', '__aarch64__', ()),
)

for arch, mac, needs in ARCHS:
  path = 'llama.cpp/ggml-quants-%s.c' % (arch.replace('_', '-'))
  with open(path, 'w') as f:
    f.write('#ifdef %s\n' % (mac))
    for func, proto in FUNCS:
      f.write('#define %s %s_%s\n' % (func, func, arch))
    f.write('#include "ggml-quants.inc"\n')
    f.write('#endif // %s\n' % (mac))

with open('llama.cpp/ggml-quants.cpp', 'w') as f:
  f.write('#include <cosmo.h>\n')
  f.write('#include <sys/auxv.h>\n')
  f.write('#include <libc/sysv/consts/hwcap.h>\n')
  f.write('#include "ggml-quants.h"\n')
  f.write('\n')
  for func, proto in FUNCS:
    for arch, mac, needs in ARCHS:
      f.write('extern "C" %s;\n' % (proto.replace(func, '%s_%s' % (func, arch))))
    f.write('\n')
  f.write('static const struct QuantFuncs {\n')
  for func, proto in FUNCS:
    f.write('    typeof(%s) *ptr_%s;\n' % (func, func))
  f.write('\n')
  f.write('    QuantFuncs() {\n')
  f.write('#ifdef __x86_64__\n')
  for arch, mac, needs in ARCHS:
    if mac == '__x86_64__':
      f.write('        if (%s) {\n' % (' && '.join('X86_HAVE(%s)' % (need) for need in needs) or '1'))
      for func, proto in FUNCS:
        f.write('            ptr_%s = %s_%s;\n' % (func, func, arch))
      f.write('            return;\n')
      f.write('        }\n')
  f.write('#else\n')
  for func, proto in FUNCS:
    f.write('        ptr_%s = %s_arm80;\n' % (func, func))
  f.write('#endif\n')
  f.write('    }\n')
  f.write('} funcs;\n')
  f.write('\n')
  for func, proto in FUNCS:
    proto = proto.replace(';', '')
    f.write(proto + ' {\n')
    if 'imatrix' in proto:
      args = 'src, dst, nrows, n_per_row, imatrix'
    elif 'quantize' in proto:
      args = 'x, y, k'
    elif 'vec_dot' in proto:
      args = 'n, s, bs, vx, bx, vy, by, nrc'
    elif 'grid_size' in proto:
      args = 'grid_size'
    elif 'validate' in proto:
      args = 'type, data, nbytes'
    else:
      args = 'type'
    f.write('  return funcs.ptr_%s(%s);\n' % (func, args))
    f.write('}\n')
    f.write('\n')
