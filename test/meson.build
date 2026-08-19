
spiral_byhand_bin = executable('spiral_byhand', 
  files('spiral_byhand.cpp'),
  dependencies: g_deps,
  link_with: mc_lib,
  include_directories: CMC_inc,
  install: false
  )



bench_lifted_bin = executable('bench_lifted',
  files('bench_lifted.cpp'),
  dependencies: g_deps,
  link_with: mc_lib,
  include_directories: CMC_inc,
  install: false
  )

test('lifted MC consistency', bench_lifted_bin)
