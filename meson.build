project('spiral_anneal_CMC', ['cpp', 'c'],
  version: '1.1.1',
  meson_version : '>= 1.5.0',
  default_options : [ 'warning_level=3', 'buildtype=release', 'cpp_std=c++2a' ]
)


json_dep = dependency('nlohmann_json', required: true)

latlib_dep = dependency('lattice_lib', version: '>= 2.3.0')

fftw_dep = dependency('fftw3', required: true)

hdf5_dep = dependency('hdf5', required: true, language: 'cpp')


build_pt = get_option('build_pt')

mpi_dep = dependency('mpi', include_type: 'system', required: build_pt)

cc=meson.get_compiler('c')

# Detect whether HDF5 was built with MPI support
if cc.compiles('''
    #include <hdf5.h>
    int main() { return 0; }
''',
    name: 'HDF5 compiles without MPI',
    dependencies: hdf5_dep
)

    message('Serial HDF5 detected')
else
  # pull in MPI

    hdf5_dep = declare_dependency(
        dependencies: [hdf5_dep, mpi_dep]
    )
    message('parallel HDF5 detected')
endif




CMC_inc=[
  include_directories(['include'])
  ]

g_deps = [
    latlib_dep,
    json_dep,
    fftw_dep,
    hdf5_dep
    ]


mc_lib = static_library('cmc_lib',
  files('src/MC.cpp'),
  dependencies: g_deps,
  include_directories: CMC_inc
  )

fieldcool_bin = executable('fieldcool', 
  files('src/fieldcool.cpp'),
  dependencies: g_deps,
  link_with: mc_lib,
  include_directories: CMC_inc
  )

if build_pt
  temper_bin = executable('temper',
    files('src/temper.cpp'),
    dependencies: [g_deps, mpi_dep],
    link_with: mc_lib,
    include_directories: CMC_inc,
    install: true
    )
endif

anneal_bin = executable('anneal', 
  files('src/anneal.cpp'),
  dependencies: g_deps,
  link_with: mc_lib,
  include_directories: CMC_inc,
  install: true
  )

eigen_dep = dependency('eigen3', required: true, include_type: 'system')

ltgs_bin = executable('ltgs',
  files('src/luttinger_tisza.cpp'),
  dependencies: g_deps + [eigen_dep],
  link_with: mc_lib,
  include_directories: CMC_inc,
  install: true
  )

py_scripts = ['plot_ssf', 'plot_q_dep', 'plot_QaB', 'acc_runs', 'ltgs_drive']

if get_option('dev')
  foreach s : py_scripts
    install_symlink(s,
      pointing_to: meson.project_source_root() / 'scripts' / s + '.py',
      install_dir: get_option('bindir'),
    )
  endforeach
else
  install_data(
    [
      'scripts/plot_ssf.py',
      'scripts/plot_q_dep.py',
      'scripts/plot_QaB.py',
      'scripts/acc_runs.py',
      'scripts/ltgs_drive.py',
    ],
    install_dir: get_option('datadir') / 'spiral_cmc' / 'scripts',
    install_mode: 'rwxr-xr-x',
  )
  foreach s : py_scripts
    install_symlink(s,
      pointing_to: '../share/spiral_cmc/scripts' / s + '.py',
      install_dir: get_option('bindir'),
    )
  endforeach
endif

subdir('test')


