
#include <algorithm>
#include <argparse/argparse.hpp>
#include <filesystem>
#include <iostream>
#include <ostream>


#include "ssf_manager.hpp"
#include "energy_manager.hpp"
#include "pyrochlore_geometry.hpp"
#include "format_bits.hpp"

using namespace std;
using namespace CMC;

using mat33d=vector3::mat33<double>;

namespace coupling {
static const mat33d Heis = mat33d::from_cols(
        {1,0,0},{0,1,0},{0,0,1});

static const mat33d Isin = mat33d::from_cols(
        {0,0,0},{0,0,0},{0,0,1});
};

int main (int argc, char *argv[]) {
    argparse::ArgumentParser prog("anneal");

    prog.add_argument("--output_dir", "-o")
        .help("Path to output")
        .required();

    prog.add_argument("L")
        .help("Linear dimension of cubic supercell")
        .scan<'i', int>();  
    prog.add_argument("Q")
        .nargs(3)
        .scan<'i', int>();

    prog.add_argument("--J1")
        .help("Nearest-neighbour Heisenberg coupling strength")
        .default_value(-1.)
        .scan<'g', double>();
    prog.add_argument("--J2")
        .help("Second-nearest-neighbour Heisenberg coupling strength")
        .default_value(0.)
        .scan<'g', double>();
    prog.add_argument("--J3")
        .help("Third-nearest-neighbour Heisenberg coupling strength")
        .default_value(0.)
        .scan<'g', double>();

    try {
        prog.parse_args(argc, argv);
    } catch (const std::exception& err){
        cerr << err.what() << endl;
        cerr << prog;
        std::exit(1);
    }

    std::stringstream name;

    std::string outdir_s = prog.get<std::string>("output_dir");
    filesystem::path outdir(outdir_s);
    if (!filesystem::exists(outdir))
        throw runtime_error("Cannot open outdir");


    int L = prog.get<int>("L");
    auto supercell_spec = imat33_t::from_cols({L, 0, 0}, {0, L, 0}, {0, 0, L});

    auto cell_spec = PyroCubicCell();
    CMC::Lattice lat = build_supercell(cell_spec, supercell_spec);

    CMC::MC_runner mc(lat, 0);

    auto J1 = prog.get<double>("--J1");
    auto J2 = prog.get<double>("--J2");
    auto J3 = prog.get<double>("--J3");
    mc.define_coupling("J1", pyrochlore::nn1_dist, 
        mat33d::from_cols({J1, 0,0}, {0,J1, 0}, {0,0,J1})
        );
    mc.define_coupling("J2", pyrochlore::nn2_dist, J2*coupling::Heis);
    mc.define_coupling("J3a", pyrochlore::nn3a_dist, J3*coupling::Heis);
    mc.define_coupling("J3b", pyrochlore::nn3b_dist, J3*coupling::Heis);

    mc.setup_lattice();

    auto Q_vec =prog.get<std::vector<int>>("Q");
    ivec3_t Q {Q_vec[0], Q_vec[1], Q_vec[2]};

    name << "L=" << L << DELIM
         << "Q=" << Q << DELIM;

    ssf_manager ssfm(lat, {"xx", "yy", "zz"}, 1);
    energy_manager e_manager;


    auto q = lat.lattice.wavevector_from_idx3(Q);
    std::cout<<"q = "<<q<<std::endl;

    for (auto& s : lat.get_objects<HeisenbergSpin>() ){
        vector3::vec3d r = s.ipos;
        auto phase = dot(q, r);
        s.S = {0, std::cos(phase), std::sin(phase)};
    }

    ssfm.new_T(0);
    e_manager.new_T(0);


    double total_E = mc.total_energy_per_unit_cell();
    ssfm.sample();
    e_manager.sample(total_E);

    std::cout<<"Total E = "<<total_E<<std::endl;


    auto file_path = outdir / (name.str() + ".out.h5");

    hid_t file_id = H5Fcreate(file_path.string().c_str(), H5F_ACC_TRUNC,
            H5P_DEFAULT, H5P_DEFAULT);
    if (file_id < 0)
        throw std::runtime_error("Failed to create HDF5 file: " + file_path.string());
    ssfm.write_group(file_id, "/ssf");
    e_manager.write_group(file_id, "/energy");
    write_geometry_group(file_id, lat);
    std::cout<<"Saved to \n"<< file_path<<std::endl;


    auto f = outdir / (name.str() + ".spins.h5");
    printf("Saving spin state to %s\n", f.string().c_str());
    save_spin_state(lat, f);

    return 0;
}
