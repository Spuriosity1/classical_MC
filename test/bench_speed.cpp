// bench_speed.cpp
//
// Speed benchmark: the two lifted-Metropolis formulations in MC.cpp.
//
//   * "displacement" : local_lifted_Metropolis      — proposes u -> u + du along
//                       h_loc (exponential step) and rebuilds S in the h_loc frame.
//   * "rotation"     : local_lifted_Metropolis_rot  — rotates S along its meridian
//                       about the axis S x h_loc, carrying a sin-ratio Jacobian.
//
// Both are driven as full sweeps (via sweep_lifted_Metropolis[_rot]) on the SAME
// mildly frustrated square-lattice J1-J2 Heisenberg model, and we report the
// wall-clock time per sweep. This is a pure throughput comparison; correctness /
// distributional agreement is covered by bench_lifted.cpp.

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include <argparse/argparse.hpp>

#include "MC.hpp"

using namespace CMC;
using mat33d = vector3::mat33<double>;
using clk = std::chrono::steady_clock;

namespace {

const mat33d Heis = mat33d::from_cols({1, 0, 0}, {0, 1, 0}, {0, 0, 1});

// One spin per cell on a simple square lattice (see bench_lifted.cpp).
UnitCellSpecifier<HeisenbergSpin> SquareCell() {
    UnitCellSpecifier<HeisenbergSpin> spec(
        imat33_t::from_cols({1, 0, 0}, {0, 1, 0}, {0, 0, 1}));
    HeisenbergSpin spin;
    spin.ipos = {0, 0, 0};
    spin.pyro_sl = 0;
    spec.add(std::move(spin));
    return spec;
}

const std::vector<std::vector<ipos_t>> j1_dist = {
    {{1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}}};
const std::vector<std::vector<ipos_t>> j2_dist = {
    {{1, 1, 0}, {1, -1, 0}, {-1, 1, 0}, {-1, -1, 0}}};

void randomise_spins(Lattice& lat, std::mt19937_64& rng) {
    std::normal_distribution<double> g(0.0, 1.0);
    for (auto& s : lat.get_objects<HeisenbergSpin>()) {
        vector3::vec3d v(g(rng), g(rng), g(rng));
        double n = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
        s.S = (1.0 / n) * v;
        s.lifted_dir = 1;
    }
}

MC_runner make_runner(Lattice& lat, double J1, double J2, double T_ref,
                      uint64_t seed) {
    MC_runner mc(lat, seed);
    mc.define_general_coupling("J1", j1_dist, J1 * Heis);
    mc.define_general_coupling("J2", j2_dist, J2 * Heis);
    mc.settings.T_ref = T_ref;
    mc.setup_lattice();
    return mc;
}

struct Timing {
    double ns_per_sweep = 0.0;
    double sweeps_per_sec = 0.0;
    double accept = 0.0;   // acceptance over the timed phase (sanity check)
};

// Times `n_sweep` sweeps of one variant after `n_warm` untimed warm-up sweeps.
template <class Sweep>
Timing time_variant(size_t n_spins, Sweep&& sweep, double T, size_t n_warm,
                    size_t n_sweep) {
    for (size_t i = 0; i < n_warm; i++) sweep(T);

    size_t accepted = 0;
    auto t0 = clk::now();
    for (size_t i = 0; i < n_sweep; i++) accepted += sweep(T);
    auto t1 = clk::now();

    double ns = std::chrono::duration<double, std::nano>(t1 - t0).count();
    Timing tm;
    tm.ns_per_sweep = ns / static_cast<double>(n_sweep);
    tm.sweeps_per_sec = 1e9 / tm.ns_per_sweep;
    tm.accept = static_cast<double>(accepted) /
                (static_cast<double>(n_sweep) * static_cast<double>(n_spins));
    return tm;
}

}  // namespace

int main(int argc, char* argv[]) {
    argparse::ArgumentParser prog("bench_speed");
    prog.add_description(
        "Throughput comparison of the displacement vs. rotation lifted-Metropolis "
        "sweeps on a J1-J2 square-lattice Heisenberg model.");

    prog.add_argument("--J1").default_value(-1.0).scan<'g', double>();
    prog.add_argument("--J2").default_value(0.3).scan<'g', double>();
    prog.add_argument("--T").default_value(0.5).scan<'g', double>();
    prog.add_argument("--T_ref").default_value(1.0).scan<'g', double>();
    prog.add_argument("-L", "--L")
        .help("Linear size of the L x L square supercell")
        .default_value(32)
        .scan<'i', int>();
    prog.add_argument("--n_warm")
        .help("Untimed warm-up sweeps per variant")
        .default_value(size_t{2000})
        .scan<'i', size_t>();
    prog.add_argument("--n_sweep")
        .help("Timed sweeps per variant per repetition")
        .default_value(size_t{20000})
        .scan<'i', size_t>();
    prog.add_argument("--reps")
        .help("Repetitions (variants alternated); best time reported")
        .default_value(size_t{3})
        .scan<'i', size_t>();
    prog.add_argument("--seed").default_value(size_t{2718281828}).scan<'i', size_t>();

    try {
        prog.parse_args(argc, argv);
    } catch (const std::exception& err) {
        std::fprintf(stderr, "%s\n", err.what());
        std::cerr << prog;
        return 2;
    }

    const double J1 = prog.get<double>("--J1");
    const double J2 = prog.get<double>("--J2");
    const double T = prog.get<double>("--T");
    const double T_ref = prog.get<double>("--T_ref");
    const int L = prog.get<int>("--L");
    const size_t n_warm = prog.get<size_t>("--n_warm");
    const size_t n_sweep = prog.get<size_t>("--n_sweep");
    const size_t reps = prog.get<size_t>("--reps");
    const size_t seed = prog.get<size_t>("--seed");

    std::printf("=== lifted Metropolis speed benchmark ===\n");
    std::printf("Model : %dx%d square-lattice J1-J2 Heisenberg (%zu spins)\n", L, L,
                static_cast<size_t>(L) * static_cast<size_t>(L));
    std::printf("        J1=%g  J2=%g  T=%g  T_ref=%g\n", J1, J2, T, T_ref);
    std::printf("Stats : n_warm=%zu  n_sweep=%zu  reps=%zu  seed=%zu\n\n", n_warm,
                n_sweep, reps, seed);

    auto supercell_spec = imat33_t::from_cols({L, 0, 0}, {0, L, 0}, {0, 0, 1});
    auto cell_spec = SquareCell();
    const size_t n_spins = static_cast<size_t>(L) * static_cast<size_t>(L);

    // Independent lattice + runner per variant, identical initial configuration.
    Lattice lat_disp = build_supercell(cell_spec, supercell_spec);
    Lattice lat_rot = build_supercell(cell_spec, supercell_spec);
    std::mt19937_64 init(seed);
    randomise_spins(lat_disp, init);
    init.seed(seed);  // same starting spins for both
    randomise_spins(lat_rot, init);

    MC_runner mc_disp = make_runner(lat_disp, J1, J2, T_ref, seed + 1);
    MC_runner mc_rot = make_runner(lat_rot, J1, J2, T_ref, seed + 1);

    auto sweep_disp = [&](double t) { return mc_disp.sweep_lifted_Metropolis(t); };
    auto sweep_rot = [&](double t) { return mc_rot.sweep_lifted_Metropolis_rot(t); };

    Timing best_disp, best_rot;
    best_disp.ns_per_sweep = best_rot.ns_per_sweep = 1e300;

    for (size_t r = 0; r < reps; r++) {
        // Alternate order each rep to balance any thermal/frequency drift.
        Timing d, ro;
        if (r % 2 == 0) {
            d = time_variant(n_spins, sweep_disp, T, r == 0 ? n_warm : 0, n_sweep);
            ro = time_variant(n_spins, sweep_rot, T, r == 0 ? n_warm : 0, n_sweep);
        } else {
            ro = time_variant(n_spins, sweep_rot, T, 0, n_sweep);
            d = time_variant(n_spins, sweep_disp, T, 0, n_sweep);
        }
        if (d.ns_per_sweep < best_disp.ns_per_sweep) best_disp = d;
        if (ro.ns_per_sweep < best_rot.ns_per_sweep) best_rot = ro;
        std::printf("rep %zu:  displacement %8.1f ns/sweep   rotation %8.1f ns/sweep\n",
                    r, d.ns_per_sweep, ro.ns_per_sweep);
    }

    const double ns_per_spin_disp = best_disp.ns_per_sweep / static_cast<double>(n_spins);
    const double ns_per_spin_rot = best_rot.ns_per_sweep / static_cast<double>(n_spins);

    std::printf("\n%-14s %14s %14s\n", "quantity", "displacement", "rotation");
    std::printf("%-14s %14.1f %14.1f\n", "ns / sweep", best_disp.ns_per_sweep,
                best_rot.ns_per_sweep);
    std::printf("%-14s %14.3f %14.3f\n", "ns / spin", ns_per_spin_disp,
                ns_per_spin_rot);
    std::printf("%-14s %14.3g %14.3g\n", "sweeps / sec", best_disp.sweeps_per_sec,
                best_rot.sweeps_per_sec);
    std::printf("%-14s %13.1f%% %13.1f%%\n", "accept", 100 * best_disp.accept,
                100 * best_rot.accept);

    const double ratio = best_rot.ns_per_sweep / best_disp.ns_per_sweep;
    const char* faster = ratio > 1.0 ? "displacement" : "rotation";
    const double factor = ratio > 1.0 ? ratio : 1.0 / ratio;
    std::printf("\nFaster: %s  (by %.2fx, best-of-%zu)\n", faster, factor, reps);

    return 0;
}
