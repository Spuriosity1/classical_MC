// bench_lifted.cpp
//
// Consistency benchmark: local_Metropolis vs. local_lifted_Metropolis.
//
// Both update schemes are supposed to sample the same Boltzmann distribution.
// This benchmark drives each of them on the SAME mildly frustrated square-lattice
// J1-J2 Heisenberg model (default J1=-1 ferromagnetic, J2=+0.3 antiferromagnetic),
// thermalises, then estimates <E> and <E^2> per spin. The test passes if the two
// estimators agree to within a tolerance of `tol_frac * min(|J1|, |J2|)`
// (default 10% of the smaller coupling scale).
//
// The samplers use independent RNG streams and independent random initial
// configurations, so agreement is a genuine cross-check that both reproduce the
// same stationary distribution rather than a shared trajectory.

#include <algorithm>
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

namespace {

const mat33d Heis = mat33d::from_cols({1, 0, 0}, {0, 1, 0}, {0, 0, 1});

// One spin per cell on a simple square lattice, embedded in the xy-plane of the
// 3D integer coordinate system used by lattice_lib. Lattice constant = 1.
UnitCellSpecifier<HeisenbergSpin> SquareCell() {
    UnitCellSpecifier<HeisenbergSpin> spec(
        imat33_t::from_cols({1, 0, 0}, {0, 1, 0}, {0, 0, 1}));
    HeisenbergSpin spin;
    spin.ipos = {0, 0, 0};
    spin.pyro_sl = 0;
    spec.add(std::move(spin));
    return spec;
}

// J1 = axial nearest neighbours; J2 = diagonal next-nearest neighbours.
// Indexed by pyro_sl (only sublattice 0 exists here).
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

// Batch-means standard error of the mean: splits the (autocorrelated) series
// into `n_batch` contiguous batches and takes the spread of the batch means.
double batch_stderr(const std::vector<double>& x, size_t n_batch) {
    if (x.size() < n_batch || n_batch < 2) return 0.0;
    const size_t bs = x.size() / n_batch;
    std::vector<double> means(n_batch, 0.0);
    for (size_t b = 0; b < n_batch; b++) {
        double s = 0.0;
        for (size_t i = 0; i < bs; i++) s += x[b * bs + i];
        means[b] = s / static_cast<double>(bs);
    }
    double m = 0.0;
    for (double v : means) m += v;
    m /= static_cast<double>(n_batch);
    double var = 0.0;
    for (double v : means) var += (v - m) * (v - m);
    var /= static_cast<double>(n_batch - 1);
    return std::sqrt(var / static_cast<double>(n_batch));
}

struct Estimates {
    double E = 0.0, E2 = 0.0;   // <E>, <E^2> per spin
    double E_err = 0.0, E2_err = 0.0;
    double accept = 0.0;        // acceptance rate over the sampling phase
};

// Runs one sampler (lifted or standard) and returns its energy estimators.
Estimates run_sampler(Lattice& lat, MC_runner& mc, bool lifted, double T,
                      size_t n_burn, size_t n_sample) {
    auto sweep = [&](double t) -> size_t {
        return lifted ? mc.sweep_lifted_Metropolis(t)
                      : mc.sweep_local_Metropolis(t);
    };

    for (size_t i = 0; i < n_burn; i++) sweep(T);

    const size_t n_spins = lat.get_objects<HeisenbergSpin>().size();
    std::vector<double> E_series, E2_series;
    E_series.reserve(n_sample);
    E2_series.reserve(n_sample);

    size_t accepted = 0;
    for (size_t i = 0; i < n_sample; i++) {
        accepted += sweep(T);
        double e = mc.total_energy_per_unit_cell();  // == per spin (1 site/cell)
        E_series.push_back(e);
        E2_series.push_back(e * e);
    }

    Estimates est;
    const double N = static_cast<double>(n_sample);
    for (double e : E_series) est.E += e;
    for (double e2 : E2_series) est.E2 += e2;
    est.E /= N;
    est.E2 /= N;
    const size_t n_batch = std::min<size_t>(50, std::max<size_t>(2, n_sample / 20));
    est.E_err = batch_stderr(E_series, n_batch);
    est.E2_err = batch_stderr(E2_series, n_batch);
    est.accept = static_cast<double>(accepted) / (N * n_spins);
    return est;
}

}  // namespace

int main(int argc, char* argv[]) {
    argparse::ArgumentParser prog("bench_lifted");
    prog.add_description(
        "Check that local_Metropolis and local_lifted_Metropolis sample the "
        "same distribution on a mildly frustrated J1-J2 square-lattice "
        "Heisenberg model.");

    prog.add_argument("--J1")
        .help("Nearest-neighbour Heisenberg coupling (default -1, ferromagnetic)")
        .default_value(-1.0)
        .scan<'g', double>();
    prog.add_argument("--J2")
        .help("Next-nearest-neighbour Heisenberg coupling (default +0.3)")
        .default_value(0.3)
        .scan<'g', double>();
    prog.add_argument("--T")
        .help("Temperature at which to compare the two samplers")
        .default_value(0.5)
        .scan<'g', double>();
    prog.add_argument("--T_ref")
        .help("Reference temperature for the proposal distribution")
        .default_value(1.0)
        .scan<'g', double>();
    prog.add_argument("-L", "--L")
        .help("Linear size of the L x L square supercell")
        .default_value(8)
        .scan<'i', int>();
    prog.add_argument("--n_burn")
        .help("Thermalisation sweeps before sampling")
        .default_value(size_t{5000})
        .scan<'i', size_t>();
    prog.add_argument("--n_sample")
        .help("Measurement sweeps (one energy measurement per sweep)")
        .default_value(size_t{50000})
        .scan<'i', size_t>();
    prog.add_argument("--tol_frac")
        .help("Tolerance as a fraction of min(|J1|,|J2|)")
        .default_value(0.1)
        .scan<'g', double>();
    prog.add_argument("--seed")
        .help("Base RNG seed")
        .default_value(size_t{2718281828})
        .scan<'i', size_t>();

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
    const size_t n_burn = prog.get<size_t>("--n_burn");
    const size_t n_sample = prog.get<size_t>("--n_sample");
    const double tol_frac = prog.get<double>("--tol_frac");
    const size_t seed = prog.get<size_t>("--seed");

    const double tol = tol_frac * std::min(std::abs(J1), std::abs(J2));

    std::printf("=== lifted vs. standard Metropolis consistency benchmark ===\n");
    std::printf("Model : %dx%d square-lattice J1-J2 Heisenberg\n", L, L);
    std::printf("        J1=%g  J2=%g  T=%g  T_ref=%g\n", J1, J2, T, T_ref);
    std::printf("Stats : n_burn=%zu  n_sample=%zu  seed=%zu\n", n_burn, n_sample,
                seed);
    std::printf("Tol   : %g  (= %g * min(|J1|,|J2|)=%g)\n\n", tol, tol_frac,
                std::min(std::abs(J1), std::abs(J2)));

    // SplitMix64 to fan the base seed out into independent 64-bit streams.
    auto splitmix = [](uint64_t& x) {
        x += 0x9E3779B97F4A7C15ULL;
        uint64_t z = x;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        return z ^ (z >> 31);
    };
    uint64_t sm = seed;
    const uint64_t seed_std_init = splitmix(sm);
    const uint64_t seed_std_mc = splitmix(sm);
    const uint64_t seed_lft_init = splitmix(sm);
    const uint64_t seed_lft_mc = splitmix(sm);

    auto supercell_spec = imat33_t::from_cols({L, 0, 0}, {0, L, 0}, {0, 0, 1});
    auto cell_spec = SquareCell();

    // --- Standard Metropolis ---
    Lattice lat_std = build_supercell(cell_spec, supercell_spec);
    std::mt19937_64 init_std(seed_std_init);
    randomise_spins(lat_std, init_std);
    MC_runner mc_std = make_runner(lat_std, J1, J2, T_ref, seed_std_mc);
    std::printf("Running standard Metropolis...\n");
    Estimates est_std = run_sampler(lat_std, mc_std, /*lifted=*/false, T, n_burn,
                                    n_sample);

    // --- Lifted Metropolis ---
    Lattice lat_lft = build_supercell(cell_spec, supercell_spec);
    std::mt19937_64 init_lft(seed_lft_init);
    randomise_spins(lat_lft, init_lft);
    MC_runner mc_lft = make_runner(lat_lft, J1, J2, T_ref, seed_lft_mc);
    std::printf("Running lifted Metropolis...\n");
    Estimates est_lft = run_sampler(lat_lft, mc_lft, /*lifted=*/true, T, n_burn,
                                    n_sample);

    const double dE = std::abs(est_std.E - est_lft.E);
    const double dE2 = std::abs(est_std.E2 - est_lft.E2);

    std::printf("\n%-10s %14s %14s %14s\n", "quantity", "standard", "lifted",
                "|difference|");
    std::printf("%-10s %14.6f %14.6f %14.6f\n", "<E>", est_std.E, est_lft.E, dE);
    std::printf("%-10s %14.6f %14.6f %14.6f\n", "<E^2>", est_std.E2, est_lft.E2,
                dE2);
    std::printf("%-10s %14.6f %14.6f\n", "stderr<E>", est_std.E_err,
                est_lft.E_err);
    std::printf("%-10s %14.6f %14.6f\n", "stderr<E2>", est_std.E2_err,
                est_lft.E2_err);
    std::printf("%-10s %13.1f%% %13.1f%%\n", "accept", 100 * est_std.accept,
                100 * est_lft.accept);

    const bool pass_E = dE < tol;
    const bool pass_E2 = dE2 < tol;
    std::printf("\n<E>   : |diff|=%.6f  tol=%.6f  -> %s\n", dE, tol,
                pass_E ? "PASS" : "FAIL");
    std::printf("<E^2> : |diff|=%.6f  tol=%.6f  -> %s\n", dE2, tol,
                pass_E2 ? "PASS" : "FAIL");

    if (pass_E && pass_E2) {
        std::printf("\nRESULT: PASS\n");
        return 0;
    }
    std::printf("\nRESULT: FAIL\n");
    return 1;
}
