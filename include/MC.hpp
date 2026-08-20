#pragma once
#include "H5Ipublic.h"
#include "H5Ppublic.h"
#include "H5Tpublic.h"
#include <nlohmann/json.hpp>
#include <filesystem>
#include <map>
#include <vector>
#include <string>
#include <cassert>

// LatticeLab 2
#include <lattice_lib/supercell.hpp>
#include <XoshiroCpp.hpp>
#include <random>


namespace CMC {

using json=nlohmann::json;
using dmat33_t = vector3::mat33<double>;


struct HeisenbergSpin;

/**
 * @brief A group of neighbours of one spin that all share the same coupling
 * matrix orientation.
 *
 * `J` points into the owning CouplingSpec (either its `J` or its transpose
 * `Jt`). The field contribution of this shell to its owner spin is
 * `*J * sum(bonds->S)`. A spin that is the *source* (lower pyro_sl) of a bond
 * points at `J`; the *target* (higher pyro_sl) points at `Jt`. A pointer (not a
 * reference) is used so BondShell stays copy-assignable, as build_supercell
 * requires of HeisenbergSpin. The pointee stays valid because `coupling_specs`
 * is never resized after setup_lattice().
 */
struct GenericBondShell {
    const dmat33_t* J;
    std::vector<HeisenbergSpin*> bonds;
};


/**
 * Like GenericBondShell, for the special (yet common) case that 
 * J is an identity matrix.
 */
struct HeisBondShell {
    const double* J;
    std::vector<HeisenbergSpin*> bonds;
};

// Biquadratic bonds, understood as K_{ij} (S_i \cdot S_j)^2
struct BQBondShell {
    const double* K;
    std::vector<HeisenbergSpin*> bonds;
};


/**
 * @brief Classical Heisenberg spin stored in the supercell.
 *
 * Satisfies the GeometricObject concept via the `ipos` member.
 *
 * `general_bonds` holds this spin's neighbours grouped by which oriented coupling
 * matrix applies (see BondShell); local_field sums J*neighbours over shells.
 * `heis_bonds` holds bilinear bonds that are pure Heisenberg.
 * `biquad_bonds` holds the bonds of the form K (S \cdot S)^2.
 */
struct HeisenbergSpin {
    ipos_t ipos;                           // required by GeometricObject concept
    int pyro_sl = 0;                       // pyrochlore sublattice 0–3
    vector3::vec3<double> S;
    int8_t lifted_dir = 1;                 // ±1; used by sweep_lifted_Metropolis, ignored otherwise

    std::vector<GenericBondShell> general_bonds;
    std::vector<HeisBondShell> heis_bonds;
    std::vector<BQBondShell> biquad_bonds;
};

/**
 * @brief Supercell of Heisenberg spins on the pyrochlore lattice.
 */
typedef Supercell<HeisenbergSpin> Lattice;


/**
 * @brief Specification of a single coupling term in the Hamiltonian.
 *
 * relative_vectors[sl] lists displacement vectors from a spin of pyrochlore
 * sublattice sl (0–3) to its coupled neighbors.
 */
struct GeneralCouplingSpec {
    std::string name;
    std::vector<std::vector<ipos_t>> relative_vectors;
    dmat33_t J;    // applied by the source (lower-pyro_sl) endpoint of a bond
    dmat33_t Jt;   // = J.tr(); applied by the target (higher-pyro_sl) endpoint
};

struct HeisenbergCouplingSpec {
    std::string name;
    std::vector<std::vector<ipos_t>> relative_vectors;
    double J;
};

struct BiquadraticCouplingSpec {
    std::string name;
    std::vector<std::vector<ipos_t>> relative_vectors;
    double K;
};

struct MC_parameters {
    double T_ref=1.0;
    size_t verbosity = 2;
    double hloc_atol = 1e-16;
    double u_atol = 1e-16;
};

/**
 * @brief Monte Carlo driver for classical spin simulations.
 * The internal bond representation stores interactions in the form
 *
 *   \sum J_{ij}^{a b} S_i^a S_j^b 
 * + \sum Jh_{ij}      S_i^a S_j^a 
 * + \sum K_{ij}/2    (S_i^a S_j^a)^2
 *
 * Heisenberg interactions are stored separately for better performance.
 */
class MC_runner {
    std::vector<GeneralCouplingSpec> general_coupling_specs;
    std::vector<HeisenbergCouplingSpec> heis_coupling_specs;
    std::vector<BiquadraticCouplingSpec> biquad_coupling_specs;

    // these are currently only used to ensure that names are unique.
    using index_entry_t = std::pair<size_t, size_t>;
    std::map<std::string, index_entry_t> index; 

    vector3::vec3<double> global_field={0,0,0};

    Lattice* lat;

    std::uniform_int_distribution<size_t> site_dist;
    std::normal_distribution<double> normal_dist;
    std::exponential_distribution<double> exp_dist;
    std::uniform_real_distribution<double> rand01;

    XoshiroCpp::Xoroshiro128PlusPlus rng;

    vector3::vec3d local_linear_field(const HeisenbergSpin* spin) const;
    vector3::vec3d local_field(const HeisenbergSpin* spin) const;

    // Single-site biquadratic energy at trial orientation S (K/2 convention).
    // Returns 0 when the spin has no biquad bonds, so pure-linear runs pay
    // nothing. Sums each bond touching the site once, so differences of this
    // quantity give the correct dE for a proposed move at that site.
    double biquad_site_energy(const HeisenbergSpin* spin,
            const vector3::vec3d& S) const;

public:
    MC_parameters settings;

    MC_runner(Lattice &lat_, size_t seed)
        : lat(&lat_),
          site_dist(0, lat_.get_objects<HeisenbergSpin>().size()-1),
          rand01(0,1),
          rng(seed)
    {}

    // Repoints this runner at a different (but congruent) Lattice — O(1), no
    // spin data copied. Used by parallel tempering to swap which replica's
    // configuration a given temperature slot is currently simulating.
    void rebind(Lattice& new_lat);

    double total_energy_per_unit_cell() const;

    void define_general_coupling(const std::string& name,
            const std::vector<std::vector<ipos_t>>& rel_vecs,
            const vector3::mat33<double>& J
            );

    void define_Heisenberg_coupling(const std::string& name,
            const std::vector<std::vector<ipos_t>>& rel_vecs,
            const double J
            );

    void define_biquad_coupling(const std::string& name,
            const std::vector<std::vector<ipos_t>>& rel_vecs,
            const double K
            );

    void set_global_field(const vector3::vec3<double>& h);
    vector3::vec3d get_global_field() const;

    void setup_lattice();

    // Set every spin to an independent uniformly-random unit vector, so the
    // simulation never starts from the zero-initialised state (which leaves the
    // local field undefined for field-based moves). Overwritten by an explicit
    // initial state, e.g. init_spiral_state.
    void randomize_spins();

    size_t local_overrelax(double T, HeisenbergSpin* spin); //*
    size_t local_Metropolis(double T, HeisenbergSpin* spin); // ** 
    size_t local_lifted_Metropolis(double T, HeisenbergSpin* spin); // **
    size_t local_lifted_Metropolis_rot(double T, HeisenbergSpin* spin); // **

    size_t overrelax_all(double T);
    size_t sweep_local_Metropolis(double T, size_t n_overrelax=1);
    size_t sweep_lifted_Metropolis(double T, size_t n_overrelax=1);
    size_t sweep_lifted_Metropolis_rot(double T, size_t n_overrelax=1);
};



// Write a "/geometry" group to an open HDF5 file with lattice statistics
// that are fixed for this disorder realisation:
//
//   n_spins            (scalar)  — number of non-deleted spins
//   n_tetras_by_intact (hsize_t[5]) — n_tetras_by_intact[k] = number of
//                                     tetrahedra with exactly k intact spins
 void write_geometry_group(hid_t file_id, Lattice& sc,
                                 const char* group_name = "geometry");



void save_spin_state(Lattice& lat, const std::filesystem::path& file_path);
void save_ft_spin_state(Lattice& lat, const std::filesystem::path& file_path);


}
