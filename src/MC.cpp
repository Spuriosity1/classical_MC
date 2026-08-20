#include "MC.hpp"
#include "H5Gpublic.h"
#include "ssf_manager.hpp"
#include <random>
#include <cmath>
#include <algorithm>

namespace CMC {


    double norm(const vector3::vec3d& v){
        return sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
    }

    void mirror_about_vector(vector3::vec3d& v, const vector3::vec3d& axis){
        v = -v + 2 *(dot(axis, v) / (dot(axis, axis) + 1e-10) ) * axis;
    }

    vector3::vec3d cross(const vector3::vec3d& a, const vector3::vec3d& b){
        return {a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]};
    }

    void rotate_about_vector(vector3::vec3d& v, const vector3::vec3d& axis, double theta){
        // generates a random vector with fixed projection onto axis
        double c = std::cos(theta);
        double s = std::sin(theta);
        double norm2 = dot(axis, axis);
        // Rodruigez formula
        v = c * v + ((1-c)/norm2) * dot(axis, v) * axis + (s / sqrt(norm2)) * cross(axis, v);
    }


    void MC_runner::setup_lattice(){
        auto& spins = lat->get_objects<HeisenbergSpin>();
        const int Np = lat->lattice.num_primitive_cells();
        const int num_sl = static_cast<int>(
            std::get<SlPos<HeisenbergSpin>>(lat->sl_positions).size());

        // For each coupling, split each spin's neighbours into two BondShells:
        // those where the spin is the source (lower pyro_sl -> apply J) and
        // those where it is the target (higher pyro_sl -> apply J^T). The two
        // directed halves of a bond thus reference J and Jt respectively, so
        // the undirected bond energy S_mu^T J S_nu is well-defined without any
        // runtime transpose. Equal pyro_sl only occurs for symmetric couplings
        // (J == Jt), so it is grouped with the source side.
        for (const auto& c : general_coupling_specs){
            for (int sl = 0; sl < num_sl; sl++) {
                const int pyro_sl = sl % 4;  // pyrochlore sublattice within FCC site
                for (int cell = 0; cell < Np; cell++) {
                    HeisenbergSpin* origin = &spins[sl * Np + cell];

                    std::vector<HeisenbergSpin*> src_bonds;  // origin is source -> J
                    std::vector<HeisenbergSpin*> tgt_bonds;  // origin is target -> Jt
                    for (const auto& v : c.relative_vectors.at(pyro_sl)) {
                        HeisenbergSpin* other =
                            lat->get_object_at<HeisenbergSpin>(origin->ipos + v);
                        if (origin->pyro_sl <= other->pyro_sl)
                            src_bonds.push_back(other);
                        else
                            tgt_bonds.push_back(other);
                    }

                    if (!src_bonds.empty())
                        origin->general_bonds.push_back({&c.J,  std::move(src_bonds)});
                    if (!tgt_bonds.empty())
                        origin->general_bonds.push_back({&c.Jt, std::move(tgt_bonds)});
                }
            }
        }

        // Heisenberg and biquadratic couplings are symmetric scalars, so there
        // is no source/target (J vs J^T) distinction: every neighbour goes into
        // a single shell. Each undirected bond is still stored at both endpoints;
        // per-site energy differences count it once, total_energy applies 0.5.
        for (const auto& c : heis_coupling_specs){
            for (int sl = 0; sl < num_sl; sl++) {
                const int pyro_sl = sl % 4;
                for (int cell = 0; cell < Np; cell++) {
                    HeisenbergSpin* origin = &spins[sl * Np + cell];
                    std::vector<HeisenbergSpin*> bonds;
                    for (const auto& v : c.relative_vectors.at(pyro_sl))
                        bonds.push_back(
                            lat->get_object_at<HeisenbergSpin>(origin->ipos + v));
                    if (!bonds.empty())
                        origin->heis_bonds.push_back({&c.J, std::move(bonds)});
                }
            }
        }

        for (const auto& c : biquad_coupling_specs){
            for (int sl = 0; sl < num_sl; sl++) {
                const int pyro_sl = sl % 4;
                for (int cell = 0; cell < Np; cell++) {
                    HeisenbergSpin* origin = &spins[sl * Np + cell];
                    std::vector<HeisenbergSpin*> bonds;
                    for (const auto& v : c.relative_vectors.at(pyro_sl))
                        bonds.push_back(
                            lat->get_object_at<HeisenbergSpin>(origin->ipos + v));
                    if (!bonds.empty())
                        origin->biquad_bonds.push_back({&c.K, std::move(bonds)});
                }
            }
        }
    }



    void MC_runner::randomize_spins(){
        for (auto& s : lat->get_objects<HeisenbergSpin>()){
            vector3::vec3d v{normal_dist(rng), normal_dist(rng), normal_dist(rng)};
            double n = norm(v);
            // Re-draw the measure-zero all-zero Gaussian sample so 1/n is finite.
            while (n < 1e-12) {
                v = vector3::vec3d{normal_dist(rng), normal_dist(rng), normal_dist(rng)};
                n = norm(v);
            }
            v /= n;
            s.S = v;
        }
    }


    void MC_runner::define_general_coupling(const std::string& name,
            const std::vector<std::vector<ipos_t>>& rel_vecs,
            const vector3::mat33<double>& J)
    {
        if (index.contains(name))
            throw std::logic_error("Coupling names must be unique");

        index[name] = index_entry_t(0, general_coupling_specs.size());
        general_coupling_specs.push_back({name, rel_vecs, J, J.tr()});
    }


    void MC_runner::define_Heisenberg_coupling(const std::string& name,
            const std::vector<std::vector<ipos_t>>& rel_vecs,
            const double J
            ){
        if (index.contains(name))
            throw std::logic_error("Coupling names must be unique");

        index[name] = index_entry_t(1, heis_coupling_specs.size());
        heis_coupling_specs.push_back({name, rel_vecs, J});
    }

    void MC_runner::define_biquad_coupling(const std::string& name,
            const std::vector<std::vector<ipos_t>>& rel_vecs,
            const double K
            ){
        if (index.contains(name))
            throw std::logic_error("Coupling names must be unique");

        index[name] = index_entry_t(2, biquad_coupling_specs.size());
        biquad_coupling_specs.push_back({name, rel_vecs, K});
        // index is literally only used here: consider deleting
    }


    void MC_runner::set_global_field(const vector3::vec3<double>& h){
        global_field = h;
    }

    vector3::vec3d MC_runner::get_global_field() const {
        return global_field;
    }


    void accumulate_field(vector3::vec3d& h,
            const std::vector<HeisenbergSpin*>& spin_list){
        for (auto& s : spin_list) {h += s->S;}
    }

    vector3::vec3d MC_runner::local_linear_field(const HeisenbergSpin *spin) const
    {
        vector3::vec3d h_loc{0,0,0};
        for (const auto& shell : spin->general_bonds) {
            vector3::vec3d tmp{0,0,0};
            accumulate_field(tmp, shell.bonds);
            h_loc += *shell.J * tmp; // this is a matrix-vector operation
        }
        for (const auto& shell : spin->heis_bonds) {
            vector3::vec3d tmp{0,0,0};
            accumulate_field(tmp, shell.bonds);
            h_loc += *shell.J * tmp; // simple scalar multiplication
        }
        return h_loc;
    }


    vector3::vec3d MC_runner::local_field(const HeisenbergSpin *spin) const 
    {
        auto h_loc = local_linear_field(spin);
        for (const auto& shell : spin->biquad_bonds) {
            vector3::vec3d tmp{0,0,0};
            for (auto& other : shell.bonds){
                tmp += dot(other->S, spin->S)*other->S;
            }
            h_loc += *shell.K * tmp;
        }
        return h_loc;
    }

    double MC_runner::biquad_site_energy(const HeisenbergSpin* spin,
            const vector3::vec3d& S) const
    {
        double E = 0;
        for (const auto& shell : spin->biquad_bonds) {
            double acc = 0;
            for (const auto* other : shell.bonds) {
                double d = dot(S, other->S);
                acc += d * d;
            }
            E += 0.5 * (*shell.K) * acc;  // K/2 convention (see local_field)
        }
        return E;
    }

    size_t MC_runner::local_Metropolis(double T, HeisenbergSpin* spin)
    {
        auto h_loc = local_linear_field(spin) - global_field;
        double curr_E = dot(spin->S, h_loc);

        auto new_S = sqrt(T/settings.T_ref) * vector3::vec3d(
                normal_dist(rng), normal_dist(rng), normal_dist(rng));
        new_S += spin->S;
        new_S /= norm(new_S);

        double new_E = dot(new_S, h_loc);

        double dE = (new_E - curr_E)
                  + biquad_site_energy(spin, new_S)
                  - biquad_site_energy(spin, spin->S);
        if (dE < 0 || rand01(rng) < exp(-dE / T)) {
            spin->S = new_S;
            return 1;
        }

        return 0;
    }

    // // overrelaxes proportion p of the spins
    // void MC_runner::overrelax_some(double p){
    //     auto& spins = lat->get_objects<HeisenbergSpin>();
    //
    //     for (int i=0; i<p*spins.size(); ++i) {
    //         auto* spin = &spins[site_dist(rng)];
    //         auto h_loc = local_field(spin) - global_field;
    //         mirror_about_vector(spin->S, h_loc);
    //     }
    // }


    size_t MC_runner::local_overrelax(double T, HeisenbergSpin* spin){
        auto h_loc = local_linear_field(spin) - global_field;

        // Rotation about a vanishing field is undefined (rotate_about_vector
        // divides by ‖h_loc‖²). This happens when a spin's neighbours are all
        // zero — e.g. spins not yet bootstrapped off the zero-initialised state,
        // which the lifted step cannot always move. Treat it as a no-op identity
        // move; the Metropolis/lifted step will orient the spin later.
        if (norm(h_loc) < settings.hloc_atol)
            return 0;

        // Rotation about h_loc leaves the linear single-site energy S·h_loc
        // invariant, so with no biquad terms the move is microcanonical and
        // always accepted (bit-for-bit the previous behaviour, zero overhead).
        if (spin->biquad_bonds.empty()) {
            rotate_about_vector(spin->S, h_loc, 2*M_PI*rand01(rng));
            return 1;
        }

        // Biquadratic terms are not conserved by the rotation; the proposal is
        // symmetric (uniform angle), so a plain Metropolis test on the biquad
        // residual restores detailed balance.
        auto old_S = spin->S;
        double old_bq = biquad_site_energy(spin, old_S);
        rotate_about_vector(spin->S, h_loc, 2*M_PI*rand01(rng));
        double dE = biquad_site_energy(spin, spin->S) - old_bq;
        if (dE <= 0 || rand01(rng) < exp(-dE / T))
            return 1;
        spin->S = old_S;
        return 0;
    }


    // overrelaxes all the spins
    size_t MC_runner::overrelax_all(double T){
        auto& spins = lat->get_objects<HeisenbergSpin>();
        size_t accepted = 0;
        for (size_t i=0; i<spins.size(); ++i) {
            auto* spin = &spins[i];
            accepted += local_overrelax(T, spin);
        }
        return accepted;
    }

    size_t MC_runner::sweep_local_Metropolis(double T, size_t n_overrelax){
        size_t accepted = 0;
        for (auto& spin : lat->get_objects<HeisenbergSpin>()){
            accepted += local_Metropolis(T, &spin);
        }
        for (size_t i=0; i<n_overrelax; i++)
            overrelax_all(T);
        return accepted;
    }

    // Lifted Metropolis step for a single spin (rotation formulation).
    //
    // Proposes a rotation of S by (lifted_dir * delta) radians around the axis
    // S × h_loc. This slides S along the meridian of the sphere whose pole is
    // ĥ_loc, i.e. it changes only ψ = angle(S, h_loc) (cos ψ = S·ĥ_loc) and
    // leaves the azimuth fixed. On rejection the lifting direction is flipped
    // instead of staying put, suppressing the diffusive back-and-forth of
    // standard Metropolis while keeping π ∝ exp(-E/T) stationary via skewed
    // detailed balance.
    //
    // The meridian shift ψ → ψ + δ is NOT measure-preserving: the uniform sphere
    // measure is sinψ dψ dφ, so the map carries a Jacobian sinψ_new / sinψ_old
    // that MUST multiply the Boltzmann factor in the acceptance ratio. Since
    // |S × h_loc| = |h_loc| sinψ and |h_loc| is fixed during the move, that
    // Jacobian is just the ratio of the rotation-axis lengths. Dropping it (as a
    // naive Metropolis test would) turns this update into a persistent descent
    // toward the field direction and systematically over-cools the system.
    //
    // Skewed detailed balance additionally requires the reverse move to be the
    // exact inverse of the forward one: g_{-ε}(g_ε(S)) = S. That holds only while
    // ψ ± δ stays inside (0, π). The rotation axis is a(S) = S × h_loc =
    // |h_loc| sinψ n̂ with n̂ the fixed normal of the (S, h_loc) plane; when a move
    // carries ψ through a pole (0 or π) the sign of sinψ flips, so a(S_new) points
    // OPPOSITE to a(S) and the -ε rotation no longer undoes the move. Rotating
    // straight through the pole therefore breaks the involution and injects a net
    // probability current (an O(δ) bias that over-cools the lattice). We instead
    // treat each pole as a reflecting wall: a proposal that would cross it is
    // rejected and the lifting direction flipped (a "bounce"), leaving S put. The
    // crossing is detected by the axis flipping orientation, dot(a(S),a(S_new))<0.
    size_t MC_runner::local_lifted_Metropolis_rot(double T, HeisenbergSpin* spin)
    {
        auto h_loc = local_linear_field(spin) - global_field;
        double curr_E = dot(spin->S, h_loc);

        double delta = sqrt(T / settings.T_ref);

        // Rotation axis perpendicular to S in the S–h_loc plane.
        // rotate_about_vector handles unnormalised axes, and dot(axis, S) = 0
        // (cross product ⊥ both factors), so the Rodriguez term in S vanishes.
        // |axis|² = |h_loc|² sin²ψ_old.
        auto axis = cross(spin->S, h_loc);
        double axis_n2_old = dot(axis, axis);

        // Poles (S ∥ ±h_loc): the meridian is undefined and the Jacobian is
        // singular. This is a measure-zero configuration, so treat it as a
        // rejection — flip the lifting direction and wait for the neighbourhood
        // (hence h_loc) to move S off the pole on a later sweep.
        if (axis_n2_old <= 1e-20) {
            spin->lifted_dir = -spin->lifted_dir;
            return 0;
        }

        auto new_S = spin->S;
        rotate_about_vector(new_S, axis, spin->lifted_dir * delta);
        new_S /= norm(new_S); // numerical safety

        // |axis_new|² = |h_loc|² sin²ψ_new, so sqrt(new/old) = sinψ_new/sinψ_old.
        auto axis_new = cross(new_S, h_loc);
        double axis_n2_new = dot(axis_new, axis_new);

        // Reflecting bounce: if the move crossed a pole the rotation axis flipped
        // orientation, so the reverse move would not invert it and skewed detailed
        // balance is broken. Reject, flip the lifting direction, and leave S put.
        if (dot(axis, axis_new) < 0) {
            spin->lifted_dir = -spin->lifted_dir;
            return 0;
        }

        double dE = dot(new_S, h_loc) - curr_E
                  + biquad_site_energy(spin, new_S)
                  - biquad_site_energy(spin, spin->S);
        // Skewed-detailed-balance acceptance min(1, [sinψ_new/sinψ_old]·e^{-ΔE/T}).
        // The biquad residual enters only through the symmetric energy, so the
        // linear-field Jacobian (axis-length ratio) is unchanged.
        // rand01 < (ratio ≥ 1) is always true, so no explicit clamp is needed.
        double accept_ratio = sqrt(axis_n2_new / axis_n2_old) * exp(-dE / T);
        if (rand01(rng) < accept_ratio) {
            spin->S = new_S;
            return 1;
        }

        spin->lifted_dir = -spin->lifted_dir;
        return 0;
    }

    // Lifted Metropolis algorithm.
    // Propose S-> S + delta*h,
    // where delta has the sign of lifted_dir
    size_t MC_runner::local_lifted_Metropolis(double T, HeisenbergSpin* spin){
        auto h_loc = local_linear_field(spin) - global_field;
        double curr_E = dot(spin->S, h_loc);

        auto n_hloc= norm(h_loc);
        // early guard in case h_loc = 0
        if (n_hloc < settings.hloc_atol)
            return local_Metropolis(T, spin);

        auto h_loc_hat = h_loc; h_loc_hat /= n_hloc;
        // u = cos∠(S, h_loc) ∈ [-1,1] mathematically, but the dot product of two
        // unit vectors can round to 1+ε when S ∥ h_loc. Clamp so that 1-u*u stays
        // non-negative; otherwise sqrt((1-new_u²)/(1-u*u)) below yields NaN. The
        // near-pole guard (abs(1-u*u) < u_atol) then rejects the degenerate case.
        double u = std::clamp(dot(h_loc_hat, spin->S), -1.0, 1.0);

        double du =  spin->lifted_dir * sqrt(T / settings.T_ref) * exp_dist(rng);

        double new_u = u+du;

        // ensure that we did not move off the sphere
        if (new_u > 1 || new_u < -1 || abs(1-u*u) < settings.u_atol){
            spin->lifted_dir = -spin->lifted_dir;
            return 0;
        }

        auto new_S = new_u * h_loc_hat + sqrt( (1-new_u*new_u)/ (1-u*u) ) * (spin->S - u*h_loc_hat);
        new_S /= norm(new_S);

        double dE = dot(new_S, h_loc) - curr_E
                  + biquad_site_energy(spin, new_S)
                  - biquad_site_energy(spin, spin->S);

        double accept_ratio =  exp(-dE / T);
        if (rand01(rng) < accept_ratio) {
            spin->S = new_S;
            return 1;
        }

        spin->lifted_dir = -spin->lifted_dir;
        return 0;
    }

    size_t MC_runner::sweep_lifted_Metropolis(double T, size_t n_overrelax){
        size_t accepted = 0;
        for (auto& spin : lat->get_objects<HeisenbergSpin>()){
            accepted += local_lifted_Metropolis(T, &spin);
        }
        for (size_t n=0; n<n_overrelax; n++)
            overrelax_all(T);

        return accepted;
    }

    size_t MC_runner::sweep_lifted_Metropolis_rot(double T, size_t n_overrelax){
        size_t accepted = 0;
        for (auto& spin : lat->get_objects<HeisenbergSpin>()){
            accepted += local_lifted_Metropolis_rot(T, &spin);
        }
        for (size_t n=0; n<n_overrelax; n++)
            overrelax_all(T);

        return accepted;
    }

    double MC_runner::total_energy_per_unit_cell() const{
        double E = 0;
        for (const auto& s : std::get<std::vector<HeisenbergSpin>>(lat->objects)){
            E += 0.5 * dot(s.S, local_linear_field(&s));
            E += 0.5 * biquad_site_energy(&s, s.S);
            E -= dot(s.S, global_field);
        }
        return E / lat->lattice.num_primitive_cells();
    }

    void MC_runner::rebind(Lattice& new_lat){
        assert(new_lat.get_objects<HeisenbergSpin>().size()
                == lat->get_objects<HeisenbergSpin>().size());
        lat = &new_lat;
    }

    void save_ft_spin_state(Lattice& lat, const std::filesystem::path& file_path){
        hid_t file = H5Fcreate(file_path.string().c_str(),
                H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

            // Sublattice-resolved DFT of the spin field, one component at a time.
            // Stored as the raw per-cell transform Ã_μ^raw(K) WITHOUT the
            // sublattice phase exp(-i q·r_μ) — matching the ssf_manager
            // convention; the phase is applied in post-processing.
            FT_Sx_t tf_x(lat);
            FT_Sy_t tf_y(lat);
            FT_Sz_t tf_z(lat);

            tf_x.transform();
            tf_y.transform();
            tf_z.transform();

            const FourierBuffer<HeisenbergSpin>* bufs[3] = {
                &tf_x.get_buffer(), &tf_y.get_buffer(), &tf_z.get_buffer()};

            const int num_sl = bufs[0]->num_sublattices;
            const ivec3_t kd = bufs[0]->k_dims;
            const size_t n_k = static_cast<size_t>(kd[0]) * kd[1] * kd[2];

            // Layout: [component (x,y,z), sublattice, k-index, {re, im}]
            std::vector<double> ft(3 * num_sl * n_k * 2);
            for (int c = 0; c < 3; c++) {
                const auto& buf = *bufs[c];
                for (int sl = 0; sl < num_sl; sl++) {
                    for (size_t k = 0; k < n_k; k++) {
                        const size_t base =
                            ((static_cast<size_t>(c) * num_sl + sl) * n_k + k) * 2;
                        ft[base]     = buf[sl][k].real();
                        ft[base + 1] = buf[sl][k].imag();
                    }
                }
            }

            hsize_t ft_dims[4] = {3, static_cast<hsize_t>(num_sl),
                                  static_cast<hsize_t>(n_k), 2};
            hid_t ft_space = H5Screate_simple(4, ft_dims, NULL);
            hid_t ft_dset = H5Dcreate(file, "spin_ft", H5T_IEEE_F64LE, ft_space,
                                      H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
            H5Dwrite(ft_dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                     ft.data());
            H5Dclose(ft_dset);
            H5Sclose(ft_space);
        
        H5Fclose(file);
    }


    void save_spin_state(Lattice& lat, const std::filesystem::path& file_path){
        const auto& spins = std::get<std::vector<HeisenbergSpin>>(lat.objects);
        const size_t N = spins.size();

        std::vector<int32_t> pos(3 * N);
        std::vector<double> ori(3 * N);

        for (size_t idx = 0; idx < N; idx++) {
            const auto& s = spins[idx];
            for (int i = 0; i < 3; i++){
                pos[3*idx+i] = static_cast<int32_t>(s.ipos[i]);
                ori[3*idx+i] = s.S[i];
            }
        }

        hid_t file = H5Fcreate(file_path.string().c_str(),
                H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

        hsize_t dims[2] = {N, 3};
        hid_t space = H5Screate_simple(2, dims, NULL);

        hid_t dset_pos = H5Dcreate(file, "spin_pos", H5T_STD_I32LE, space,
                H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5Dwrite(dset_pos, H5T_NATIVE_INT32, H5S_ALL, H5S_ALL, H5P_DEFAULT, pos.data());
        H5Dclose(dset_pos);

        hid_t dset_ori =
            H5Dcreate(file, "spin_orientation", H5T_IEEE_F64LE, space,
                      H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5Dwrite(dset_ori, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                 ori.data());
        H5Dclose(dset_ori);
        H5Sclose(space);

        H5Fclose(file);
    }



// Write a "/geometry" group to an open HDF5 file with lattice statistics
// that are fixed for this disorder realisation:
//
//   n_spins            (scalar)  — number of non-deleted spins
//   n_tetras_by_intact (hsize_t[5]) — n_tetras_by_intact[k] = number of
//                                     tetrahedra with exactly k intact spins
 void write_geometry_group(hid_t file_id, Lattice& sc,
                                 const char* group_name) {

    hid_t grp = H5Gcreate2(file_id, group_name, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

//    hid_t scalar_space = H5Screate(H5S_SCALAR);
//    hid_t ds = H5Dcreate2(grp, "n_spins", H5T_NATIVE_HSIZE,
//                           scalar_space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
//    H5Dwrite(ds, H5T_NATIVE_HSIZE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &n_spins);
//    H5Dclose(ds);
//    H5Sclose(scalar_space);


    auto write_mat = [&](const char* name, const auto& matrix, hid_t type) {
        hsize_t dims[2] = { 3, 3 };
        decltype(matrix(0,0)) data_rm[9];
        for (int i=0; i<9; i++){
            data_rm[i] = matrix(i/3, i%3);
        }
        hid_t sp = H5Screate_simple(2, dims, nullptr);
        hid_t ds = H5Dcreate2(grp, name, type, sp,
                               H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        if (ds < 0) {
            H5Sclose(sp);
            H5Gclose(grp);
            throw std::runtime_error(std::string("ssf_manager: failed to create ") + name);
        }
        H5Dwrite(ds, type, H5S_ALL, H5S_ALL, H5P_DEFAULT, data_rm);
        H5Dclose(ds);
        H5Sclose(sp);

    };


    write_mat("index_cell", sc.lattice.get_lattice_vectors(), H5T_NATIVE_INT64);
    write_mat("recip_vectors", sc.lattice.get_reciprocal_lattice_vectors(), H5T_NATIVE_DOUBLE);

    H5Gclose(grp);
}


} // end namespace
