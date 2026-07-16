#pragma once

#include "MC.hpp"
#include "abstract_manager.hpp"
#include <lattice_lib/fourier.hpp>
#include "H5Apublic.h"
#include "H5Gpublic.h"
#include "H5Ipublic.h"
#include "H5Ppublic.h"
#include "H5Tpublic.h"
#include <algorithm>
#include <cassert>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>


// Scalar accessor for spin component at compile-time offset (0=x, 1=y, 2=z).
template<int Offset>
struct OffsetSpinAccessor {
    static_assert(Offset >= 0 && Offset < 3, "offset must be 0, 1 or 2");
    static double get(const CMC::HeisenbergSpin& s) { return s.S[Offset]; }
    static void   set(CMC::HeisenbergSpin& s, double v) { s.S[Offset] = v; }
};

using FT_Sx_t = FourierTransformC2C<CMC::HeisenbergSpin,
    OffsetSpinAccessor<0>, CMC::HeisenbergSpin>;
using FT_Sy_t = FourierTransformC2C<CMC::HeisenbergSpin,
    OffsetSpinAccessor<1>, CMC::HeisenbergSpin>;
using FT_Sz_t = FourierTransformC2C<CMC::HeisenbergSpin,
    OffsetSpinAccessor<2>, CMC::HeisenbergSpin>;


struct CorrSpec {
    int alpha, beta;   // 0=x, 1=y, 2=z
    std::string label; // e.g. "xy"
};


// Measures a user-specified set of spin component correlators, phase-folded
// over sublattice indices into a scalar S(q) per k-point.
//
// Accumulates, per temperature and per requested (α,β) pair:
//
//   S^{αβ}(q) = Σ_samples  Σ_{μ,ν} w_{μν}(q) · conj(Ã_μ^α(q)) · Ã_ν^β(q)
//
// where w_{μν}(q) = exp(+i q·(r_μ−r_ν)) is the sublattice structure-factor
// phase (SublatWeightMatrix::phase_factors).  The contraction over (μ,ν) is
// performed here, in C++, so the on-disk array is a factor of n_sl^2 smaller
// than storing the full sublattice-resolved correlator.
//
//   and the error term,
//   sq_S^{αβ}(q) = Σ_samples  |S^{αβ}(q)|^2 entrywise as real and imaginary parts
//
// Only one temperature's accumulators are held in memory at once.
//
// Two construction modes:
//
//   (A) Non-streaming (temper.cpp, fieldcool.cpp):
//       ssf_manager ssf(lat, {"xx","yy","zz"});
//       ssf.new_T(T); for (...) ssf.sample(); ssf.write_group(file_id, "/ssf");
//
//   (B) Streaming (anneal.cpp): file is opened upfront and static_corr is
//       pre-allocated on disk; each temperature's data is written and freed
//       after sampling, keeping only one temperature in memory at a time.
//       ssf_manager ssf(lat, {"xx","yy","zz"}, file_id, "/ssf", T_sample);
//       for each T: ssf.new_T(T); for (...) ssf.sample(); ssf.flush();
//       ssf.write_group(file_id, "/ssf");   // metadata only
//
// HDF5 output layout (write_group):
//   static_corr  [n_corr, n_T, n_k, 2]  — last dim = re/im (sublattice-contracted)
//   static_corr_2  [n_corr, n_T, n_k, 2]
//   corr_lookup  [n_corr]                             — VL strings, e.g. "xx", "xz"
//   T_list       [n_T]                                — temperatures, sorted ascending
//   n_samples    [n_T]
//   sl_positions [n_sl, 3]                            — int64 sublattice coords
//   attribute n_spins                                 — total spin count
//   attribute k_dims[3]                               — k-space grid dimensions
class ssf_manager : public abstract_manager {
    // Only the needed component FTs are constructed (avoids unnecessary FFTW plans).
    std::optional<FT_Sx_t> ft_x_;
    std::optional<FT_Sy_t> ft_y_;
    std::optional<FT_Sz_t> ft_z_;

    std::vector<CorrSpec> corr_specs_;

    // Sublattice structure-factor phase weights w_{μν}(q) = exp(+i q·(r_μ−r_ν)),
    // built once from the lattice + sublattice positions.  Used to contract each
    // sample's FourierCorrelator into a scalar S(q) before accumulation.
    std::optional<SublatWeightMatrix> phase_w_;

    // One temperature's phase-folded accumulators, indexed [corr][k].
    // Freed by flush() in streaming mode.
    //   curr_S_[c][k]    = Σ_samples S^{αβ}_c(q_k)
    //   curr_S_sq_[c][k] = Σ_samples |S^{αβ}_c(q_k)|^2  (re²/im² entrywise)
    std::vector<std::vector<std::complex<double>>> curr_S_;
    std::vector<std::vector<std::complex<double>>> curr_S_sq_;

    int n_sl_;
    int n_kpoints_;
    int n_spins_;
    ivec3_t k_dims_;
    std::vector<ipos_t> sl_positions_;

    const bool store_error_term_;

    // Streaming-mode HDF5 state. All handles are -1 in non-streaming mode.
    bool streaming_mode_ = false;
    hid_t grp_       = -1;
    hid_t sum_ds_    = -1;
    hid_t sum_sq_ds_ = -1;
    // T_sample values sorted ascending; index = HDF5 slot for that temperature.
    std::vector<double> T_sorted_;

    void alloc_curr_T() {
        const size_t nk = static_cast<size_t>(n_kpoints_);
        curr_S_.assign(corr_specs_.size(),
                       std::vector<std::complex<double>>(nk, 0.0));
        if (store_error_term_)
            curr_S_sq_.assign(corr_specs_.size(),
                              std::vector<std::complex<double>>(nk, 0.0));
        else
            curr_S_sq_.clear();
    }

    void on_new_temp() override {
        alloc_curr_T();
    }

    // Flatten curr_S_ (and curr_S_sq_) into contiguous double buffers.
    // Layout: [n_corr, n_k, 2]  (matches one T-slice of the HDF5 dataset).
    void flatten_current(std::vector<double>& buf,
                         std::vector<double>& sq_buf) const {
        const size_t nc = corr_specs_.size();
        const size_t nk = static_cast<size_t>(n_kpoints_);
        const size_t n_elem = nc * nk * 2;
        buf.resize(n_elem);
        if (store_error_term_) sq_buf.resize(n_elem);

        for (size_t c = 0; c < nc; ++c)
            for (size_t k = 0; k < nk; ++k) {
                const size_t base = c * nk + k;
                buf[base * 2]     = curr_S_[c][k].real();
                buf[base * 2 + 1] = curr_S_[c][k].imag();
                if (store_error_term_) {
                    sq_buf[base * 2]     = curr_S_sq_[c][k].real();
                    sq_buf[base * 2 + 1] = curr_S_sq_[c][k].imag();
                }
            }
    }

    // Write metadata (corr_lookup, T_list, n_samples, sl_positions, attributes)
    // to an already-open HDF5 group. T_list and n_samples are sorted ascending.
    void write_metadata(hid_t grp) const;

public:
    // Non-streaming constructor — for temper.cpp and fieldcool.cpp.
    // n_temperatures_reserve is accepted for API compatibility but unused.
    ssf_manager(CMC::Lattice& sc,
                const std::vector<std::string>& correlator_names,
                size_t /*n_temperatures_reserve*/ = 0,
                bool store_error_term = true)
        : n_sl_(static_cast<int>(
              std::get<SlPos<CMC::HeisenbergSpin>>(sc.sl_positions).size())),
          n_kpoints_(sc.lattice.num_primitive_cells()),
          n_spins_(static_cast<int>(sc.get_objects<CMC::HeisenbergSpin>().size())),
          k_dims_(sc.lattice.size()),
          store_error_term_(store_error_term)
    {
        for (const auto& slp : std::get<SlPos<CMC::HeisenbergSpin>>(sc.sl_positions)){
            sl_positions_.push_back(slp);
        }

        auto axis_idx = [](char c) -> int {
            if (c == 'x') return 0;
            if (c == 'y') return 1;
            if (c == 'z') return 2;
            throw std::invalid_argument(
                std::string("ssf_manager: unknown spin component '") + c + "'");
        };
        bool need[3] = {false, false, false};
        for (const auto& name : correlator_names) {
            if (name.size() != 2)
                throw std::invalid_argument(
                    "ssf_manager: correlator name must be 2 chars: " + name);
            int a = axis_idx(name[0]);
            int b = axis_idx(name[1]);
            corr_specs_.push_back({a, b, name});
            need[a] = true;
            need[b] = true;
        }
        if (need[0]) ft_x_.emplace(sc);
        if (need[1]) ft_y_.emplace(sc);
        if (need[2]) ft_z_.emplace(sc);

        // Precompute sublattice phase weights once for the sample() contraction.
        phase_w_.emplace(
            SublatWeightMatrix::phase_factors(sc.lattice, sl_positions_));
    }

    // Streaming constructor — for anneal.cpp.
    // Creates the HDF5 group and pre-allocates static_corr / static_corr_2 on
    // disk immediately, then flush() writes one temperature at a time so only
    // O(n_corr * n_k) memory is held per temperature instead of O(n_T * n_corr * n_k).
    // Sublattice indices are contracted away in sample(), so neither memory nor
    // disk carries the n_sl^2 factor.
    //
    // Pre-allocation with H5D_ALLOC_TIME_EARLY reserves the full file extent at
    // creation, so subsequent flush() calls write in-place without growing the
    // file — the preferred pattern on Lustre.
    ssf_manager(CMC::Lattice& sc,
                const std::vector<std::string>& correlator_names,
                hid_t file_id, const char* group_name,
                const std::vector<double>& T_sample,
                bool store_error_term = true);

    ~ssf_manager() {
        if (sum_ds_    >= 0) H5Dclose(sum_ds_);
        if (sum_sq_ds_ >= 0) H5Dclose(sum_sq_ds_);
        if (grp_       >= 0) H5Gclose(grp_);
    }

    void point_at(CMC::Lattice& sc2){
        if (ft_x_) ft_x_->point_at(sc2);
        if (ft_y_) ft_y_->point_at(sc2);
        if (ft_z_) ft_z_->point_at(sc2);
    }

    // Metadata accessors (for MPI merge in temper.cpp).
    int     get_n_sl()       const { return n_sl_; }
    int     get_n_kpoints()  const { return n_kpoints_; }
    ivec3_t get_k_dims()     const { return k_dims_; }
    int     get_n_spins()    const { return n_spins_; }
    size_t  get_n_corr()     const { return corr_specs_.size(); }
    size_t  get_n_samples_at(size_t t_idx) const { return n_samples.at(t_idx); }
    double  get_T_at(size_t t_idx)         const { return T_list.at(t_idx); }

    std::vector<std::string> get_corr_labels() const {
        std::vector<std::string> labels;
        for (const auto& s : corr_specs_) labels.push_back(s.label);
        return labels;
    }
    const std::vector<ipos_t>& get_sl_positions() const { return sl_positions_; }

    // Serialise current temperature's data into flat double arrays for MPI (temper.cpp).
    // Layout: [n_corr, n_k, 2].  t_idx must equal curr_idx.
    void get_flat_buffer(size_t t_idx,
                         std::vector<double>& corr_buf,
                         std::vector<double>& corr_sq_buf) const {
        assert(t_idx == curr_idx);
        (void)t_idx;
        assert(!curr_S_.empty());
        flatten_current(corr_buf, corr_sq_buf);
    }

    // Fourier-transform the current spin configuration, fold in the sublattice
    // phases, and accumulate the scalar S(q).
    void sample() {
        assert(!T_list.empty());
        assert(!curr_S_.empty() && "sample() called after flush() without new_T()");

        if (ft_x_) ft_x_->transform();
        if (ft_y_) ft_y_->transform();
        if (ft_z_) ft_z_->transform();

        auto buf = [&](int axis) -> const FourierBuffer<CMC::HeisenbergSpin>& {
            if (axis == 0) return ft_x_->get_buffer();
            if (axis == 1) return ft_y_->get_buffer();
            return ft_z_->get_buffer();
        };

        const size_t nk = static_cast<size_t>(n_kpoints_);
        for (size_t c = 0; c < corr_specs_.size(); ++c) {
            // C_{μν}(q) = conj(Ã_μ^α) Ã_ν^β, then contract to S(q) = Σ_{μν} w_{μν} C_{μν}.
            auto this_corr = correlate(buf(corr_specs_[c].alpha),
                    buf(corr_specs_[c].beta));
            const auto S_k = phase_w_->contract(this_corr);  // [n_k]

            for (size_t k = 0; k < nk; ++k)
                curr_S_[c][k] += S_k[k];

            if (store_error_term_) {
                for (size_t k = 0; k < nk; ++k)
                    curr_S_sq_[c][k] += std::complex<double>(
                        S_k[k].real() * S_k[k].real(),
                        S_k[k].imag() * S_k[k].imag());
            }
        }

        n_samples[curr_idx]++;
    }

    // Streaming mode only: write the current temperature's data to the pre-allocated
    // HDF5 dataset and release the correlator memory.  Call after each T_sample
    // block, before calling new_T() for the next temperature.
    void flush();

    // Write the SSF group to HDF5.
    //
    // Streaming mode: data already written by flush(); this call writes
    //   metadata only (T_list, n_samples, corr_lookup, sl_positions, attributes)
    //   and closes the open HDF5 handles.  file_id / group_name are ignored
    //   (the group was opened in the constructor).
    //
    // Non-streaming mode: creates the group and dataset, writes data for the
    //   current (last) temperature, and writes all metadata.
    void write_group(hid_t file_id, const char* group_name = "/ssf") override;
};


// ---------------------------------------------------------------------------
// Streaming constructor
// ---------------------------------------------------------------------------
inline ssf_manager::ssf_manager(CMC::Lattice& sc,
                                 const std::vector<std::string>& correlator_names,
                                 hid_t file_id, const char* group_name,
                                 const std::vector<double>& T_sample,
                                 bool store_error_term)
    : n_sl_(static_cast<int>(
          std::get<SlPos<CMC::HeisenbergSpin>>(sc.sl_positions).size())),
      n_kpoints_(sc.lattice.num_primitive_cells()),
      n_spins_(static_cast<int>(sc.get_objects<CMC::HeisenbergSpin>().size())),
      k_dims_(sc.lattice.size()),
      sl_positions_(
          std::get<SlPos<CMC::HeisenbergSpin>>(sc.sl_positions).begin(),
          std::get<SlPos<CMC::HeisenbergSpin>>(sc.sl_positions).end()),
      store_error_term_(store_error_term),
      streaming_mode_(true)
{
    auto axis_idx = [](char c) -> int {
        if (c == 'x') return 0;
        if (c == 'y') return 1;
        if (c == 'z') return 2;
        throw std::invalid_argument(
            std::string("ssf_manager: unknown spin component '") + c + "'");
    };
    bool need[3] = {false, false, false};
    for (const auto& name : correlator_names) {
        if (name.size() != 2)
            throw std::invalid_argument(
                "ssf_manager: correlator name must be 2 chars: " + name);
        int a = axis_idx(name[0]);
        int b = axis_idx(name[1]);
        corr_specs_.push_back({a, b, name});
        need[a] = true;
        need[b] = true;
    }
    if (need[0]) ft_x_.emplace(sc);
    if (need[1]) ft_y_.emplace(sc);
    if (need[2]) ft_z_.emplace(sc);

    // Precompute sublattice phase weights once for the sample() contraction.
    phase_w_.emplace(
        SublatWeightMatrix::phase_factors(sc.lattice, sl_positions_));

    T_sorted_ = T_sample;
    std::sort(T_sorted_.begin(), T_sorted_.end());

    const size_t n_T = T_sorted_.size();
    T_list.reserve(n_T);
    n_samples.reserve(n_T);

    // Open or create group
    if (H5Lexists(file_id, group_name, H5P_DEFAULT) > 0)
        grp_ = H5Gopen2(file_id, group_name, H5P_DEFAULT);
    else
        grp_ = H5Gcreate2(file_id, group_name,
                           H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (grp_ < 0)
        throw std::runtime_error(
            std::string("ssf_manager: failed to open/create group ") + group_name);

    // Pre-allocate static_corr[n_corr, n_T, n_k, 2].
    // H5D_ALLOC_TIME_EARLY reserves the full file extent at dataset creation so
    // flush() writes are purely in-place — no file growth during the MC run.
    const hsize_t nc = corr_specs_.size();
    const hsize_t nk = static_cast<hsize_t>(n_kpoints_);
    const hsize_t dims[4] = { nc, static_cast<hsize_t>(n_T), nk, 2 };

    hid_t fspace = H5Screate_simple(4, dims, nullptr);
    hid_t dcpl   = H5Pcreate(H5P_DATASET_CREATE);
    H5Pset_layout(dcpl, H5D_CONTIGUOUS);
    H5Pset_alloc_time(dcpl, H5D_ALLOC_TIME_EARLY);

    sum_ds_ = H5Dcreate2(grp_, "static_corr", H5T_NATIVE_DOUBLE, fspace,
                          H5P_DEFAULT, dcpl, H5P_DEFAULT);
    if (sum_ds_ < 0) {
        H5Pclose(dcpl); H5Sclose(fspace); H5Gclose(grp_); grp_ = -1;
        throw std::runtime_error("ssf_manager: failed to create static_corr");
    }

    if (store_error_term_) {
        sum_sq_ds_ = H5Dcreate2(grp_, "static_corr_2", H5T_NATIVE_DOUBLE, fspace,
                                 H5P_DEFAULT, dcpl, H5P_DEFAULT);
        if (sum_sq_ds_ < 0) {
            H5Pclose(dcpl); H5Sclose(fspace);
            H5Dclose(sum_ds_); sum_ds_ = -1;
            H5Gclose(grp_); grp_ = -1;
            throw std::runtime_error("ssf_manager: failed to create static_corr_2");
        }
    }

    H5Pclose(dcpl);
    H5Sclose(fspace);
}


// ---------------------------------------------------------------------------
// flush(): write current temperature to HDF5, free its correlator memory
// ---------------------------------------------------------------------------
inline void ssf_manager::flush() {
    assert(streaming_mode_ && "flush() called on non-streaming ssf_manager");
    assert(!T_list.empty());
    assert(!curr_S_.empty() && "flush() called with no accumulated data");

    // Map current temperature to its pre-sorted slot in the dataset.
    const double curr_T = T_list[curr_idx];
    auto it = std::find(T_sorted_.begin(), T_sorted_.end(), curr_T);
    if (it == T_sorted_.end())
        throw std::runtime_error(
            "ssf_manager::flush: current temperature not found in T_sample");
    const hsize_t slot = static_cast<hsize_t>(std::distance(T_sorted_.begin(), it));

    const hsize_t nc = static_cast<hsize_t>(corr_specs_.size());
    const hsize_t nk = static_cast<hsize_t>(n_kpoints_);

    // Flatten [n_corr, n_k, 2]
    std::vector<double> buf, sq_buf;
    flatten_current(buf, sq_buf);

    // Write one T-slice via hyperslab.
    // Each flush is nc * nk * 2 * 8 bytes — large sequential I/O on Lustre.
    const hsize_t start[4] = { 0, slot, 0, 0 };
    const hsize_t count[4] = { nc, 1,    nk, 2 };
    const hsize_t mem_dims  = nc * nk * 2;
    hid_t mem_sp  = H5Screate_simple(1, &mem_dims, nullptr);

    hid_t file_sp = H5Dget_space(sum_ds_);
    H5Sselect_hyperslab(file_sp, H5S_SELECT_SET, start, nullptr, count, nullptr);
    herr_t rc = H5Dwrite(sum_ds_, H5T_NATIVE_DOUBLE,
                          mem_sp, file_sp, H5P_DEFAULT, buf.data());
    H5Sclose(file_sp);

    if (store_error_term_) {
        file_sp = H5Dget_space(sum_sq_ds_);
        H5Sselect_hyperslab(file_sp, H5S_SELECT_SET, start, nullptr, count, nullptr);
        herr_t rc2 = H5Dwrite(sum_sq_ds_, H5T_NATIVE_DOUBLE,
                               mem_sp, file_sp, H5P_DEFAULT, sq_buf.data());
        H5Sclose(file_sp);
        if (rc2 < 0) rc = rc2;
    }

    H5Sclose(mem_sp);

    if (rc < 0)
        throw std::runtime_error("ssf_manager::flush: H5Dwrite failed");

    // Release accumulator memory; on_new_temp() will reallocate for the next T.
    curr_S_.clear();
    curr_S_.shrink_to_fit();
    curr_S_sq_.clear();
    curr_S_sq_.shrink_to_fit();
}


// ---------------------------------------------------------------------------
// Shared metadata writer
// ---------------------------------------------------------------------------
inline void ssf_manager::write_metadata(hid_t grp) const {
    const size_t n_T    = T_list.size();
    const size_t n_corr = corr_specs_.size();
    const hsize_t ns    = static_cast<hsize_t>(n_sl_);

    // Sort temperatures ascending (matching existing postprocessing expectation)
    std::vector<size_t> ord(n_T);
    std::iota(ord.begin(), ord.end(), 0);
    std::sort(ord.begin(), ord.end(),
              [&](size_t a, size_t b) { return T_list[a] < T_list[b]; });

    std::vector<double> sT(n_T);
    std::vector<size_t> sN(n_T);
    for (size_t i = 0; i < n_T; ++i) {
        sT[i] = T_list[ord[i]];
        sN[i] = n_samples[ord[i]];
    }

    auto write_1d = [&](const char* name, hid_t type, hsize_t len, const void* data) {
        hid_t sp = H5Screate_simple(1, &len, nullptr);
        hid_t ds = H5Dcreate2(grp, name, type, sp,
                               H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        if (ds < 0) { H5Sclose(sp); throw std::runtime_error(
            std::string("ssf_manager: failed to create ") + name); }
        H5Dwrite(ds, type, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
        H5Dclose(ds);
        H5Sclose(sp);
    };

    write_1d("T_list",    H5T_NATIVE_DOUBLE, n_T, sT.data());
    write_1d("n_samples", H5T_NATIVE_ULLONG, n_T, sN.data());

    // corr_lookup: variable-length UTF-8 strings
    {
        hid_t str_t = H5Tcopy(H5T_C_S1);
        H5Tset_size(str_t, H5T_VARIABLE);
        H5Tset_cset(str_t, H5T_CSET_UTF8);
        hsize_t nc = static_cast<hsize_t>(n_corr);
        hid_t sp = H5Screate_simple(1, &nc, nullptr);
        hid_t ds = H5Dcreate2(grp, "corr_lookup", str_t, sp,
                               H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        if (ds < 0) { H5Sclose(sp); H5Tclose(str_t);
            throw std::runtime_error("ssf_manager: failed to create corr_lookup"); }
        std::vector<const char*> ptrs(n_corr);
        for (size_t c = 0; c < n_corr; ++c) ptrs[c] = corr_specs_[c].label.c_str();
        H5Dwrite(ds, str_t, H5S_ALL, H5S_ALL, H5P_DEFAULT, ptrs.data());
        H5Dclose(ds);
        H5Sclose(sp);
        H5Tclose(str_t);
    }

    // sl_positions [n_sl, 3]
    {
        hsize_t dims[2] = { ns, 3 };
        hid_t sp = H5Screate_simple(2, dims, nullptr);
        hid_t ds = H5Dcreate2(grp, "sl_positions", H5T_NATIVE_INT64, sp,
                               H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        if (ds < 0) { H5Sclose(sp);
            throw std::runtime_error("ssf_manager: failed to create sl_positions"); }
        H5Dwrite(ds, H5T_NATIVE_INT64, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                 sl_positions_.data());
        H5Dclose(ds);
        H5Sclose(sp);
    }

    // Attributes: n_spins, k_dims
    {
        hid_t sp = H5Screate(H5S_SCALAR);
        hid_t at = H5Acreate2(grp, "n_spins", H5T_NATIVE_INT, sp,
                               H5P_DEFAULT, H5P_DEFAULT);
        H5Awrite(at, H5T_NATIVE_INT, &n_spins_);
        H5Aclose(at);
        H5Sclose(sp);
    }
    {
        hsize_t adim = 3;
        int kd[3] = { static_cast<int>(k_dims_[0]),
                      static_cast<int>(k_dims_[1]),
                      static_cast<int>(k_dims_[2]) };
        hid_t sp = H5Screate_simple(1, &adim, nullptr);
        hid_t at = H5Acreate2(grp, "k_dims", H5T_NATIVE_INT, sp,
                               H5P_DEFAULT, H5P_DEFAULT);
        H5Awrite(at, H5T_NATIVE_INT, kd);
        H5Aclose(at);
        H5Sclose(sp);
    }
}


// ---------------------------------------------------------------------------
// write_group
// ---------------------------------------------------------------------------
inline void ssf_manager::write_group(hid_t file_id, const char* group_name) {
    if (streaming_mode_) {
        // Flush any temperature that wasn't flushed explicitly.
        if (!curr_S_.empty() && !T_list.empty())
            flush();

        // Write metadata to the already-open group.
        write_metadata(grp_);

        // Close dataset and group handles; mark them invalid so the destructor skips them.
        if (sum_ds_    >= 0) { H5Dclose(sum_ds_);    sum_ds_    = -1; }
        if (sum_sq_ds_ >= 0) { H5Dclose(sum_sq_ds_); sum_sq_ds_ = -1; }
        if (grp_       >= 0) { H5Gclose(grp_);        grp_       = -1; }
        return;
    }

    // Non-streaming mode: create everything now and write current temperature's data.
    const size_t  n_T    = T_list.size();
    const size_t  n_corr = corr_specs_.size();
    const hsize_t nk     = static_cast<hsize_t>(n_kpoints_);

    hid_t grp;
    if (H5Lexists(file_id, group_name, H5P_DEFAULT) > 0)
        grp = H5Gopen2(file_id, group_name, H5P_DEFAULT);
    else
        grp = H5Gcreate2(file_id, group_name,
                         H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (grp < 0)
        throw std::runtime_error(
            std::string("ssf_manager: failed to open/create group ") + group_name);

    // Dataset shape: [n_corr, n_T, n_k, 2]
    const hsize_t dims[4] = { static_cast<hsize_t>(n_corr),
                               static_cast<hsize_t>(n_T),
                               nk, 2 };
    hid_t fspace = H5Screate_simple(4, dims, nullptr);

    hid_t sum_ds = H5Dcreate2(grp, "static_corr", H5T_NATIVE_DOUBLE, fspace,
                               H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    hid_t sum_sq_ds = -1;
    if (store_error_term_)
        sum_sq_ds = H5Dcreate2(grp, "static_corr_2", H5T_NATIVE_DOUBLE, fspace,
                                H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    if (sum_ds < 0 || (store_error_term_ && sum_sq_ds < 0)) {
        H5Sclose(fspace); H5Gclose(grp);
        throw std::runtime_error(
            "ssf_manager: failed to create static_corr or static_corr_2");
    }

    // Write curr_S_ into slot curr_idx of the dataset.
    std::vector<double> buf, sq_buf;
    flatten_current(buf, sq_buf);

    const hsize_t start[4] = { 0, static_cast<hsize_t>(curr_idx), 0, 0 };
    const hsize_t count[4] = { static_cast<hsize_t>(n_corr), 1, nk,  2 };
    const hsize_t mem_dims  = static_cast<hsize_t>(n_corr) * nk *  2;
    hid_t mem_sp = H5Screate_simple(1, &mem_dims, nullptr);

    hid_t file_sp = H5Dget_space(sum_ds);
    H5Sselect_hyperslab(file_sp, H5S_SELECT_SET, start, nullptr, count, nullptr);
    H5Dwrite(sum_ds, H5T_NATIVE_DOUBLE, mem_sp, file_sp, H5P_DEFAULT, buf.data());
    H5Sclose(file_sp);
    H5Dclose(sum_ds);

    if (store_error_term_) {
        file_sp = H5Dget_space(sum_sq_ds);
        H5Sselect_hyperslab(file_sp, H5S_SELECT_SET, start, nullptr, count, nullptr);
        H5Dwrite(sum_sq_ds, H5T_NATIVE_DOUBLE, mem_sp, file_sp, H5P_DEFAULT, sq_buf.data());
        H5Sclose(file_sp);
        H5Dclose(sum_sq_ds);
    }

    H5Sclose(mem_sp);
    H5Sclose(fspace);

    write_metadata(grp);
    H5Gclose(grp);
}
