#pragma once

#include "MC.hpp"


/**
 * Factory function: builds a UnitCellSpecifier for the pyrochlore spin lattice.
 *
 * 16 spin sublattices = 4 FCC sites × 4 link directions.
 * Sublattice ordering: pyrochlore_sl = latlib_sl % 4.
 */
inline UnitCellSpecifier<CMC::HeisenbergSpin> PyroCubicCell() {
    UnitCellSpecifier<CMC::HeisenbergSpin> spec(
        imat33_t::from_cols({8,0,0}, {0,8,0}, {0,0,8}));

    static constexpr ipos_t R_fcc[4] = {
        {0,0,0}, {0,4,4}, {4,0,4}, {4,4,0}
    };
    static constexpr ipos_t link_positions[4] = {
        {1,1,1}, {1,-1,-1}, {-1,1,-1}, {-1,-1,1}
    };

    for (int fcc = 0; fcc < 4; fcc++) {
        for (int p = 0; p < 4; p++) {
            CMC::HeisenbergSpin spin;
            spin.ipos = link_positions[p] + R_fcc[fcc];
            spin.pyro_sl = p;
            spec.add(std::move(spin));
        }
    }
    return spec;
}


inline UnitCellSpecifier<CMC::HeisenbergSpin> PrimitiveCell() {
    UnitCellSpecifier<CMC::HeisenbergSpin> spec(
        imat33_t::from_cols({8,0,0}, {0,8,0}, {0,0,8}));

    static constexpr ipos_t link_positions[4] = {
        {1,1,1}, {1,-1,-1}, {-1,1,-1}, {-1,-1,1}
    };

    for (int p = 0; p < 4; p++) {
        CMC::HeisenbergSpin spin;
        spin.ipos = link_positions[p];
        spin.pyro_sl = p;
        spec.add(std::move(spin));
    }
    return spec;
}



namespace pyrochlore {
    using idx3_t = ipos_t;

static const idx3_t pyro[4] = {
    {1, 1, 1}, {1, -1, -1}, {-1, 1, -1}, {-1, -1, 1}
};

static const idx3_t fcc_Dy[4] = {
    {0,0,0},
    {0,4,4},
    {4,0,4},
    {4,4,0}
};

static const idx3_t fcc_Ti[4] = {
    {4,4,4},
    {4,0,0},
    {0,4,0},
    {0,0,4}};

// Vectors from a plaquette centre to its six vertices
static const idx3_t plaqt[4][6] = {
    {
        { 0,-2, 2},
        { 2,-2, 0},
        { 2, 0,-2},
        { 0, 2,-2},
        {-2, 2, 0},
        {-2, 0, 2}
    },
    {
        { 0, 2,-2},
        { 2, 2, 0},
        { 2, 0, 2},
        { 0,-2, 2},
        {-2,-2, 0},
        {-2, 0,-2}
    },
    {
        { 0,-2,-2},
        {-2,-2, 0},
        {-2, 0, 2},
        { 0, 2, 2},
        { 2, 2, 0},
        { 2, 0,-2}
    },
    {
        { 0, 2, 2},
        {-2, 2, 0},
        {-2, 0,-2},
        { 0,-2,-2},
        { 2,-2, 0},
        { 2, 0, 2}
    }
};

static const std::vector<std::vector<idx3_t>> nn1_dist = {
    {{0, -2, -2}, {-2, 0, -2}, {-2, -2, 0}, {0, 2, 2},   {2, 0, 2},   {2, 2, 0}},
    {{-2, 2, 0},  {-2, 0, 2},  {0, 2, 2},   {2, -2, 0},  {2, 0, -2},  {0, -2, -2}},
    {{0, -2, 2},  {2, 0, 2},   {2, -2, 0},  {0, 2, -2},  {-2, 0, -2}, {-2, 2, 0}},
    {{2, 2, 0},   {2, 0, -2},  {0, 2, -2},  {-2, -2, 0}, {-2, 0, 2},  {0, -2, 2}}
};

static const std::vector<std::vector<idx3_t>> nn2_dist = {
    {
        {-2, 4, 2}, {-2, 2, 4}, {4, -2, 2}, {2, -2, 4}, {4, 2, -2}, {2, 4, -2}, 
        {2, -4, -2}, {2, -2, -4}, {-4, 2, -2}, {-2, 2, -4}, {-4, -2, 2}, {-2, -4, 2}
    }, 
    {
        {2, -4, 2}, {4, -2, 2}, {2, 2, -4}, {4, 2, -2}, {-2, -2, -4}, {-2, -4, -2}, 
        {-2, 4, -2}, {-4, 2, -2}, {-2, -2, 4}, {-4, -2, 2}, {2, 2, 4}, {2, 4, 2}
    }, 
    {
        {2, 4, -2}, {2, 2, -4}, {-4, -2, -2}, {-2, -2, -4}, {-4, 2, 2}, {-2, 4, 2},
        {-2, -4, 2}, {-2, -2, 4}, {4, 2, 2}, {2, 2, 4}, {4, -2, -2}, {2, -4, -2}
    }, 
    {
        {-2, -4, -2}, {-4, -2, -2}, {-2, 2, 4}, {-4, 2, 2}, {2, -2, 4}, {2, -4, 2},
        {2, 4, 2}, {4, 2, 2}, {2, -2, -4}, {4, -2, -2}, {-2, 2, -4}, {-2, 4, -2}
    }
};

// 3rd neighbours WITH spin in the middle
static const std::vector<std::vector<idx3_t>> nn3a_dist = {
    {{0, 4, 4}, {4, 0, 4}, {4, 4, 0}, {0, -4, -4}, {-4, 0, -4}, {-4, -4, 0}},
    {{4, -4, 0}, {4, 0, -4}, {0, -4, -4}, {-4, 4, 0}, {-4, 0, 4}, {0, 4, 4}},
    {{0, 4, -4}, {-4, 0, -4}, {-4, 4, 0}, {0, -4, 4}, {4, 0, 4}, {4, -4, 0}},
    {{-4, -4, 0}, {-4, 0, 4}, {0, -4, 4}, {4, 4, 0}, {4, 0, -4}, {0, 4, -4}}
};

static const std::vector<std::vector<idx3_t>> nn3b_dist = {
    {{-4, 4, 0}, {4, 0, -4}, {0, -4, 4}, {4, -4, 0}, {-4, 0, 4}, {0, 4, -4}},
    {{0, -4, 4}, {-4, 0, -4}, {4, 4, 0}, {0, 4, -4}, {4, 0, 4}, {-4, -4, 0}},
    {{4, 4, 0}, {-4, 0, 4}, {0, -4, -4}, {-4, -4, 0}, {4, 0, -4}, {0, 4, 4}},
    {{0, -4, -4}, {4, 0, 4}, {-4, 4, 0}, {0, 4, 4}, {-4, 0, -4}, {4, -4, 0}}
};

// nn1 bonds split by pyrochlore sublattice pair (μ,ν), μ < ν.
// nn1_pair_MN[pyro_sl] lists displacement vectors from a spin of pyro_sl
// to its nn1 neighbours of the opposite sublattice in the pair.
// Only the two involved sublattices have non-empty entries.
static const std::vector<std::vector<idx3_t>> nn1_pair_01 = {
    {{ 0,-2,-2}, { 0, 2, 2}},  // pyro_sl=0 → pyro_sl=1
    {{ 0, 2, 2}, { 0,-2,-2}},  // pyro_sl=1 → pyro_sl=0
    {},
    {}
};
static const std::vector<std::vector<idx3_t>> nn1_pair_02 = {
    {{-2, 0,-2}, { 2, 0, 2}},  // pyro_sl=0 → pyro_sl=2
    {},
    {{ 2, 0, 2}, {-2, 0,-2}},  // pyro_sl=2 → pyro_sl=0
    {}
};
static const std::vector<std::vector<idx3_t>> nn1_pair_03 = {
    {{-2,-2, 0}, { 2, 2, 0}},  // pyro_sl=0 → pyro_sl=3
    {},
    {},
    {{ 2, 2, 0}, {-2,-2, 0}}   // pyro_sl=3 → pyro_sl=0
};
static const std::vector<std::vector<idx3_t>> nn1_pair_12 = {
    {},
    {{-2, 2, 0}, { 2,-2, 0}},  // pyro_sl=1 → pyro_sl=2
    {{ 2,-2, 0}, {-2, 2, 0}},  // pyro_sl=2 → pyro_sl=1
    {}
};
static const std::vector<std::vector<idx3_t>> nn1_pair_13 = {
    {},
    {{-2, 0, 2}, { 2, 0,-2}},  // pyro_sl=1 → pyro_sl=3
    {},
    {{ 2, 0,-2}, {-2, 0, 2}}   // pyro_sl=3 → pyro_sl=1
};
static const std::vector<std::vector<idx3_t>> nn1_pair_23 = {
    {},
    {},
    {{ 0,-2, 2}, { 0, 2,-2}},  // pyro_sl=2 → pyro_sl=3
    {{ 0, 2,-2}, { 0,-2, 2}}   // pyro_sl=3 → pyro_sl=2
};

// Square roots of 2, 3, and 6 for normalisation
#define S2 1.414213562373095048801688724209698078569671875376948073176
#define S3 1.732050807568877293527446341505872366942805253810380628055
#define S6 2.449489742783178098197284074705891391965947480656670128432

using vec3d=vector3::vec3<double>;

static const vec3d axis[4][3] = {
    {(1. / S6) * vec3d(1, 1, -2), (1. / S2) * vec3d(-1, 1, 0),
     (1. / S3) * vec3d(1, 1, 1)},
    {(1. / S6) * vec3d(1, -1, 2), (1. / S2) * vec3d(-1, -1, 0),
     (1. / S3) * vec3d(1, -1, -1)},
    {(1. / S6) * vec3d(-1, 1, 2), (1. / S2) * vec3d(1, 1, 0),
     (1. / S3) * vec3d(-1, 1, -1)},
    {(1. / S6) * vec3d(-1, -1, -2), (1. / S2) * vec3d(1, -1, 0),
     (1. / S3) * vec3d(-1, -1, 1)}
};

} // end namespace pyrochlore
