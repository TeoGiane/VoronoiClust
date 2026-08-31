#pragma once

// Standard library includes
#include <vector>
#include <memory>
#include <random>
#include <chrono>
#include <utility>
#include <cmath>
#include <limits>

// Rcpp includes
#include <RcppArmadillo.h>
#include <progress.hpp>

// Local includes
#include "likelihoods.h"
#include "priors.h"
#include "sampler_types.h"
#include "sufficient_stats_cache.h"


class VoronoiSampler {
  // Class members
  private:
    // Data
    arma::mat distance_matrix;
    size_t n_data;
    // Model definition
    std::shared_ptr<AbstractLikelihood> likelihood;
    std::shared_ptr<AbstractPrior> prior;
    // Algorithm Parameters
    AlgorithmParams algo_params;
    // Internal MCMC state
    TessellationState curr_state;
    // Random number generator
    std::mt19937 rng;
    // Incremental sufficient-statistics cache (only one of these is ever
    // populated, depending on the concrete `likelihood` type; both stay
    // nullptr -- and every call site below falls back to the original
    // full-recompute behaviour -- for any other likelihood).
    //
    // IMPORTANT: both caches key clusters by the STABLE point-index of their
    // centre, never by the positional id (0..K-1) used in cluster_allocs.
    // Positional ids are just an index into the sorted `cluster_centres`
    // vector, so inserting/removing/moving a centre can silently renumber
    // every OTHER cluster whose centre sorts after the changed one -- centre
    // point indices don't have this problem, since a surviving centre's own
    // point index never changes regardless of what happens elsewhere.
    std::unique_ptr<QuadraticStatsCache> stats_cache;
    std::unique_ptr<LinearStatsCache> linear_cache;
    // cache_id_allocs(x) = the point-index of x's cluster's centre (i.e.
    // curr_state.cluster_centres[curr_state.cluster_allocs(x)]), kept in
    // sync with curr_state.cluster_allocs. This is what gets fed to the
    // caches' point_to_clusters/add_point/etc, so cluster identity is always
    // a stable point index, never a positional id that can be renumbered.
    arma::uvec cache_id_allocs;
    // dist_to_own_centre(x) = distance from point x to its own cluster's
    // centre, kept in sync alongside cache_id_allocs. Lets "would point x
    // prefer this candidate centre instead" be an O(1) check.
    arma::vec dist_to_own_centre;
    // ---- Adaptive cache-rebuild schedule (mirrors PYSampler) ------------
    // A full O(n) (linear cache) or O(n^2) (quadratic cache) rebuild
    // corrects floating-point drift accumulated by repeated incremental
    // add/subtract updates. The interval is DERIVED from the drift actually
    // measured at the previous rebuild (pre-rebuild lpdf vs. post-rebuild
    // lpdf bracket exactly the error accumulated since then), rather than
    // firing on a guessed fixed interval.
    //
    // The budget is counted in committed single-point moves (see
    // commit_movers / commit_linear), not iterations, since that is what
    // drift accumulates with: a birth/death/move proposal can touch anywhere
    // from 0 to O(n) points depending on how many points switch centre.
    double drift_target_rel = 1e-11;
    double drift_rate_ewma = -1.0;
    double drift_rate_decay = 0.3;
    long long updates_since_rebuild = 0;
    long long rebuild_budget = 0;
    long long rebuild_budget_min = 0;
    long long rebuild_budget_max = 0;
    // Diagnostics, reported in debug mode.
    size_t n_rebuilds = 0;
    double worst_rel_drift = 0.0;

  // Public class methods
  public:
    // Constructor & destructor
    VoronoiSampler(const arma::mat & _distance_matrix,
                   std::shared_ptr<AbstractLikelihood> _likelihood_ptr,
                   std::shared_ptr<AbstractPrior> _prior_ptr,
                   const AlgorithmParams & _algo_params);
    ~VoronoiSampler() = default;
    // Proper initialization method
    void init();
    // Main MCMC loop
    MCMCOutput run();
    // Public proposal generator (useful for MultiView version)
    TessellationProposal generate_proposal(size_t curr_iter);
    // Force apply a state update (useful for MultiView version)
    void apply_accepted_proposal(const TessellationProposal& prop);
    // Getters for results
    TessellationState get_current_state() const { return curr_state; };
    std::shared_ptr<AbstractPrior> get_prior() const { return prior; };

  // Private class methods
  private:
    // Utilities (maybe compute_tessellation as an external function?)
    arma::uvec compute_tessellation(const std::vector<arma::uword> & centres) const;
    arma::vec apply_tempering(const arma::vec& log_probs, double tempering, const arma::uvec& valid_idx) const;
    // Probability Generators
    arma::vec compute_birth_probs() const;
    arma::vec compute_death_probs() const;
    arma::vec compute_move_probs(int old_center_idx) const;
    // Proposal generators
    TessellationProposal generate_birth_proposal();
    TessellationProposal generate_death_proposal();
    TessellationProposal generate_move_proposal();
    // Proposal tester
    void test_proposal(const TessellationProposal & prop_state);
    // MCMC Step
    void step(size_t curr_iter);

    /* ---------------- Incremental sufficient-statistics support ---------------- */
    // A mover: point index -> the point index of the centre of its NEW cluster.
    using MoverList = std::vector<std::pair<arma::uword, arma::uword>>;

    // Sets up stats_cache / linear_cache (whichever matches `likelihood`),
    // cache_id_allocs and dist_to_own_centre from the current state. Called
    // once from init(), and again after any accepted proposal.
    void refresh_cache_state();
    // O(n) rebuild of cache_id_allocs + dist_to_own_centre from the current
    // curr_state (cluster_allocs / cluster_centres). Does NOT touch the
    // sufficient-statistics caches themselves.
    void sync_cache_ids();
    // Adaptive drift control (see the schedule members above). Checks the
    // committed-update budget and, if exhausted, rebuilds + re-derives the
    // next budget from the measured drift.
    void maybe_rebuild_cache();
    void rebuild_cache_and_adapt();

    // Derives every changed point from the proposed positional allocations.
    // Mapping each positional id through `prop_centres` makes this exactly
    // match compute_tessellation(), including its tie and zero-distance
    // behaviour. With no centres the baseline represents the single
    // placeholder allocation as label 0, so do the same in the cache.
    MoverList movers_for_proposal(const arma::uvec& prop_cluster_allocs,
                    const std::vector<arma::uword>& prop_centres) const;

    // Scores `movers` against the current cache state WITHOUT mutating it
    // (quadratic: via a scoped QuadraticStatsTrialCache; linear: via
    // snapshot/restore), returning the total lpdf delta. `scratch_cache_ids`
    // must start equal to cache_id_allocs and is left reflecting the movers
    // applied (needed by callers that go on to commit the same movers).
    double score_movers(const MoverList& movers, arma::uvec& scratch_cache_ids) const;
    // Linear-likelihood equivalent of score_movers: scores `movers` (plus the
    // resulting between-term) against a throwaway COPY of linear_cache,
    // leaving the persistent one untouched.
    double score_linear(const MoverList& movers, const std::vector<arma::uword>& new_centres, arma::uvec& scratch_cache_ids) const;
    // Commits `movers` for real: mutates stats_cache and cache_id_allocs /
    // dist_to_own_centre, returning the total lpdf delta (same value
    // score_movers would have returned, computed once here instead of twice).
    double commit_movers(const MoverList& movers);
    // Linear-likelihood equivalent of commit_movers. Unlike the quadratic
    // path, this is NOT used for the tempered candidate-scoring loops (which
    // still use the original full-recompute path for LinearTessellationLikelihood)
    // -- only for committing the one actually-accepted proposal, where its
    // O(1)-per-point within-term updates plus an O(K^2) between-term
    // recompute are still a large win over a full O(n) tessellation + O(n*K^2)
    // likelihood recompute.
    void commit_linear(const MoverList& movers, const std::vector<arma::uword>& new_centres);
};