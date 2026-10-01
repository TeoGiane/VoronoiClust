#pragma once

// Standard library includes
#include <vector>
#include <memory>
#include <random>
#include <algorithm>
#include <functional>
#include <chrono>
#include <unordered_map>
#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>

// Rcpp includes
#include <RcppArmadillo.h>
#include <progress.hpp>

// Local includes
#include "likelihoods.h"
#include "priors.h"
#include "sampler_types.h"
#include "mcmc_utils.h"
#include "sufficient_stats_cache.h"


// Callback types defintions (for multi-view extension)
// Full state evaluation (used for Split/Merge)
using FullCouplingCallback = std::function<double(const arma::uvec &)>;
// O(1) Gibbs evaluation: inputs are (obs_i, old_clust, new_clust)
using GibbsCouplingCallback = std::function<double(int, int, int)>;
// O(1) Gibbs state update: inputs are (obs_i, old_clust, new_clust)
using GibbsUpdateCallback = std::function<void(int, int, int)>;



class PYSampler {
  // Class members
  private:
    // Data
    arma::mat distance_matrix;
    size_t n_data;
    // Model definition
    std::shared_ptr<AbstractLikelihood> likelihood;
    std::shared_ptr<AbstractMixturePrior> prior;
    // Algorithm Parameters
    MixtureAlgorithmParams algo_params;
    // Internal MCMC state
    MixtureState curr_state;
	// Initial adaptive RWMH sds
	double discount_sd = 0.1;
	double concentration_sd = 0.1;
    // Random number generator
    std::mt19937 rng;
    // Incremental sufficient-statistics cache. Only populated when
    // `likelihood` is a QuadraticTessellationLikelihood -- for any other
    // likelihood type this stays nullptr and every code path below falls
    // back to the original full-recompute behaviour, so correctness for
    // unrecognized likelihoods is unaffected.
    std::unique_ptr<QuadraticStatsCache> stats_cache;
    // ---- Adaptive cache-rebuild schedule --------------------------------
    // A full O(n^2) rebuild corrects the floating-point drift accumulated by
    // repeated incremental add/subtract updates. Rather than firing on a
    // guessed fixed interval, the interval is DERIVED from the drift
    // actually measured at the previous rebuild, which costs nothing: the
    // pre-rebuild cached_lpdf and the post-rebuild one bracket exactly the
    // error accumulated since the last one.
    //
    // The budget is counted in commit_move() applications, not iterations,
    // because that is what drift accumulates with: a Gibbs sweep contributes
    // O(n) updates, a split/merge only O(cluster size).
    //
    // Relative error we are willing to carry before forcing a rebuild.
    double drift_target_rel = 1e-11;
    // Relative error at which sync_state_lpdf's debug check complains.
    double drift_warn_rel = 1e-9;
    // EWMA of measured relative drift PER incremental update. Negative
    // means "no measurement yet".
    double drift_rate_ewma = -1.0;
    double drift_rate_decay = 0.3;      // weight given to the newest measurement
    // Budget (in updates) until the next rebuild, plus its clamps. All three
    // are sized from n_data in init(); 0 means "no cache, never rebuild".
    long long rebuild_budget = 0;
    long long rebuild_budget_min = 0;
    long long rebuild_budget_max = 0;
    // Diagnostics, reported in debug mode.
    size_t n_rebuilds = 0;
    size_t n_drift_warnings = 0;
    double worst_rel_drift = 0.0;

  // Public class methods
  public:
    // Constructor & destructor
    PYSampler(const arma::mat & _distance_matrix,
						std::shared_ptr<AbstractLikelihood> _likelihood_ptr,
						std::shared_ptr<AbstractMixturePrior> _prior_ptr,
						const MixtureAlgorithmParams & _algo_params);
    ~PYSampler() = default;
    // Proper initialization method
    void init();
    // Step method (exposed for milti-view extension)
    void step(size_t curr_iter, FullCouplingCallback full_coupling_cb = nullptr, GibbsCouplingCallback gibbs_coupling_cb = nullptr,
              GibbsUpdateCallback gibbs_update_cb = nullptr); 
    // Main MCMC loop
    MixtureMCMCOutput run();
    // Getters for results
    MixtureState get_current_state() const { return curr_state; }
    // Overrides the seed used by init(). Needed in case of multi-view extensions
    void set_random_seed(unsigned int seed) { algo_params.random_seed = seed; }

  // Private class methods
  private:
    // MCMC Steps
    void split_step(int obs_i, int obs_j, FullCouplingCallback full_coupling_cb = nullptr);
    void merge_step(int obs_i, int obs_j, FullCouplingCallback full_coupling_cb = nullptr);
    void gibbs_step(GibbsCouplingCallback gibbs_coupling_cb = nullptr, GibbsUpdateCallback gibbs_update_cb = nullptr);
    void sample_discount(size_t curr_iter);
    void sample_concentration(size_t curr_iter);
    // Utilities
    void sync_state_lpdf(bool validate = false);
    // One-off, UNCONDITIONAL, throwing check that the cache's closed-form
    // terms encode the same model as likelihood->eval_lpdf(). Run once in
    // init() against a fresh rebuild: if these disagree, every delta and
    // every acceptance ratio in the run is built on a false premise.
    void validate_cache_against_likelihood() const;
    // Adaptive drift control (see the schedule members above).
    void maybe_rebuild_cache();
    void rebuild_cache_and_adapt();
    arma::uvec standardize_allocs(const arma::uvec & allocs) const;
    // The id_map implied by standardize_allocs (old id -> new id), used to
    // keep stats_cache's keys in sync when a merge closes an id gap.
    std::unordered_map<int,int> standardize_id_map(const arma::uvec & allocs) const;
    double restricted_gibbs_sweep(arma::uvec & allocs, const arma::uvec & members,
                    unsigned int obs_i, unsigned int obs_j,
                    unsigned int clust_i, unsigned int clust_j,
                    bool sample, const arma::uvec & target_allocs,
                    double base_lpdf,
                    QuadraticStatsTrialCache * trial_cache = nullptr);

};