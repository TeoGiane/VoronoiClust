#include "py_sampler.h"

// File-local helper: relabels a single point m from `from` to `to` in
// `allocs`, and -- if a trial cache is supplied -- threads the SAME move
// through it via point_to_clusters/commit_move first, WITHOUT accumulating
// any log-probability. This keeps the trial cache's internal stats in sync
// with the raw allocation vector for state-only relabeling steps (random
// launch-state initialization, forcing an anchor point back to its original
// cluster) that must not contribute to the split/merge proposal probability
// but still need to be reflected in every subsequent cache-based likelihood
// evaluation. No-op (state-wise) when from == to.
static void relabel_point(QuadraticStatsTrialCache * trial, const arma::mat & dist_matrix,
                           arma::uvec & allocs, arma::uword m, int from, int to) {
    if (trial && from != to) {
        auto agg = trial->point_to_clusters(dist_matrix, allocs, m);
        trial->commit_move(agg, from, to);
    }
    allocs(m) = to;
}

PYSampler::PYSampler(const arma::mat & _distance_matrix, std::shared_ptr<AbstractLikelihood> _likelihood_ptr, std::shared_ptr<AbstractMixturePrior> _prior_ptr, const MixtureAlgorithmParams & _algo_params): distance_matrix(_distance_matrix), likelihood(std::move(_likelihood_ptr)), prior(std::move(_prior_ptr)), algo_params(_algo_params) {};

MixtureMCMCOutput PYSampler::run() {
    // Initialize the sampler
    this->init();
    // Deduce retained samples + checks
    if (algo_params.thinning == 0) {
        throw std::invalid_argument("Thinning must be at least 1.");
    }
    if (algo_params.burnin >= algo_params.iterations) {
        throw std::invalid_argument("Burnin is greater than iterations or thinning is too large.");
    }
    const auto post_burnin = algo_params.iterations - algo_params.burnin;
    int n_retained = static_cast<int>((post_burnin + algo_params.thinning - 1) / algo_params.thinning);
    if (n_retained <= 0) {
        throw std::invalid_argument("Burnin is greater than iterations or thinning is too large.");
    }
    // Pre-allocate output structures
    MixtureMCMCOutput out;
    out.cluster_allocs.set_size(n_retained, n_data);
    out.n_clust.set_size(n_retained);
    out.discount.set_size(n_retained);
    out.concentration.set_size(n_retained);
    out.lpdf.set_size(n_retained);
    out.iteration_time.set_size(n_retained);
    // Print inital message (if not in debug mode)
    if (!algo_params.debug) {
        Rcpp::Rcout << "VoronoiClust: Pitman-Yor MCMC (" << algo_params.iterations << " iterations)" << std::endl;
    }
    // Initialize progress bar and save_idx
    Progress prog_bar(algo_params.iterations, !algo_params.debug);
    int save_idx = 0;
    // Main MCMC loop
    for (size_t i = 0; i < algo_params.iterations; ++i) {
        // Check for user interrupt
        if (Progress::check_abort()) {
            Rcpp::Rcout << "\nSampling interrupted by user. Returning available samples..." << "\n";
            // Truncate matrices to save_idx so you don't return blocks of zeros (guarded)
            if (save_idx < n_retained) {
                out.cluster_allocs.shed_rows(save_idx, n_retained - 1);
                out.n_clust.shed_rows(save_idx, n_retained - 1);
                out.discount.shed_rows(save_idx, n_retained - 1);
                out.concentration.shed_rows(save_idx, n_retained - 1);
                out.lpdf.shed_rows(save_idx, n_retained - 1);
                out.iteration_time.shed_rows(save_idx, n_retained - 1);
            }
            break;
        }
        // Single MCMC step (Jain & Neal 2004 algorithm)
        auto start = std::chrono::high_resolution_clock::now();
        this->step(i);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        // Store output if past burn-in and respecting thinning
        if (i >= algo_params.burnin && (i - algo_params.burnin) % algo_params.thinning == 0) {
            out.cluster_allocs.row(save_idx) = curr_state.cluster_allocs.t();
            out.n_clust(save_idx) = curr_state.n_clust;
            out.discount(save_idx) = curr_state.discount;
            out.concentration(save_idx) = curr_state.concentration;
            out.lpdf(save_idx) = curr_state.lpdf;
            out.iteration_time(save_idx) = elapsed.count();
            save_idx++;
        }
        // Increment the progress bar
        prog_bar.increment();
        // Debug log
        if (algo_params.debug) {
            Rcpp::Rcout << "Iter: " << i << std::endl;
            Rcpp::Rcout << " | K: " << curr_state.n_clust << std::endl;
            Rcpp::Rcout << " | Log-Lik: " << curr_state.lpdf << std::endl;
        }
    }
    // Return (truncated) output
    return out;
};

void PYSampler::init() {
    // Debug log
    if (algo_params.debug) { Rcpp::Rcout << "init()" << std::endl; }
    // Set n_data based on the distance matrix
    n_data = distance_matrix.n_rows;
    // Input validation
    if (distance_matrix.n_rows != distance_matrix.n_cols) {
        throw std::invalid_argument("PYSampler: distance matrix must be square.");
    }
    if (n_data < 2) {
        throw std::invalid_argument("PYSampler: need at least 2 observations.");
    }
    if (!distance_matrix.is_finite()) {
        throw std::invalid_argument("PYSampler: distance matrix contains NaN or Inf entries.");
    }
    if (distance_matrix.min() < 0.0) {
        throw std::invalid_argument("PYSampler: distance matrix contains negative entries.");
    }
    // The caches read only the upper triangle (rebuild) or a single row
    // (point_to_clusters), so an asymmetric input would be silently
    // interpreted inconsistently between the two.
    if (arma::abs(distance_matrix - distance_matrix.t()).max()
        > 1e-10 * std::max(1.0, distance_matrix.max())) {
        throw std::invalid_argument("PYSampler: distance matrix is not symmetric.");
    }
    // n_sweeps is used as `n_sweeps - 1`: check accordingly
    if (algo_params.n_sweeps < 1) {
        throw std::invalid_argument("PYSampler: n_sweeps must be at least 1.");
    }
    // Set seed for reproducibility
    rng.seed(algo_params.random_seed);
    // Check: initital number of cluster must be at most n_data
    unsigned int k = algo_params.init_n_clust;
    if (k > n_data) {
        k = n_data;
    }
    // Random initial allocation
    std::uniform_int_distribution<arma::uword> ui_dist(0, k > 0 ? k - 1 : 0);
    curr_state.cluster_allocs = arma::zeros<arma::uvec>(n_data);
    for(size_t i = 0; i < n_data; ++i) {
        curr_state.cluster_allocs(i) = ui_dist(rng);
    }
    // Set current state
    curr_state.cluster_allocs = standardize_allocs(curr_state.cluster_allocs);
    curr_state.n_clust = arma::unique(curr_state.cluster_allocs).eval().n_elem;
    auto py_prior = std::dynamic_pointer_cast<PYFixedPrior>(prior);
    if (py_prior) {
        curr_state.discount = py_prior->get_discount();
        curr_state.concentration = py_prior->get_concentration();
    }
    curr_state.lpdf = likelihood->eval_lpdf(distance_matrix, curr_state.cluster_allocs);
    // Set up the incremental sufficient-statistics cache if the likelihood supports it.
    auto quad_lik = std::dynamic_pointer_cast<QuadraticTessellationLikelihood>(likelihood);
    if (quad_lik) {
        stats_cache = std::make_unique<QuadraticStatsCache>(quad_lik->get_params());
        stats_cache->rebuild(distance_matrix, curr_state.cluster_allocs);
        // Cache validation agains independent implementation
        validate_cache_against_likelihood();
        // The rebuild above is ground truth, no extra eval_lpdf to call.
        sync_state_lpdf(false);
        // Seed the adaptive rebuild schedule
        rebuild_budget_min = std::max<long long>(1, static_cast<long long>(n_data));
        rebuild_budget_max = 200 * rebuild_budget_min;
        rebuild_budget     = std::min(rebuild_budget_max, 10 * rebuild_budget_min);
    }
    // Initialization complete
    if (algo_params.debug) { curr_state.print(); }
    return;
};

void PYSampler::split_step(int obs_i, int obs_j, FullCouplingCallback full_coupling_cb) {
    // Debug log
    if (algo_params.debug) { Rcpp::Rcout << "split_step()" << std::endl; }
    // Prepare prop_allocs buffer
    arma::uvec prop_allocs = curr_state.cluster_allocs;
    // Select elements involved in the split
    int target_clust = prop_allocs(obs_i);
    int new_clust_id = curr_state.n_clust;
    arma::uvec members = arma::find(prop_allocs == target_clust);
    // Scoped trial cache: scores the launch-state construction AND the
    // restricted Gibbs sweeps against the persistent stats_cache without
    // mutating it, so a rejected split costs nothing beyond this function's
    // stack frame. Falls back to nullptr (full recompute) when the
    // likelihood isn't cache-able.
    std::unique_ptr<QuadraticStatsTrialCache> trial;
    if (stats_cache) trial = std::make_unique<QuadraticStatsTrialCache>(*stats_cache);
    // Initial random split (launch-state construction)
    prop_allocs(obs_i) = target_clust;
    relabel_point(trial.get(), distance_matrix, prop_allocs, obs_j, target_clust, new_clust_id);
    std::uniform_real_distribution<double> runif(0.0, 1.0);
    for (size_t idx = 0; idx < members.n_elem; ++idx) {
        int m = members(idx);
        if (m != obs_i && m != obs_j) {
            int chosen = (runif(rng) < 0.5) ? target_clust : new_clust_id;
            relabel_point(trial.get(), distance_matrix, prop_allocs, m, target_clust, chosen);
        }
    }
    // Intermediate Gibbs sweeps (t-1 sweeps), clamped
    const int n_extra_sweeps = std::max(0, static_cast<int>(algo_params.n_sweeps) - 1);
    for (int s = 0; s < n_extra_sweeps; ++s) {
        restricted_gibbs_sweep(prop_allocs, members, obs_i, obs_j, target_clust, new_clust_id, true, prop_allocs, curr_state.lpdf, trial.get());
    }
    // Final sweep to accumulate proposal probability
    double log_q_split = restricted_gibbs_sweep(prop_allocs, members, obs_i, obs_j, target_clust, new_clust_id, true, prop_allocs, curr_state.lpdf, trial.get());
    // No standardization here, deliberately. Safe by construction
    int prop_n_clust = curr_state.n_clust + 1;
    if (algo_params.debug) {
        // The invariant the above rests on, checked where it is cheap to.
        const arma::uvec uids = arma::unique(prop_allocs);
        if (static_cast<int>(uids.n_elem) != prop_n_clust ||
            uids(0) != 0 || uids(uids.n_elem - 1) != static_cast<arma::uword>(curr_state.n_clust)) {
            Rcpp::Rcout << "Warning: split_step produced non-contiguous cluster ids; "
                        << "the no-standardization fast path is no longer valid." << std::endl;
        }
    }
    double prop_lpdf = stats_cache ? (curr_state.lpdf + trial->delta_lpdf()) : likelihood->eval_lpdf(distance_matrix, prop_allocs);
    // Evaluate couplings callbacks (only if this class is used in  multi-view extension)
    double curr_coupling = full_coupling_cb ? full_coupling_cb(curr_state.cluster_allocs) : 0.0;
    double prop_coupling = full_coupling_cb ? full_coupling_cb(prop_allocs) : 0.0;
    // Compute acceptance ratio (reverse merge is deterministic, so log_q_merge = 0)
    double log_arate = (prop_lpdf - prop_coupling) - (curr_state.lpdf - curr_coupling) +
        prior->eval_lpdf(prop_allocs) - prior->eval_lpdf(curr_state.cluster_allocs) - log_q_split;
    // Test for acceptance
    if(std::log(runif(rng)) < log_arate) {
        if (algo_params.debug) { Rcpp::Rcout << "Split accepted" << std::endl;}
        curr_state.cluster_allocs = std::move(prop_allocs);
        curr_state.n_clust = prop_n_clust;
        if (stats_cache) {
            // No remap_ids: a split introduces a brand-new id and moves no
            // existing one, so every cache key stays valid by construction.
            trial->fold_into(*stats_cache);
            sync_state_lpdf(algo_params.debug);
        } else {
            curr_state.lpdf = prop_lpdf;
        }
    } else {
        if (algo_params.debug) { Rcpp::Rcout << "Split rejected" << std::endl;}
    }
    // Drift control and cache rebuild
    maybe_rebuild_cache();
    // Debug log
    if (algo_params.debug) { curr_state.print(); }
    return;
};

void PYSampler::merge_step(int obs_i, int obs_j, FullCouplingCallback full_coupling_cb) {
    // Debug log
    if (algo_params.debug) { Rcpp::Rcout << "merge_step()" << std::endl; }
    // Prepare allocation buffer
    arma::uvec prop_allocs = curr_state.cluster_allocs;
    // Select elements involved in the merge
    int clust_i = curr_state.cluster_allocs(obs_i);
    int clust_j = curr_state.cluster_allocs(obs_j);
    arma::uvec members_j = arma::find(prop_allocs == clust_j);
    // Generate merge proposal
    prop_allocs.elem(members_j).fill(clust_i);
    // NO standardization here. We can tolerate absent labels.
    // Relabeling is therefore deferred to the acceptance branch, where it is actually needed,
    int prop_n_clust = curr_state.n_clust - 1;
    // Scoped trial cache scoring the actual merge itself (clust_j's members
    // reassigned to clust_i). This one -- and ONLY this one -- gets folded
    // into the persistent cache on acceptance, since it's the only one that
    // reflects the real accepted state.
    std::unique_ptr<QuadraticStatsTrialCache> merge_trial;
    double prop_lpdf;
    if (stats_cache) {
        merge_trial = std::make_unique<QuadraticStatsTrialCache>(*stats_cache);
        arma::uvec scratch_allocs = curr_state.cluster_allocs;
        for (size_t idx = 0; idx < members_j.n_elem; ++idx) {
            arma::uword m = members_j(idx);
            relabel_point(merge_trial.get(), distance_matrix, scratch_allocs, m, clust_j, clust_i);
        }
        prop_lpdf = curr_state.lpdf + merge_trial->delta_lpdf();
    } else {
        prop_lpdf = likelihood->eval_lpdf(distance_matrix, prop_allocs);
    }
    // Evaluate couplings (only if this class is used in multi-view extension)
    double curr_coupling = full_coupling_cb ? full_coupling_cb(curr_state.cluster_allocs) : 0.0;
    double prop_coupling = full_coupling_cb ? full_coupling_cb(standardize_allocs(prop_allocs)) : 0.0;
    // Compute reverse probability (splitting the merged state back into exact curr_state).
    arma::uvec merged_members = arma::find(prop_allocs == clust_i);
    arma::uvec dummy_allocs = prop_allocs;
    std::unique_ptr<QuadraticStatsTrialCache> reverse_trial;
    if (stats_cache) {
        // Explicit base cast: layers a FRESH overlay on top of merge_trial
        reverse_trial = std::make_unique<QuadraticStatsTrialCache>(
            static_cast<const QuadraticStatsCache &>(*merge_trial));
    }
    dummy_allocs(obs_i) = clust_i;
    relabel_point(reverse_trial.get(), distance_matrix, dummy_allocs, obs_j, clust_i, clust_j);
    // Initial random split (launch-state construction
    std::uniform_real_distribution<double> runif(0.0, 1.0);
    for (size_t idx = 0; idx < merged_members.n_elem; ++idx) {
        int m = merged_members(idx);
        if (m != obs_i && m != obs_j) {
            int chosen = (runif(rng) < 0.5) ? clust_i : clust_j;
            relabel_point(reverse_trial.get(), distance_matrix, dummy_allocs, m, clust_i, chosen);
        }
    }
    // Absolute lpdf of the ALREADY-MERGED state, which is the baseline the
    // reverse sweeps score against. merge_trial is null whenever the
    // likelihood isn't cache-able, and this is precisely the fallback path
    // the whole trial machinery is supposed to degrade gracefully into --
    // dereferencing it unconditionally crashed every merge attempt for any
    // non-quadratic likelihood.
    const double merged_base_lpdf = curr_state.lpdf
                                  + (merge_trial ? merge_trial->delta_lpdf() : 0.0);
    // Intermediate sweeps (t-1 sweeps), clamped
    const int n_extra_sweeps = std::max(0, static_cast<int>(algo_params.n_sweeps) - 1);
    for (int s = 0; s < n_extra_sweeps; ++s) {
        restricted_gibbs_sweep(dummy_allocs, merged_members, obs_i, obs_j, clust_i, clust_j, true, dummy_allocs, merged_base_lpdf, reverse_trial.get());
    }
    // Final sweep: do not sample, but force transition into curr_state and calculate probability
    double log_q_split_rev = restricted_gibbs_sweep(dummy_allocs, merged_members, obs_i, obs_j, clust_i, clust_j, false, curr_state.cluster_allocs, merged_base_lpdf, reverse_trial.get());
    // Compute acceptance ratio (forward merge is deterministic, so log_q_merge = 0)
    double log_arate = (prop_lpdf - prop_coupling) - (curr_state.lpdf - curr_coupling) +
        prior->eval_lpdf(prop_allocs) - prior->eval_lpdf(curr_state.cluster_allocs) + log_q_split_rev;
    // Test for acceptance
    if(std::log(runif(rng)) < log_arate) {
        if (algo_params.debug) { Rcpp::Rcout << "Merge accepted" << std::endl;}
        curr_state.cluster_allocs = standardize_allocs(prop_allocs);
        curr_state.n_clust = prop_n_clust;
        if (stats_cache) {
            // Fold the (pre-standardization-id) merge stats into the
            // persistent cache, then remap every id through the SAME
            // mapping standardize_allocs used, so the cache's keys match
            // the ids actually stored in curr_state.cluster_allocs going
            // forward (a merge can close an id gap, unlike a split).
            merge_trial->fold_into(*stats_cache);
            stats_cache->remap_ids(standardize_id_map(prop_allocs));
            sync_state_lpdf(algo_params.debug);
        } else {
            curr_state.lpdf = prop_lpdf;
        }
    } else {
        if (algo_params.debug) { Rcpp::Rcout << "Merge rejected" << std::endl;}
    }
    // Drift control and cache rebuild
    maybe_rebuild_cache();
    // Debug log
    if (algo_params.debug) { curr_state.print(); }
    return;
};

void PYSampler::gibbs_step(GibbsCouplingCallback gibbs_coupling_cb,
                                     GibbsUpdateCallback gibbs_update_cb) {
    // Debug log
    if (algo_params.debug) { Rcpp::Rcout << "gibbs_step()" << std::endl; }
    // Downcast to PYPrior to access conditional cluster updates.
    auto py_prior = std::dynamic_pointer_cast<PYFixedPrior>(prior);
    if (!py_prior) {
        throw std::runtime_error("Prior must be of type PYFixedPrior (or derived).");
    }
    // Initialize global cluster counts (max possible clusters is strictly bounded by n_data)
    arma::uvec sizes = arma::zeros<arma::uvec>(n_data);
    int K_minus_i = 0;
    for (size_t j = 0; j < n_data; ++j) {
        if (sizes(curr_state.cluster_allocs(j))++ == 0) {
            K_minus_i++; // Track the number of active clusters
        }
    }
    // Initialize random number generator
    std::uniform_real_distribution<double> runif(0.0, 1.0);
    // Gibbs-update: loop through the observations
    for (size_t i = 0; i < n_data; ++i) {
        int old_clust = curr_state.cluster_allocs(i);
        // Temporarily remove observation 'i' in O(1) time
        sizes(old_clust)--;
        if (sizes(old_clust) == 0) {
            K_minus_i--;
        }
        // Pre-allocate vectors to avoid reallocation overhead during push_back
        std::vector<double> log_probs; log_probs.reserve(K_minus_i + 1);
        std::vector<int> clust_ids; clust_ids.reserve(K_minus_i + 1);
        // Point i's aggregate distance stats to every currently active
        // cluster, computed ONCE (O(n)) and reused for every candidate below
        // -- replaces the previous per-candidate full O(n) or worse
        // eval_lpdf recompute inside this loop.
        ClusterAggMap agg;
        if (stats_cache) agg = stats_cache->point_to_clusters(distance_matrix, curr_state.cluster_allocs, i);
        // Compute probabilities for existing clusters assignments
        for (arma::uword k = 0; k < sizes.n_elem; ++k) {
            int n_k_minus_i = sizes(k);
            if (n_k_minus_i > 0) {
                double log_cond_prior = py_prior->eval_pred_lpdf_existing(n_k_minus_i);
                // Add coupling penalty (if this class is used in multi-view extension)
                double coupling_penalty = gibbs_coupling_cb ? gibbs_coupling_cb(i, old_clust, k) : 0.0;
                double lp;
                if (stats_cache) {
                    lp = curr_state.lpdf + stats_cache->eval_move_delta(agg, old_clust, k) - coupling_penalty + log_cond_prior;
                } else {
                    curr_state.cluster_allocs(i) = k; // Mutate state directly (fallback path only)
                    lp = likelihood->eval_lpdf(distance_matrix, curr_state.cluster_allocs) - coupling_penalty + log_cond_prior;
                }
                log_probs.push_back(lp);
                clust_ids.push_back(k);
            }
        }
        // Compute probability for a new cluster (ID recycling for the new cluster label)
        int new_clust_id = 0;
        while (sizes(new_clust_id) > 0) { new_clust_id++; }
        double log_cond_prior_new = py_prior->eval_pred_lpdf_new(K_minus_i);
        // Add coupling penalty (if this class is used in multi-view extension)
        double coupling_penalty_new = gibbs_coupling_cb ? gibbs_coupling_cb(i, old_clust, new_clust_id) : 0.0;
        double lp_new;
        if (stats_cache) {
            lp_new = curr_state.lpdf + stats_cache->eval_move_delta(agg, old_clust, new_clust_id) - coupling_penalty_new + log_cond_prior_new;
        } else {
            curr_state.cluster_allocs(i) = new_clust_id;
            lp_new = likelihood->eval_lpdf(distance_matrix, curr_state.cluster_allocs) - coupling_penalty_new + log_cond_prior_new;
        }
        log_probs.push_back(lp_new);
        clust_ids.push_back(new_clust_id);
        // Normalize cluster assignment probabilities
        double max_lp = *std::max_element(log_probs.begin(), log_probs.end());
        std::vector<double> probs(log_probs.size());
        double sum_probs = 0.0;
        for (size_t j = 0; j < log_probs.size(); ++j) {
            probs[j] = std::exp(log_probs[j] - max_lp);
            sum_probs += probs[j];
        }
        // Sample the new cluster assignment
        double u = runif(rng) * sum_probs;
        double cumulative = 0.0;
        int chosen_clust = clust_ids.back();
        for (size_t j = 0; j < probs.size(); ++j) {
            cumulative += probs[j];
            if (u <= cumulative) {
                chosen_clust = clust_ids[j];
                break;
            }
        }
        // Apply choice: commit to the persistent cache (this is a full
        // Gibbs update, not an accept/reject proposal, so it commits
        // directly -- no trial/rollback needed) and update the allocation
        if (stats_cache) {
            stats_cache->commit_move(agg, old_clust, chosen_clust);
            curr_state.lpdf = stats_cache->current_lpdf();
        }
        curr_state.cluster_allocs(i) = chosen_clust;
        if (sizes(chosen_clust)++ == 0) {
            K_minus_i++;
        }
        // Trigger callback updater if the cluster has been changed
        if (gibbs_update_cb && chosen_clust != old_clust) {
            gibbs_update_cb(i, old_clust, chosen_clust);
        }
    }
    // Update the current state after the Gibbs sampler loop through observations.
    // standardize_allocs can close ids gaps left by clusters that emptied out
    // during the sweep and were never recycled -- keep stats_cache's keys in
    // sync via the same id_map (see merge_step for the identical concern).
    if (stats_cache) {
        stats_cache->remap_ids(standardize_id_map(curr_state.cluster_allocs));
    }
    curr_state.cluster_allocs = standardize_allocs(curr_state.cluster_allocs);
    // K_minus_i is maintained exactly across the sweep
    curr_state.n_clust = K_minus_i;
    sync_state_lpdf(algo_params.debug);
    // Drift control and cache rebuild
    maybe_rebuild_cache();
    // Debug log
    if (algo_params.debug) { curr_state.print(); }
    return;
};

void PYSampler::sample_discount(size_t curr_iter) {
    // Debug log
    if (algo_params.debug) { Rcpp::Rcout << "sample_discount()" << std::endl; }
    // Downcast to PYHierarchicalPrior pointer to access class specific methods
    auto py_prior = std::dynamic_pointer_cast<PYHierarchicalPrior>(prior);
    if (!py_prior) return;
    // Get required parameters from PY prior
    double curr_discount = py_prior->get_discount();
    double curr_concentration = py_prior->get_concentration();
    double alpha_d = py_prior->get_hyper_params().discount_alpha;
    double beta_d = py_prior->get_hyper_params().discount_beta;
    // Propose new discount
    double lower_bound = std::max(0.0, -curr_concentration);
    double upper_bound = 1.0 - 1e-7;
    double prop_discount = truncated_normal_rng(curr_discount, discount_sd, lower_bound, upper_bound, rng);
    // Evaluate lpdf at current state
    double curr_lpdf = py_prior->eval_lpdf(curr_state.cluster_allocs) +
        beta_lpdf(curr_discount, alpha_d, beta_d);
    // Evaluate lpdf at proposed state
    py_prior->set_discount(prop_discount);
    double prop_lpdf = py_prior->eval_lpdf(curr_state.cluster_allocs) +
        beta_lpdf(prop_discount, alpha_d, beta_d);
    // Compute Metropolis Hastings corrections
    double fwd_lpdf = truncated_normal_lpdf(prop_discount, curr_discount, discount_sd, lower_bound, upper_bound);
    double rev_lpdf = truncated_normal_lpdf(curr_discount, prop_discount, discount_sd, lower_bound, upper_bound);
    // Compute acceptance ratio
    double log_arate = prop_lpdf - curr_lpdf + rev_lpdf - fwd_lpdf;
    // Test for acceptance
    std::uniform_real_distribution<double> runif(0.0, 1.0);
    if (std::log(runif(rng)) < log_arate) {
        curr_state.discount = prop_discount;
        if (algo_params.debug) Rcpp::Rcout << "Discount updated to: " << prop_discount << std::endl;
    } else {
        py_prior->set_discount(curr_discount);
    }
    // Robbins-Monro Adaptation
    double alpha = std::exp(std::min(0.0, log_arate));
    if (curr_iter <= algo_params.burnin) {
        double gamma = 1.0 / std::pow(curr_iter + 1.0, algo_params.adapt_decay);
        double log_sigma = std::log(discount_sd) + gamma * (alpha - algo_params.target_acc_rate);
        discount_sd = std::exp(log_sigma);
    }
};

void PYSampler::sample_concentration(size_t curr_iter) {
    // Debug log
    if (algo_params.debug) { Rcpp::Rcout << "sample_concentration()" << std::endl; }
    // Downcast to PYHierarchicalPrior pointer to access class specific methods
    auto py_prior = std::dynamic_pointer_cast<PYHierarchicalPrior>(prior);
    if (!py_prior) return;
    // Get required parameters
    double curr_concentration = py_prior->get_concentration();
    double alpha_c = py_prior->get_hyper_params().concentration_shape;
    double beta_c = py_prior->get_hyper_params().concentration_rate;
    // Propose new concentration
    double prop_concentration = truncated_normal_rng(curr_concentration, concentration_sd, 0.0, arma::datum::inf, rng);
    // Evaluate lpdf at current state
    double curr_lpdf = py_prior->eval_lpdf(curr_state.cluster_allocs) +
        gamma_lpdf(curr_concentration, alpha_c, beta_c);
    // Evaluate lpdf at proposed state
    py_prior->set_concentration(prop_concentration);
    double prop_lpdf = py_prior->eval_lpdf(curr_state.cluster_allocs) +
        gamma_lpdf(prop_concentration, alpha_c, beta_c);
    // Compute Metropolis Hastings corrections
    double fwd_lpdf = truncated_normal_lpdf(prop_concentration, curr_concentration, concentration_sd, 0.0, arma::datum::inf);
    double rev_lpdf = truncated_normal_lpdf(curr_concentration, prop_concentration, concentration_sd, 0.0, arma::datum::inf);
    // Compute acceptance ratio
    double log_arate = prop_lpdf - curr_lpdf + rev_lpdf - fwd_lpdf;
    // Test for acceptance
    std::uniform_real_distribution<double> runif(0.0, 1.0);
    if (std::log(runif(rng)) < log_arate) {
        curr_state.concentration = prop_concentration;
        if (algo_params.debug) Rcpp::Rcout << "Concentration updated to: " << prop_concentration << std::endl;
    } else {
        py_prior->set_concentration(curr_concentration);
    }
    // Robbins-Monro Adaptation
    double alpha = std::exp(std::min(0.0, log_arate));
    if (curr_iter <= algo_params.burnin) {
        double gamma = 1.0 / std::pow(curr_iter + 1.0, algo_params.adapt_decay);
        double log_sigma = std::log(concentration_sd) + gamma * (alpha - algo_params.target_acc_rate);
        concentration_sd = std::exp(log_sigma);
    }
};

void PYSampler::step(size_t curr_iter,
                               FullCouplingCallback full_coupling_cb,
                               GibbsCouplingCallback gibbs_coupling_cb,
                               GibbsUpdateCallback gibbs_update_cb) {
    // Debug log
    if (algo_params.debug) { Rcpp::Rcout << "step()" << std::endl; }
    // Jain and Neal (2004) approach: Alternate standard Gibbs scans with Split-Merge proposals
    if (curr_iter % 2 == 0) {
        this->gibbs_step(gibbs_coupling_cb, gibbs_update_cb);
    } else {
        // Check: if data are too few, you can' do S&M algorithm
        if (n_data < 2) return;
        // Sample two random observations
        std::uniform_int_distribution<int> dist(0, n_data - 1);
        int obs_i = dist(rng);
        int obs_j = dist(rng);
        while(obs_i == obs_j) {
            obs_j = dist(rng);
        }
        // Select the MCMC move
        if (curr_state.cluster_allocs(obs_i) == curr_state.cluster_allocs(obs_j)) {
            // Observations in same cluster: try to split it
            this->split_step(obs_i, obs_j, full_coupling_cb);
        } else {
            // Observations in different clusters: try to merge them
            this->merge_step(obs_i, obs_j, full_coupling_cb);
        }
    }
    // Sample hyperparameters of the PY process
    this->sample_discount(curr_iter);
    this->sample_concentration(curr_iter);
    // NOTE: drift control deferred to each step
};

// Relative difference between two log-densities, guarded against the
// degenerate |x| < 1 regime where an absolute comparison is the right one.
static inline double rel_diff(double a, double b) {
    const double scale = std::max(1.0, std::max(std::fabs(a), std::fabs(b)));
    return std::fabs(a - b) / scale;
}

void PYSampler::sync_state_lpdf(bool validate) {
    if (!stats_cache) {
        curr_state.lpdf = likelihood->eval_lpdf(distance_matrix, curr_state.cluster_allocs);
        return;
    }
    // The cache is the source of truth.
    curr_state.lpdf = stats_cache->current_lpdf();
    if (!validate) return;

    // Debug-only drift check. RELATIVE, not absolute
    const double full_lpdf = likelihood->eval_lpdf(distance_matrix, curr_state.cluster_allocs);
    const double rel = rel_diff(full_lpdf, curr_state.lpdf);
    if (rel > drift_warn_rel) {
        ++n_drift_warnings;
        if (n_drift_warnings <= 5) {
            Rcpp::Rcout << "Warning: cached log-density deviates from full recomputation by "
                        << std::fabs(full_lpdf - curr_state.lpdf) << " (relative " << rel << ")";
            if (n_drift_warnings == 5) Rcpp::Rcout << " [further warnings suppressed]";
            Rcpp::Rcout << std::endl;
        }
        // Repair rather than carry on with a source of truth we have just
        // shown to be untrustworthy, and watch closely afterwards.
        stats_cache->rebuild(distance_matrix, curr_state.cluster_allocs);
        curr_state.lpdf = stats_cache->current_lpdf();
        rebuild_budget = rebuild_budget_min;
    }
}

void PYSampler::validate_cache_against_likelihood() const {
    if (!stats_cache) return;
    const double full   = likelihood->eval_lpdf(distance_matrix, curr_state.cluster_allocs);
    const double cached = stats_cache->current_lpdf();
    // Both sides sum ~n^2/2 terms in different orders, so the achievable
    // agreement is bounded by naive-summation error, O(#terms * eps) -- a
    // tolerance that scales with n^2 rather than a fixed constant. A genuine
    // model mismatch is O(1) relative and will blow through this by many
    // orders of magnitude, so the tolerance can afford to be generous.
    const double n_terms = 0.5 * static_cast<double>(n_data) * static_cast<double>(n_data);
    const double tol = std::max(1e-10, 8.0 * n_terms * std::numeric_limits<double>::epsilon());
    if (!std::isfinite(full) || !std::isfinite(cached) || rel_diff(full, cached) > tol) {
        std::ostringstream msg;
        msg << "PYSampler: the incremental statistics cache does not agree with "
            << "likelihood->eval_lpdf() on a fresh rebuild (cache " << cached
            << " vs likelihood " << full << ", relative difference "
            << rel_diff(full, cached) << ", tolerance " << tol << "). "
            << "These must encode the same model -- check the d <= 0 convention, "
            << "the empty-cluster contribution and the shape/rate parameterisation.";
        throw std::runtime_error(msg.str());
    }
}

void PYSampler::maybe_rebuild_cache() {
    if (!stats_cache || rebuild_budget <= 0) return;
    if (stats_cache->updates_since_rebuild() < rebuild_budget) return;
    rebuild_cache_and_adapt();
}

void PYSampler::rebuild_cache_and_adapt() {
    const long long ops    = std::max<long long>(1, stats_cache->updates_since_rebuild());
    const double    before = stats_cache->current_lpdf();

    // Structural audit BEFORE the rebuild erases the evidence
    if (algo_params.debug) {
        const CacheAudit rep = stats_cache->audit(distance_matrix, curr_state.cluster_allocs);
        if (!rep.ok) {
            Rcpp::Rcout << "Warning: stats cache failed structural audit -- "
                        << rep.within_mismatches << " within / " << rep.between_mismatches
                        << " between entries disagree with a fresh rebuild"
                        << " (worst n_pairs delta " << rep.worst_n_pairs_delta
                        << ", worst relative sum_d error " << rep.worst_rel_sum_d
                        << ", lpdf relative difference " << rep.lpdf_rel_diff << ")";
            if (rep.first_bad_cluster >= 0) Rcpp::Rcout << " first bad cluster " << rep.first_bad_cluster;
            if (rep.first_bad_pair_a >= 0)  Rcpp::Rcout << " first bad pair (" << rep.first_bad_pair_a
                                                        << ", " << rep.first_bad_pair_b << ")";
            Rcpp::Rcout << std::endl;
            // A nonzero n_pairs delta is a logic error, never round-off:
            // no amount of rebuilding makes the run downstream of it valid.
            if (rep.worst_n_pairs_delta != 0) {
                Rcpp::Rcout << "  (n_pairs is integral -- this is a logic error, not drift)" << std::endl;
            }
        }
    }

    stats_cache->rebuild(distance_matrix, curr_state.cluster_allocs);
    const double after = stats_cache->current_lpdf();
    // The rebuild is ground truth for the cache
    sync_state_lpdf(false);

    // Drift measured for free: `before` carried `ops` incremental updates of
    // accumulated error, `after` carries none.
    const double rel_drift = rel_diff(before, after);
    worst_rel_drift = std::max(worst_rel_drift, rel_drift);
    ++n_rebuilds;

    // Per-update relative drift rate, EWMA-smoothed. Floor the observation
    // at one ulp so a lucky exact match doesn't send the budget to the
    // ceiling on a single sample.
    const double obs_rate = std::max(rel_drift, std::numeric_limits<double>::epsilon())
                          / static_cast<double>(ops);
    drift_rate_ewma = (drift_rate_ewma < 0.0)
                    ? obs_rate
                    : (1.0 - drift_rate_decay) * drift_rate_ewma + drift_rate_decay * obs_rate;

    // Extrapolate LINEARLY to the target. That is the conservative choice:
    // accumulated round-off grows at worst like O(ops) and typically like
    // O(sqrt(ops)), so a linear model rebuilds no later than necessary.
    double next = drift_target_rel / drift_rate_ewma;
    // Damp: at most a 2x move in either direction per rebuild, so one noisy
    // measurement cannot swing the schedule wildly.
    next = std::min(next, 2.0 * static_cast<double>(rebuild_budget));
    next = std::max(next, 0.5 * static_cast<double>(rebuild_budget));
    next = std::min(next, static_cast<double>(rebuild_budget_max));
    next = std::max(next, static_cast<double>(rebuild_budget_min));
    rebuild_budget = static_cast<long long>(std::llround(next));

    if (algo_params.debug) {
        Rcpp::Rcout << "Cache rebuild #" << n_rebuilds << ": " << ops
                    << " updates, relative drift " << rel_drift
                    << ", next budget " << rebuild_budget << " updates" << std::endl;
    }
}

std::unordered_map<int,int> PYSampler::standardize_id_map(const arma::uvec & allocs) const {
    std::unordered_map<int,int> id_map;
    arma::uvec unique_ids = arma::unique(allocs);
    for (size_t k = 0; k < unique_ids.n_elem; ++k) {
        id_map[unique_ids(k)] = k;
    }
    return id_map;
};

arma::uvec PYSampler::standardize_allocs(const arma::uvec & allocs) const {
    // Create std_allocs buffer
    arma::uvec std_allocs(allocs.n_elem, arma::fill::none);
    // Map old cluster ids to new contiguous ids
    arma::uvec unique_ids = arma::unique(allocs);
    int max_id = allocs.max();
    arma::uvec id_map(max_id + 1);
    for (size_t k = 0; k < unique_ids.n_elem; ++k) {
        id_map(unique_ids(k)) = k;
    }
    for (size_t i = 0; i < allocs.n_elem; ++i) {
        std_allocs(i) = id_map(allocs(i));
    }
    return std_allocs;
};

double PYSampler::restricted_gibbs_sweep(arma::uvec & allocs, const arma::uvec & members, unsigned int obs_i, unsigned int obs_j, unsigned int clust_i, unsigned int clust_j, bool sample, const arma::uvec & target_allocs, double base_lpdf, QuadraticStatsTrialCache * trial_cache) {
    // Downcast to PYFixedPrior pointer to access class specific methods
    auto py_prior = std::dynamic_pointer_cast<PYFixedPrior>(prior);
    if (!py_prior) {
        throw std::runtime_error("Prior must be of type PYFixedPrior (or derived) for Gibbs.");
    }
    // Initialize distribution
    std::uniform_real_distribution<double> runif(0.0, 1.0);
    // Buffer for log-transition probability (used in the acceptance ratio)
    double log_transition_prob = 0.0;
    // Pre-compute running cluster sizes for O(1) updates inside the loop
    int n_i = arma::accu(allocs == clust_i);
    int n_j = arma::accu(allocs == clust_j);
    // Restricted loop within members elements
    for (size_t idx = 0; idx < members.n_elem; ++idx) {
        unsigned int m = members(idx);
        if (m == obs_i || m == obs_j) continue;
        // Temporarily remove m from its current cluster count
        if (allocs(m) == clust_i) n_i--;
        if (allocs(m) == clust_j) n_j--;
        double log_p_i, log_p_j;
        // m's current label (whatever it was left at by a previous sweep, or
        // by the launch-state random init). Both trial_cache and the
        // fallback path need m's aggregate distances relative to THIS state.
        int m_from = allocs(m);
        ClusterAggMap agg; // filled below when trial_cache is used; reused for the final commit
        double current_lpdf = base_lpdf + (trial_cache ? trial_cache->delta_lpdf() : 0.0);
        if (trial_cache) {
            // O(n): one pass, reused for both candidate evaluations AND the commit below.
            agg = trial_cache->point_to_clusters(distance_matrix, allocs, m);
            double log_cond_prior_i = py_prior->eval_pred_lpdf_existing(n_i);
            log_p_i = current_lpdf + trial_cache->eval_move_delta(agg, m_from, clust_i) + log_cond_prior_i;
            double log_cond_prior_j = py_prior->eval_pred_lpdf_existing(n_j);
            log_p_j = current_lpdf + trial_cache->eval_move_delta(agg, m_from, clust_j) + log_cond_prior_j;
        } else {
            // Fallback: full recompute (used when the likelihood isn't cache-able).
            allocs(m) = clust_i;
            double log_cond_prior_i = py_prior->eval_pred_lpdf_existing(n_i);
            log_p_i = likelihood->eval_lpdf(distance_matrix, allocs) + log_cond_prior_i;
            allocs(m) = clust_j;
            double log_cond_prior_j = py_prior->eval_pred_lpdf_existing(n_j);
            log_p_j = likelihood->eval_lpdf(distance_matrix, allocs) + log_cond_prior_j;
        }
        // Normalization (Log-Sum-Exp)
        double max_log_p = std::max(log_p_i, log_p_j);
        double sum_p = std::exp(log_p_i - max_log_p) + std::exp(log_p_j - max_log_p);
        double log_norm = max_log_p + std::log(sum_p);
        double prob_i_log = log_p_i - log_norm;
        double prob_j_log = log_p_j - log_norm;
        // Sample or force path (according to the MH step in which this move is called)
        unsigned int chosen_clust;
        if (sample) {
            chosen_clust = (std::log(runif(rng)) < prob_i_log) ? clust_i : clust_j;
        }
        else {
            chosen_clust = target_allocs(m);
        }
        // Accumulate transition probabilites
        log_transition_prob += (chosen_clust == clust_i) ? prob_i_log : prob_j_log;
        // Apply choice (through the trial cache, to keep it in sync -- reuse
        // the SAME agg computed above, since allocs(m) hasn't changed yet)
        // and restore the cluster count for the next iteration
        if (trial_cache) {
            trial_cache->commit_move(agg, m_from, chosen_clust);
        }
        allocs(m) = chosen_clust;
        if (chosen_clust == clust_i) n_i++;
        if (chosen_clust == clust_j) n_j++;
    }
    // Return total transition probability (used for the computation of the MH acceptance rate)
    return log_transition_prob;
};
