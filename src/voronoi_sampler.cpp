#include "voronoi_sampler.h"

void VoronoiSampler::sync_cache_ids() {
    cache_id_allocs.set_size(n_data);
    dist_to_own_centre.set_size(n_data);
    if (curr_state.n_clust == 0) {
        dist_to_own_centre.fill(arma::datum::inf);
        cache_id_allocs.zeros();
        return;
    }
    for (arma::uword i = 0; i < n_data; ++i) {
        arma::uword centre = curr_state.cluster_centres[curr_state.cluster_allocs(i)];
        cache_id_allocs(i) = centre;
        dist_to_own_centre(i) = distance_matrix(centre, i);
    }
}

void VoronoiSampler::refresh_cache_state() {
    auto quad_lik = std::dynamic_pointer_cast<QuadraticTessellationLikelihood>(likelihood);
    auto lin_lik  = std::dynamic_pointer_cast<LinearTessellationLikelihood>(likelihood);
    sync_cache_ids();
    if (quad_lik) {
        if (!stats_cache) stats_cache = std::make_unique<QuadraticStatsCache>(quad_lik->get_params());
        stats_cache->rebuild(distance_matrix, cache_id_allocs);
        curr_state.lpdf = stats_cache->current_lpdf();
    } else if (lin_lik) {
        if (!linear_cache) linear_cache = std::make_unique<LinearStatsCache>(lin_lik->get_params());
        linear_cache->rebuild(distance_matrix, cache_id_allocs, curr_state.cluster_centres);
        curr_state.lpdf = linear_cache->current_lpdf();
    }
    updates_since_rebuild = 0;
}

// Relative difference between two log-densities, guarded against the
// degenerate |x| < 1 regime where an absolute comparison is the right one.
static inline double rel_diff(double a, double b) {
    const double scale = std::max(1.0, std::max(std::fabs(a), std::fabs(b)));
    return std::fabs(a - b) / scale;
}

void VoronoiSampler::maybe_rebuild_cache() {
    if ((!stats_cache && !linear_cache) || rebuild_budget <= 0) return;
    if (updates_since_rebuild < rebuild_budget) return;
    rebuild_cache_and_adapt();
}

void VoronoiSampler::rebuild_cache_and_adapt() {
    const long long ops = std::max<long long>(1, updates_since_rebuild);
    const double before = curr_state.lpdf;

    refresh_cache_state(); // rebuilds the cache from scratch and resets updates_since_rebuild
    const double after = curr_state.lpdf;

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

    // Extrapolate LINEARLY to the target -- conservative, since accumulated
    // round-off grows at worst like O(updates) and typically like
    // O(sqrt(updates)).
    double next = drift_target_rel / drift_rate_ewma;
    // Damp: at most a 2x move in either direction per rebuild.
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

VoronoiSampler::MoverList VoronoiSampler::movers_for_proposal(
    const arma::uvec& prop_cluster_allocs,
    const std::vector<arma::uword>& prop_centres) const {
    MoverList movers;
    movers.reserve(n_data);
    for (arma::uword x = 0; x < n_data; ++x) {
        arma::uword destination = 0;
        if (!prop_centres.empty()) {
            destination = prop_centres[prop_cluster_allocs(x)];
        }
        if (cache_id_allocs(x) != destination) {
            movers.emplace_back(x, destination);
        }
    }
    return movers;
}

double VoronoiSampler::score_movers(const MoverList& movers, arma::uvec& scratch_cache_ids) const {
    if (!stats_cache) return 0.0;
    QuadraticStatsTrialCache trial(*stats_cache);
    for (const auto& mv : movers) {
        arma::uword point = mv.first, to = mv.second, from = scratch_cache_ids(point);
        if (from == to) continue;
        auto agg = trial.point_to_clusters(distance_matrix, scratch_cache_ids, point);
        trial.commit_move(agg, static_cast<int>(from), static_cast<int>(to));
        scratch_cache_ids(point) = to;
    }
    return trial.delta_lpdf();
}

double VoronoiSampler::score_linear(const MoverList& movers, const std::vector<arma::uword>& new_centres, arma::uvec& scratch_cache_ids) const {
    if (!linear_cache) return 0.0;
    LinearStatsCache trial = *linear_cache; // cheap: within_stats/between are O(K)
    for (const auto& mv : movers) {
        arma::uword point = mv.first, to = mv.second, from = scratch_cache_ids(point);
        if (from != to) {
            if (from != point) trial.remove_point(distance_matrix, point, static_cast<int>(from), from);
            if (to != point)   trial.add_point(distance_matrix, point, static_cast<int>(to), to);
            scratch_cache_ids(point) = to;
        }
    }
    trial.recompute_between(distance_matrix, new_centres);
    return trial.current_lpdf() - linear_cache->current_lpdf();
}

double VoronoiSampler::commit_movers(const MoverList& movers) {
    if (!stats_cache) return 0.0;
    updates_since_rebuild += static_cast<long long>(movers.size());
    double before = stats_cache->current_lpdf();
    for (const auto& mv : movers) {
        arma::uword point = mv.first, to = mv.second, from = cache_id_allocs(point);
        if (from != to) {
            auto agg = stats_cache->point_to_clusters(distance_matrix, cache_id_allocs, point);
            stats_cache->commit_move(agg, static_cast<int>(from), static_cast<int>(to));
            cache_id_allocs(point) = to;
        }
        // No surviving cluster ever changes its centre's point-index identity
        // (see header note), so only movers -- not the whole dataset -- need
        // their dist_to_own_centre refreshed here.
        dist_to_own_centre(point) = (point == to) ? 0.0 : distance_matrix(to, point);
    }
    return stats_cache->current_lpdf() - before;
}

void VoronoiSampler::commit_linear(const MoverList& movers, const std::vector<arma::uword>& new_centres) {
    if (!linear_cache) return;
    updates_since_rebuild += static_cast<long long>(movers.size());
    for (const auto& mv : movers) {
        arma::uword point = mv.first, to = mv.second, from = cache_id_allocs(point);
        if (from != to) {
            if (from != point) linear_cache->remove_point(distance_matrix, point, static_cast<int>(from), from);
            if (to != point)   linear_cache->add_point(distance_matrix, point, static_cast<int>(to), to);
            cache_id_allocs(point) = to;
        }
        dist_to_own_centre(point) = (point == to) ? 0.0 : distance_matrix(to, point);
    }
    // The between term depends on the full centre set, not on individual
    // point moves -- O(K^2) recompute, cheap since K is small.
    linear_cache->recompute_between(distance_matrix, new_centres);
}

// Public methods
VoronoiSampler::VoronoiSampler(const arma::mat & _distance_matrix, std::shared_ptr<AbstractLikelihood> _likelihood_ptr, std::shared_ptr<AbstractPrior> _prior_ptr, const AlgorithmParams & _algo_params): distance_matrix(_distance_matrix), likelihood(std::move(_likelihood_ptr)), prior(std::move(_prior_ptr)), algo_params(_algo_params) {};

MCMCOutput VoronoiSampler::run() {
    // Initialize the sampler
    this->init();
    // Deduce retained samples
    int n_retained = (algo_params.iterations - algo_params.burnin) / algo_params.thinning;
    if (n_retained <= 0) {
        throw std::invalid_argument("Burnin is greater than iterations or thinning is too large.");
    }
    // Pre-allocate output structures
    MCMCOutput out;
    out.cluster_allocs.set_size(n_retained, n_data);
    out.centres.reserve(n_retained);
    out.n_clust.set_size(n_retained);
    out.lpdf.set_size(n_retained);
    out.iteration_time.set_size(n_retained);
    // Print inital message (if not in debug mode)
    if (!algo_params.debug) {
        Rcpp::Rcout << "VoronoiClust: Tessellation MCMC (" << algo_params.iterations << " iterations)" << std::endl;
    }
    // Initialize progress bar and save_idx
    Progress prog_bar(algo_params.iterations, !algo_params.debug);
    int save_idx = 0;
    // Main MCMC loop
    for (size_t i = 0; i < algo_params.iterations; ++i) {
        // Check for user interrupt
        if (Progress::check_abort()) {
            Rcpp::Rcout << "\nSampling interrupted by user. Returning available samples..." << "\n";
            // Truncate matrices to save_idx so you don't return blocks of zeros
            out.cluster_allocs.shed_rows(save_idx, n_retained - 1);
            out.n_clust.shed_rows(save_idx, n_retained - 1);
            out.lpdf.shed_rows(save_idx, n_retained - 1);
            out.iteration_time.shed_rows(save_idx, n_retained - 1);
            // Return
            break;
        }
        // Single MCMC step (birth, death, or move)
        auto start = std::chrono::high_resolution_clock::now();
        this->step(i);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        // Store output if past burn-in and respecting thinning
        if (i >= algo_params.burnin && (i - algo_params.burnin) % algo_params.thinning == 0) {
            out.cluster_allocs.row(save_idx) = curr_state.cluster_allocs.t();
            out.centres.push_back(curr_state.cluster_centres);
            out.n_clust(save_idx) = curr_state.n_clust;
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

// Private methods
arma::uvec VoronoiSampler::compute_tessellation(const std::vector<arma::uword> & centres) const {
    // Debug log
    if (algo_params.debug) { Rcpp::Rcout << "compute_tessellation()" << std::endl;}
    // If no centres, return zero vector
    if (centres.empty()) {
        return arma::zeros<arma::uvec>(distance_matrix.n_rows);
    }
    // Convert the std::vector of absolute indices to an Armadillo unsigned vector
    arma::uvec c_indices(const_cast<arma::uword*>(centres.data()), centres.size(), false, true);
    // This returns an N-sized vector with relative cluster IDs from 0 to (K-1).
    return arma::index_min(distance_matrix.cols(c_indices), 1);
};

arma::vec VoronoiSampler::apply_tempering(const arma::vec& log_probs, double tempering, const arma::uvec& valid_idx) const {
    // Debug log
    if (algo_params.debug) { Rcpp::Rcout << "apply_tempering()" << std::endl; }
    // Initialize
    arma::vec probs = arma::zeros<arma::vec>(log_probs.n_elem);
    if (valid_idx.is_empty()) {
        return probs;
    }
    // Extract only the valid log probabilities
    arma::vec valid_log_probs = log_probs.elem(valid_idx);
    // If all proposals are physically impossible (-inf), return uniform probabilities
    if (valid_log_probs.max() == -arma::datum::inf) {
        probs.elem(valid_idx).fill(1.0 / valid_idx.n_elem);
        return probs;
    }
    // Apply tempering if applicable
    if (tempering > 0.0) {
        valid_log_probs *= tempering;
    } else if (tempering != -1.0) {
        throw std::invalid_argument("Tempering parameter must be strictly positive or -1.0.");
    }
    // Normalization via log-sum-exp trick
    valid_log_probs -= valid_log_probs.max();
    arma::vec valid_probs = arma::exp(valid_log_probs);
    valid_probs /= arma::sum(valid_probs);
    // Negative tempering case
    if (tempering == -1.0) {
        valid_probs /= (1.0 + valid_probs);
        valid_probs /= arma::sum(valid_probs); // Re-normalize
    }
    // Return the probabilities, filling in only the valid indices
    probs.elem(valid_idx) = valid_probs;
    return probs;
};

arma::vec VoronoiSampler::compute_birth_probs() const {
    // Debug log
    if (algo_params.debug) { Rcpp::Rcout << "compute_birth_probs()" << std::endl; }
    // Get n_clust
    int n_clust = curr_state.n_clust;
    // Initialize probabilities
    arma::vec probs = arma::zeros<arma::vec>(n_data);
    // Compute probabilites
    if (algo_params.tempering == 0.0) {
        // No tempering, uniform probabilities for all non-clustered points
        if (n_data - n_clust > 0) {
            probs.elem(arma::find(curr_state.is_centre == 0)).fill(1.0 / (n_data - n_clust));
        }
        return probs;
    } else {
        // Tempering is applied to informed proposal
        arma::vec log_probs(n_data, arma::fill::value(-arma::datum::inf));
        arma::uvec indices_to_flip = arma::find(curr_state.is_centre == 0);
        #pragma omp parallel for
        for (size_t i = 0; i < indices_to_flip.n_elem; ++i) {
            int k = indices_to_flip[i];
            std::vector<arma::uword> cand_centres = curr_state.cluster_centres;
            auto it = std::lower_bound(cand_centres.begin(), cand_centres.end(), (arma::uword) k);
            cand_centres.insert(it, (arma::uword) k);
            arma::uvec cand_allocs = compute_tessellation(cand_centres);
            double cand_lpdf;
            if (stats_cache) {
                MoverList movers = movers_for_proposal(cand_allocs, cand_centres);
                arma::uvec scratch = cache_id_allocs;
                cand_lpdf = curr_state.lpdf + score_movers(movers, scratch);
            } else {
                cand_lpdf = likelihood->eval_lpdf(distance_matrix, cand_allocs, cand_centres);
            }
            log_probs(k) = cand_lpdf + prior->eval_lpdf(curr_state.n_clust + 1);
        }
        return apply_tempering(log_probs, algo_params.tempering, indices_to_flip);
    }
};

arma::vec VoronoiSampler::compute_death_probs() const {
    // Debug log
    if (algo_params.debug) { Rcpp::Rcout << "compute_death_probs()" << std::endl; }
    // Get num clusters
    int n_clust = curr_state.n_clust;
    // Initialize probabilities
    arma::vec probs = arma::zeros<arma::vec>(n_data);
    // Compute probabilities
    if (algo_params.tempering == 0.0) {
        // No tempering, uniform probabilities for all active centres
        if (n_clust > 0) {
            probs.elem(arma::find(curr_state.is_centre == 1)).fill(1.0 / n_clust);
        }
        return probs;
    } else {
        // Tempering is applied to informed proposal
        arma::vec log_probs(n_data, arma::fill::value(-arma::datum::inf));
        arma::uvec indices_to_flip = arma::find(curr_state.is_centre == 1);
        #pragma omp parallel for
        for (size_t i = 0; i < indices_to_flip.n_elem; ++i) {
            int k = indices_to_flip[i];
            std::vector<arma::uword> cand_centres = curr_state.cluster_centres;
            cand_centres.erase(
                std::remove(cand_centres.begin(), cand_centres.end(), (arma::uword) k),
                cand_centres.end()
            );
            arma::uvec cand_allocs = compute_tessellation(cand_centres);
            double cand_lpdf;
            if (stats_cache) {
                MoverList movers = movers_for_proposal(cand_allocs, cand_centres);
                arma::uvec scratch = cache_id_allocs;
                cand_lpdf = curr_state.lpdf + score_movers(movers, scratch);
            } else {
                cand_lpdf = likelihood->eval_lpdf(distance_matrix, cand_allocs, cand_centres);
            }
            log_probs(k) = cand_lpdf + prior->eval_lpdf(curr_state.n_clust - 1);
        }
        return apply_tempering(log_probs, algo_params.tempering, indices_to_flip);
    }
};

arma::vec VoronoiSampler::compute_move_probs(int old_centre_idx) const {
    // Debug log
    if (algo_params.debug) { Rcpp::Rcout << "compute_move_probs()" << std::endl; }
    // Get num clusters
    int n_clust = curr_state.n_clust;
    // Initialize probabilities
    arma::vec probs = arma::zeros<arma::vec>(n_data);
    // Compute probabilities
    if (algo_params.tempering == 0.0) {
        // No tempering, uniform probabilities for all non-clustered points
        if (n_data - n_clust > 0) {
            probs.elem(arma::find(curr_state.is_centre == 0)).fill(1.0 / (n_data - n_clust));
        }
        return probs;
    } else {
        // Tempering is applied to informed proposal
        arma::vec log_probs(n_data, arma::fill::value(-arma::datum::inf));
        arma::uvec indices_to_flip = arma::find(curr_state.is_centre == 0);
        #pragma omp parallel for
        for (size_t i = 0; i < indices_to_flip.n_elem; ++i) {
            int k = indices_to_flip[i];
            std::vector<arma::uword> cand_centres = curr_state.cluster_centres;
            std::replace(cand_centres.begin(), cand_centres.end(), (arma::uword) old_centre_idx, (arma::uword) k);
            std::sort(cand_centres.begin(), cand_centres.end());
            arma::uvec cand_allocs = compute_tessellation(cand_centres);
            double cand_lpdf;
            if (stats_cache) {
                MoverList movers = movers_for_proposal(cand_allocs, cand_centres);
                arma::uvec scratch = cache_id_allocs;
                cand_lpdf = curr_state.lpdf + score_movers(movers, scratch);
            } else {
                cand_lpdf = likelihood->eval_lpdf(distance_matrix, cand_allocs, cand_centres);
            }
            log_probs(k) = cand_lpdf + prior->eval_lpdf(curr_state.n_clust);
        }
        return apply_tempering(log_probs, algo_params.tempering, indices_to_flip);
    }
};

void VoronoiSampler::init() {
    // Debug log
    if (algo_params.debug) { Rcpp::Rcout << "init()" << std::endl; }
    // Set n_data based on the distance matrix
    n_data = distance_matrix.n_rows;
    // Set seed for reproducibility
    rng.seed(algo_params.random_seed);
    // Check: cannot have more centres than data points
    unsigned int k = algo_params.init_n_clust;
    if (k > n_data) {
        k = n_data;
    }
    // Sample initial centres
    std::uniform_int_distribution<arma::uword> ui_dist(0, n_data > 0 ? n_data - 1 : 0);
    std::vector<arma::uword> initial_centres;
    initial_centres.reserve(k);
    while(initial_centres.size() < k) {
        arma::uword cand = ui_dist(rng);
        // Only add if we haven't picked this centre already
        if (std::find(initial_centres.begin(), initial_centres.end(), cand) == initial_centres.end()) {
            initial_centres.push_back(cand);
        }
    }
    // Initialize state vectors
    curr_state.n_clust = k;
    curr_state.cluster_centres = initial_centres;
    // Sort centres to guarantee deterministic ordering
    std::sort(curr_state.cluster_centres.begin(), curr_state.cluster_centres.end());
    curr_state.is_centre = arma::zeros<arma::uvec>(n_data);
    for (arma::uword c : curr_state.cluster_centres) {
        curr_state.is_centre(c) = 1;
    }
    // Compute initial tessellation and likelihood
    curr_state.cluster_allocs = compute_tessellation(curr_state.cluster_centres);
    curr_state.lpdf = likelihood->eval_lpdf(distance_matrix, curr_state.cluster_allocs, curr_state.cluster_centres);
    // Set up the incremental cache (if the likelihood type supports it) and
    // the cache_id_allocs / dist_to_own_centre bookkeeping used by the
    // birth/death/move mover-determination logic.
    refresh_cache_state();
    // Seed the adaptive rebuild schedule (only meaningful once a cache exists).
    if (stats_cache || linear_cache) {
        rebuild_budget_min = std::max<long long>(1, static_cast<long long>(n_data));
        rebuild_budget_max = 200 * rebuild_budget_min;
        rebuild_budget     = std::min(rebuild_budget_max, 10 * rebuild_budget_min);
    }
    // Initialization complete
    if (algo_params.debug) {curr_state.print();}    
    return;
};

TessellationProposal VoronoiSampler::generate_birth_proposal() {
    // Debug log
    if (algo_params.debug) { Rcpp::Rcout << "generate_birth_proposal()" << std::endl; }
    // Prepare buffer
    TessellationProposal res;
    // Choose a new centre to add based on the birth probabilities
    arma::vec fwd_probs = compute_birth_probs();
    // Debug log
    if (algo_params.debug) { Rcpp::Rcout << "Birth probabilities: " << fwd_probs.t() << std::endl; }
    std::discrete_distribution<int> birth_dist(fwd_probs.begin(), fwd_probs.end());
    int new_centre_idx = birth_dist(rng);
    res.prob_new_old = fwd_probs(new_centre_idx);
    // Temporary mutate current state
    curr_state.n_clust += 1;
    curr_state.is_centre(new_centre_idx) = 1;
    auto it = std::lower_bound(curr_state.cluster_centres.begin(), curr_state.cluster_centres.end(), new_centre_idx);
    curr_state.cluster_centres.insert(it, new_centre_idx);
    // Compute proposed tessellation and likelihood
    res.prop_n_clust = curr_state.n_clust;
    res.prop_centres = curr_state.cluster_centres;
    res.prop_cluster_allocs = compute_tessellation(res.prop_centres);
    if (stats_cache) {
        MoverList movers = movers_for_proposal(res.prop_cluster_allocs, res.prop_centres);
        arma::uvec scratch = cache_id_allocs;
        res.prop_lpdf = curr_state.lpdf + score_movers(movers, scratch);
    } else if (linear_cache) {
        MoverList movers = movers_for_proposal(res.prop_cluster_allocs, res.prop_centres);
        arma::uvec scratch = cache_id_allocs;
        res.prop_lpdf = curr_state.lpdf + score_linear(movers, res.prop_centres, scratch);
    } else {
        res.prop_lpdf = likelihood->eval_lpdf(distance_matrix, res.prop_cluster_allocs, res.prop_centres);
    }
    // Compute reverse probabilities
    arma::vec rev_probs = compute_death_probs();
    res.prob_old_new = rev_probs(new_centre_idx);
    // Set forward delta for O(1) acceptance update
    res.centre_to_add = new_centre_idx;
    res.centre_to_remove = -1;
    // Roll-back internal state to original
    curr_state.n_clust -= 1;
    curr_state.is_centre(new_centre_idx) = 0;
    curr_state.cluster_centres.erase(std::remove(curr_state.cluster_centres.begin(), curr_state.cluster_centres.end(), new_centre_idx), curr_state.cluster_centres.end());
    // Return proposal
    return res;
};

TessellationProposal VoronoiSampler::generate_death_proposal() {
    // Debug log
    if (algo_params.debug) { Rcpp::Rcout << "generate_death_proposal()" << std::endl; }
    // Prepare buffer
    TessellationProposal res;
    // Choose a centre to remove based on the death probabilities
    arma::vec fwd_probs = compute_death_probs();
    // Debug log
    if (algo_params.debug) { Rcpp::Rcout << "Death probabilities: " << fwd_probs.t() << std::endl; }
    std::discrete_distribution<int> death_dist(fwd_probs.begin(), fwd_probs.end());
    int dead_centre_idx = death_dist(rng);
    res.prob_new_old = fwd_probs(dead_centre_idx);
    // Temporary mutate current state
    curr_state.n_clust -= 1;
    curr_state.is_centre(dead_centre_idx) = 0;
    curr_state.cluster_centres.erase(std::remove(curr_state.cluster_centres.begin(), curr_state.cluster_centres.end(), dead_centre_idx), curr_state.cluster_centres.end());
    // Compute proposed tessellation and likelihood
    res.prop_n_clust = curr_state.n_clust;
    res.prop_centres = curr_state.cluster_centres;
    res.prop_cluster_allocs = compute_tessellation(res.prop_centres);
    if (stats_cache) {
        MoverList movers = movers_for_proposal(res.prop_cluster_allocs, res.prop_centres);
        arma::uvec scratch = cache_id_allocs;
        res.prop_lpdf = curr_state.lpdf + score_movers(movers, scratch);
    } else if (linear_cache) {
        MoverList movers = movers_for_proposal(res.prop_cluster_allocs, res.prop_centres);
        arma::uvec scratch = cache_id_allocs;
        res.prop_lpdf = curr_state.lpdf + score_linear(movers, res.prop_centres, scratch);
    } else {
        res.prop_lpdf = likelihood->eval_lpdf(distance_matrix, res.prop_cluster_allocs, res.prop_centres);
    }
    // Compute reverse probabilities
    arma::vec rev_probs = compute_birth_probs();
    res.prob_old_new = rev_probs(dead_centre_idx);
    // Set forward delta for O(1) acceptance update
    res.centre_to_add = -1;
    res.centre_to_remove = dead_centre_idx;
    // Roll-back internal state to original
    curr_state.n_clust += 1;
    curr_state.is_centre(dead_centre_idx) = 1;
    auto it = std::lower_bound(curr_state.cluster_centres.begin(), curr_state.cluster_centres.end(), dead_centre_idx);
    curr_state.cluster_centres.insert(it, dead_centre_idx);
    // Return proposal
    return res;
};

TessellationProposal VoronoiSampler::generate_move_proposal() {
    // Debug log
    if (algo_params.debug) { Rcpp::Rcout << "generate_move_proposal()" << std::endl; }
    // Prepare buffer
    TessellationProposal res;
    // Choose a current centre to move using the move probabilities
    std::uniform_int_distribution<int> centre_dist(0, curr_state.n_clust - 1);
    int old_centre_idx = curr_state.cluster_centres[centre_dist(rng)];
    // Compute probabilities for where it will move
    arma::vec fwd_probs = compute_move_probs(old_centre_idx);
    // Debug log
    if (algo_params.debug) { Rcpp::Rcout << "Move probabilities: " << fwd_probs.t() << std::endl;}
    std::discrete_distribution<int> move_dist(fwd_probs.begin(), fwd_probs.end());
    int new_centre_idx = move_dist(rng);
    res.prob_new_old = fwd_probs(new_centre_idx);
    // Temporary mutate current state
    curr_state.is_centre(old_centre_idx) = 0;
    curr_state.is_centre(new_centre_idx) = 1;
    std::replace(curr_state.cluster_centres.begin(), curr_state.cluster_centres.end(), old_centre_idx, new_centre_idx);
    std::sort(curr_state.cluster_centres.begin(), curr_state.cluster_centres.end());
    // Compute proposed tessellation and likelihood
    res.prop_n_clust = curr_state.n_clust;
    res.prop_centres = curr_state.cluster_centres;
    res.prop_cluster_allocs = compute_tessellation(res.prop_centres);
    if (stats_cache) {
        MoverList movers = movers_for_proposal(res.prop_cluster_allocs, res.prop_centres);
        arma::uvec scratch = cache_id_allocs;
        res.prop_lpdf = curr_state.lpdf + score_movers(movers, scratch);
    } else if (linear_cache) {
        MoverList movers = movers_for_proposal(res.prop_cluster_allocs, res.prop_centres);
        arma::uvec scratch = cache_id_allocs;
        res.prop_lpdf = curr_state.lpdf + score_linear(movers, res.prop_centres, scratch);
    } else {
        res.prop_lpdf = likelihood->eval_lpdf(distance_matrix, res.prop_cluster_allocs, res.prop_centres);
    }
    // Compute reverse probabilities
    arma::vec rev_probs = compute_move_probs(new_centre_idx);
    res.prob_old_new = rev_probs(old_centre_idx);
    // Set forward delta for O(1) acceptance update
    res.centre_to_add = new_centre_idx;
    res.centre_to_remove = old_centre_idx;
    // Roll-back internal state to original
    curr_state.is_centre(new_centre_idx) = 0;
    curr_state.is_centre(old_centre_idx) = 1;
    std::replace(curr_state.cluster_centres.begin(), curr_state.cluster_centres.end(), new_centre_idx, old_centre_idx);
    std::sort(curr_state.cluster_centres.begin(), curr_state.cluster_centres.end());
    // Return proposal
    return res;
};

void VoronoiSampler::test_proposal(const TessellationProposal & prop_state) {
    // Debug log
    if (algo_params.debug) { Rcpp::Rcout << "test_proposal()" << std::endl; }
    // Compute acceptance ratio
    double log_arate = prop_state.prop_lpdf - curr_state.lpdf +
        prior->eval_lpdf(prop_state.prop_n_clust) - prior->eval_lpdf(curr_state.n_clust) +
        std::log(prop_state.prob_old_new) - std::log(prop_state.prob_new_old);
    // Test for acceptance
    if(std::log(std::uniform_real_distribution<double>(0.0, 1.0)(rng)) < log_arate) {
        // Derive the exact changes before replacing the current state.
        MoverList movers;
        if (stats_cache || linear_cache) {
            movers = movers_for_proposal(prop_state.prop_cluster_allocs, prop_state.prop_centres);
        }
        // Update state components
        curr_state.n_clust = prop_state.prop_n_clust;
        curr_state.cluster_allocs = std::move(prop_state.prop_cluster_allocs);
        curr_state.cluster_centres = std::move(prop_state.prop_centres);
        curr_state.lpdf = prop_state.prop_lpdf;
        // O(1) update of is_centre vector
        if(prop_state.centre_to_add != -1){
            curr_state.is_centre(prop_state.centre_to_add) = 1;
        }
        if(prop_state.centre_to_remove != -1){
            curr_state.is_centre(prop_state.centre_to_remove) = 0;
        }
        if (stats_cache) {
            commit_movers(movers);
        } else if (linear_cache) {
            commit_linear(movers, curr_state.cluster_centres);
        }
        // Debug log
        if(algo_params.debug) { Rcpp::Rcout << "Birth accepted" << std::endl;}
    } else {
        // Debug log
        if(algo_params.debug) { Rcpp::Rcout << "Brith rejected" << std::endl;}
    }
};

void VoronoiSampler::step(size_t curr_iter) {
    // Debug log
    if (algo_params.debug) { Rcpp::Rcout << "step()" << std::endl; }
    // Generate proposal
    TessellationProposal prop = generate_proposal(curr_iter);
    // Test proposal
    test_proposal(prop);
    // Adaptive drift control: rebuilds the incremental cache from scratch
    // once enough committed updates have accumulated to risk meaningful
    // floating-point drift (see the schedule members in the header).
    maybe_rebuild_cache();
};

// Public proposal generator (useful for MultiView version)
TessellationProposal VoronoiSampler::generate_proposal(size_t curr_iter) {
    // Debug log
    if (algo_params.debug) { Rcpp::Rcout << "generate_proposal()" << std::endl; }
    // Choose the move to perform at this iteration
    std::uniform_real_distribution<double> uniform_dist(0.0, 1.0);
    size_t n_clust = curr_state.n_clust;
    if (curr_iter % 2 == 0 || n_clust == 0 || n_clust == n_data) {
        bool do_birth = (uniform_dist(rng) < 0.5 || n_clust < 2) && (n_clust != n_data);
        if (do_birth) {
            return generate_birth_proposal();
        } else {
            return generate_death_proposal();
        }
    } else {
        return generate_move_proposal();
    }
};

// Force apply a state update (useful for MultiView version)
void VoronoiSampler::apply_accepted_proposal(const TessellationProposal& prop) {
    MoverList movers;
    if (stats_cache || linear_cache) {
        movers = movers_for_proposal(prop.prop_cluster_allocs, prop.prop_centres);
    }
    curr_state.n_clust = prop.prop_n_clust;
    curr_state.cluster_allocs = std::move(prop.prop_cluster_allocs);
    curr_state.cluster_centres = std::move(prop.prop_centres);
    curr_state.lpdf = prop.prop_lpdf;
    if (prop.centre_to_add != -1){
        curr_state.is_centre(prop.centre_to_add) = 1;
    }
    if (prop.centre_to_remove != -1) {
        curr_state.is_centre(prop.centre_to_remove) = 0;
    }
    if (stats_cache) {
        commit_movers(movers);
    } else if (linear_cache) {
        commit_linear(movers, curr_state.cluster_centres);
    }
};
