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
    iters_since_rebuild = 0;
}

VoronoiSampler::MoverList VoronoiSampler::determine_birth_movers(arma::uword new_centre_idx) const {
    // NOTE: relies on dist_to_own_centre reflecting the PRE-proposal state
    // (sync_cache_ids/refresh_cache_state only run at init and after
    // acceptance) -- including the degenerate curr_state.n_clust == 0 case,
    // where sync_cache_ids fills dist_to_own_centre with +inf, so the
    // general loop below already treats every point as a mover with no
    // special-casing needed. Do NOT branch on curr_state.n_clust here: by
    // the time callers reach this point they may have already temporarily
    // mutated it (see generate_birth_proposal).
    MoverList movers;
    for (arma::uword x = 0; x < n_data; ++x) {
        if (x == new_centre_idx) continue;
        double d_new = distance_matrix(x, new_centre_idx);
        if (d_new > 0.0 && d_new < dist_to_own_centre(x)) {
            movers.emplace_back(x, new_centre_idx);
        }
    }
    movers.emplace_back(new_centre_idx, new_centre_idx);
    return movers;
}

VoronoiSampler::MoverList VoronoiSampler::determine_death_movers(arma::uword dying_centre_idx) const {
    MoverList movers;
    arma::uvec members = arma::find(cache_id_allocs == dying_centre_idx);
    for (arma::uword idx = 0; idx < members.n_elem; ++idx) {
        arma::uword x = members(idx);
        arma::uword best_centre = dying_centre_idx; // degenerate fallback if no survivors (mirrors compute_tessellation's empty-centres behaviour)
        double best_dist = arma::datum::inf;
        for (arma::uword c : curr_state.cluster_centres) {
            if (c == dying_centre_idx) continue;
            double d = distance_matrix(x, c);
            if (d < best_dist) { best_dist = d; best_centre = c; }
        }
        movers.emplace_back(x, best_centre);
    }
    return movers;
}

VoronoiSampler::MoverList VoronoiSampler::determine_move_movers(arma::uword old_centre_idx, arma::uword new_centre_idx) const {
    MoverList movers;
    arma::uvec old_members = arma::find(cache_id_allocs == old_centre_idx);
    for (arma::uword idx = 0; idx < old_members.n_elem; ++idx) {
        arma::uword x = old_members(idx);
        if (x == new_centre_idx) continue; // handled unconditionally below
        arma::uword best_centre = new_centre_idx;
        double best_dist = distance_matrix(x, new_centre_idx);
        for (arma::uword c : curr_state.cluster_centres) {
            if (c == old_centre_idx) continue;
            double d = distance_matrix(x, c);
            if (d < best_dist) { best_dist = d; best_centre = c; }
        }
        movers.emplace_back(x, best_centre);
    }
    for (arma::uword x = 0; x < n_data; ++x) {
        if (x == new_centre_idx || cache_id_allocs(x) == old_centre_idx) continue;
        double d_new = distance_matrix(x, new_centre_idx);
        if (d_new > 0.0 && d_new < dist_to_own_centre(x)) {
            movers.emplace_back(x, new_centre_idx);
        }
    }
    movers.emplace_back(new_centre_idx, new_centre_idx); // the moved centre always belongs to its own new cluster
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
            double cand_lpdf;
            if (stats_cache) {
                // O(n) mover determination + O(n) scored delta, vs. the
                // O(n*K) tessellation + O(n^2) likelihood this replaces.
                MoverList movers = determine_birth_movers(k);
                arma::uvec scratch = cache_id_allocs;
                cand_lpdf = curr_state.lpdf + score_movers(movers, scratch);
            } else {
                std::vector<arma::uword> cand_centres = curr_state.cluster_centres;
                auto it = std::lower_bound(cand_centres.begin(), cand_centres.end(), (arma::uword) k);
                cand_centres.insert(it, (arma::uword) k);
                arma::uvec cand_allocs = compute_tessellation(cand_centres);
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
            double cand_lpdf;
            if (stats_cache) {
                MoverList movers = determine_death_movers(k);
                arma::uvec scratch = cache_id_allocs;
                cand_lpdf = curr_state.lpdf + score_movers(movers, scratch);
            } else {
                // Build candidate by removing the target centre
                std::vector<arma::uword> cand_centres = curr_state.cluster_centres;
                cand_centres.erase(
                    std::remove(cand_centres.begin(), cand_centres.end(), k),
                    cand_centres.end()
                );
                // Evaluate
                arma::uvec cand_allocs = compute_tessellation(cand_centres);
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
            double cand_lpdf;
            if (stats_cache) {
                MoverList movers = determine_move_movers((arma::uword) old_centre_idx, (arma::uword) k);
                arma::uvec scratch = cache_id_allocs;
                cand_lpdf = curr_state.lpdf + score_movers(movers, scratch);
            } else {
                // Build candidate by replacing the old centre and re-sorting
                std::vector<arma::uword> cand_centres = curr_state.cluster_centres;
                std::replace(cand_centres.begin(), cand_centres.end(), (arma::uword) old_centre_idx, (arma::uword) k);
                std::sort(cand_centres.begin(), cand_centres.end());
                // Evaluate
                arma::uvec cand_allocs = compute_tessellation(cand_centres);
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
        MoverList movers = determine_birth_movers(new_centre_idx);
        arma::uvec scratch = cache_id_allocs;
        res.prop_lpdf = curr_state.lpdf + score_movers(movers, scratch);
    } else if (linear_cache) {
        MoverList movers = determine_birth_movers(new_centre_idx);
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
        MoverList movers = determine_death_movers(dead_centre_idx);
        arma::uvec scratch = cache_id_allocs;
        res.prop_lpdf = curr_state.lpdf + score_movers(movers, scratch);
    } else if (linear_cache) {
        MoverList movers = determine_death_movers(dead_centre_idx);
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
        MoverList movers = determine_move_movers((arma::uword) old_centre_idx, (arma::uword) new_centre_idx);
        arma::uvec scratch = cache_id_allocs;
        res.prop_lpdf = curr_state.lpdf + score_movers(movers, scratch);
    } else if (linear_cache) {
        MoverList movers = determine_move_movers((arma::uword) old_centre_idx, (arma::uword) new_centre_idx);
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
        // Recompute the mover list once more (cheap) against the STILL
        // pre-acceptance cache_id_allocs/dist_to_own_centre/cluster_centres,
        // before any of them get updated below, then commit it to whichever
        // incremental cache is active.
        MoverList movers;
        if (stats_cache || linear_cache) {
            if (prop_state.centre_to_add != -1 && prop_state.centre_to_remove == -1) {
                movers = determine_birth_movers((arma::uword) prop_state.centre_to_add);
            } else if (prop_state.centre_to_remove != -1 && prop_state.centre_to_add == -1) {
                movers = determine_death_movers((arma::uword) prop_state.centre_to_remove);
            } else if (prop_state.centre_to_add != -1 && prop_state.centre_to_remove != -1) {
                movers = determine_move_movers((arma::uword) prop_state.centre_to_remove, (arma::uword) prop_state.centre_to_add);
            }
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
    // Periodic full rebuild of the incremental cache to correct floating-
    // point drift accumulated by many repeated add/subtract updates.
    iters_since_rebuild++;
    if ((stats_cache || linear_cache) && rebuild_every > 0 && iters_since_rebuild >= rebuild_every) {
        refresh_cache_state();
    }
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
        if (prop.centre_to_add != -1 && prop.centre_to_remove == -1) {
            movers = determine_birth_movers((arma::uword) prop.centre_to_add);
        } else if (prop.centre_to_remove != -1 && prop.centre_to_add == -1) {
            movers = determine_death_movers((arma::uword) prop.centre_to_remove);
        } else if (prop.centre_to_add != -1 && prop.centre_to_remove != -1) {
            movers = determine_move_movers((arma::uword) prop.centre_to_remove, (arma::uword) prop.centre_to_add);
        }
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
