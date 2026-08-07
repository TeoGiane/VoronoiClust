#include "multiview_py_sampler.h"

namespace {
// Rand index clamps. The CEILING is the load-bearing one: ri == 1 gives
// dist_val == 0, and gsl_sf_hyperg_U returns GSL_EDOM there -- whose default
// handler calls abort(), taking down the whole R session. Identical
// partitions are not a corner case: without distinct per-view seeds every
// view starts from the same allocation, so init() hits it immediately.
constexpr double RI_FLOOR = 1e-10;
constexpr double RI_CEIL  = 1.0 - 1e-12;
const     double LN10     = std::log(10.0);
} // namespace

MultiViewPYSampler::MultiViewPYSampler(const std::vector<std::shared_ptr<PYSampler>> & _model_in_view,
                                       const CouplingParams& _coupling_params,
                                       const MixtureAlgorithmParams & _algo_params): 
    model_in_view(_model_in_view), coupling_params(_coupling_params), algo_params(_algo_params) {};

double MultiViewPYSampler::pair_coupling_energy(double rand_index) const {
    // Clamp both ends. The `!(ri > RI_FLOOR)` form also catches NaN.
    double ri = rand_index;
    if (!(ri > RI_FLOOR)) ri = RI_FLOOR;
    if (ri > RI_CEIL) ri = RI_CEIL;
    const double dist_val = (1.0 / ri) - 1.0;

    // The _e10_ variant returns val * 10^e10, so an extreme argument cannot
    // silently underflow to 0 and turn log(U) into -inf. (U(a,b,x) ~ x^-a for
    // large x, which underflows a plain double once a is moderately large.)
    gsl_sf_result_e10 res;
    const int status = gsl_sf_hyperg_U_e10_e(coupling_params.strength_alpha,
                                             1.0 - coupling_params.strength_beta,
                                             dist_val, &res);
    if (status != GSL_SUCCESS || !(res.val > 0.0)) {
        std::ostringstream msg;
        msg << "MultiViewPYSampler: gsl_sf_hyperg_U_e10_e failed (status " << status
            << ": " << gsl_strerror(status) << ") at a=" << coupling_params.strength_alpha
            << ", b=" << (1.0 - coupling_params.strength_beta) << ", x=" << dist_val << ".";
        throw std::runtime_error(msg.str());
    }
    const double log_U = std::log(res.val) + res.e10 * LN10;

    // NEGATED. `coupling_log_const + log(U(dist))` is a log-DENSITY: U is
    // decreasing in its argument and dist_val is decreasing in agreement, so
    // that quantity is LARGE when views agree. PYSampler subtracts whatever
    // the callbacks return (`lp = ... - coupling_penalty`, and
    // `(prop_lpdf - prop_coupling) - (curr_lpdf - curr_coupling)`), so
    // handing it the raw log-density would reward DISagreement and drive the
    // views apart. Negating here makes it an energy and leaves PYSampler's
    // convention -- and its already-reviewed acceptance ratios -- untouched.
    return -(coupling_log_const + log_U);
};

MultiViewMixtureMCMCOutput MultiViewPYSampler::run() {
    // Initialize the sampler
    this->init();
    // Deduce retained samples. Compare BEFORE subtracting (unsigned fields
    // wrap), and take the CEILING: samples land at i = burnin, burnin+thinning,
    // ... <= iterations-1, so flooring under-allocates by one row whenever
    // thinning does not divide the post-burnin range and the final store runs
    // off the end of every output buffer.
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
    MultiViewMixtureMCMCOutput out;
    out.views.resize(n_views);
    size_t n_data = model_in_view[0]->get_current_state().cluster_allocs.n_elem;
    for (size_t v = 0; v < n_views; ++v) {
        out.views[v].cluster_allocs.set_size(n_retained, n_data);
        out.views[v].n_clust.set_size(n_retained);
        out.views[v].discount.set_size(n_retained);
        out.views[v].concentration.set_size(n_retained);
        out.views[v].lpdf.set_size(n_retained);
    }
    out.joint_lpdf.set_size(n_retained);
    out.iteration_time.set_size(n_retained);
    // Print initial message (if not in debu mode)
    if (!algo_params.debug) {
        Rcpp::Rcout << "VoronoiClust: Multiview Pitman-Yor MCMC (" << algo_params.iterations << " iterations)" << std::endl;
    }
    // Initialize the progress bar (if not in debug mode)
    Progress prog_bar(algo_params.iterations, !algo_params.debug);
    int save_idx = 0;
    // Main MCMC loop
    for (size_t i = 0; i < algo_params.iterations; ++i) {
        if (Progress::check_abort()) {
            // Check for user interrupt
            Rcpp::Rcout << "\nSampling interrupted by user. Returning available samples..." << "\n";
            // Guarded: with every slot already filled there is nothing to
            // shed, and shed_rows(n, n-1) has first > last, which Armadillo
            // rejects.
            if (save_idx < n_retained) {
                for (size_t v = 0; v < n_views; ++v) {
                    out.views[v].cluster_allocs.shed_rows(save_idx, n_retained - 1);
                    out.views[v].n_clust.shed_rows(save_idx, n_retained - 1);
                    out.views[v].discount.shed_rows(save_idx, n_retained - 1);
                    out.views[v].concentration.shed_rows(save_idx, n_retained - 1);
                    out.views[v].lpdf.shed_rows(save_idx, n_retained - 1);
                }
                out.joint_lpdf.shed_rows(save_idx, n_retained - 1);
                out.iteration_time.shed_rows(save_idx, n_retained - 1);
            }
            break;
        }
        // Single MCMC step
        auto start = std::chrono::high_resolution_clock::now();
        this->step(i);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        // Store output if past burn-in and respecting thinning
        if (i >= algo_params.burnin && (i - algo_params.burnin) % algo_params.thinning == 0) {
            // Start the joint likelihood with the tracked total_coupling
            double current_joint_lpdf = -total_coupling;
            for (size_t v = 0; v < n_views; ++v) {
                MixtureState v_state = model_in_view[v]->get_current_state();
                out.views[v].cluster_allocs.row(save_idx) = v_state.cluster_allocs.t();
                out.views[v].n_clust(save_idx) = v_state.n_clust;
                out.views[v].discount(save_idx) = v_state.discount;
                out.views[v].concentration(save_idx) = v_state.concentration;
                out.views[v].lpdf(save_idx) = v_state.lpdf;
                current_joint_lpdf += v_state.lpdf;
            }
            out.joint_lpdf(save_idx) = current_joint_lpdf;
            out.iteration_time(save_idx) = elapsed.count();
            save_idx++;
        }
        // Increment the progress bar
        prog_bar.increment();
    }
    // Return (truncated) output
    return out;
};

void MultiViewPYSampler::init() {
    // Debug log
    if (algo_params.debug) { Rcpp::Rcout << "init()" << std::endl; }
    // GSL's DEFAULT error handler calls abort(), which would kill the entire
    // R session rather than raising a catchable condition. Turn it off once,
    // up front; pair_coupling_energy() checks every return status by hand.
    gsl_set_error_handler_off();
    // Initialize RNG
    rng.seed(algo_params.random_seed);
    // Initialize number of views
    n_views = model_in_view.size();        
    if (n_views == 0) {
        throw std::invalid_argument("At least one view must be provided.");
    }
    // Initialize states in each view, each on its OWN random stream. Sharing
    // algo_params.random_seed across views made every view draw an identical
    // initial allocation (Rand index exactly 1 -> coupling distance 0 -> GSL
    // domain error on the very first compute_total_coupling()) and then
    // propose the same split-merge candidate pair at every iteration.
    for (size_t v = 0; v < n_views; ++v) {
        model_in_view[v]->set_random_seed(
            algo_params.random_seed + static_cast<unsigned int>(v) * 7919u + 1u);
        model_in_view[v]->init();
    }
    // Initialize tracker (sync with cluster allocations in each view)
    size_t n_data = model_in_view[0]->get_current_state().cluster_allocs.n_elem;
    tracker.init(n_views, n_data);
    for (size_t v = 0; v < n_views; ++v) {
        tracker.sync_view(v, model_in_view[v]->get_current_state().cluster_allocs);
    }
    // Precompute the coupling normalization constant
    coupling_log_const = R::lgammafn(coupling_params.strength_alpha) - R::lbeta(coupling_params.strength_alpha, coupling_params.strength_beta);
    // Compute total coupling
    this->compute_total_coupling();
};

double MultiViewPYSampler::compute_conditional_coupling(size_t curr_view, const arma::uvec& target_z) const {
    // Debug log
    if (algo_params.debug) { Rcpp::Rcout << "compute_conditional_coupling()" << std::endl; }
    // Initialize buffer
    double total_conditional_sum = 0.0;
    // size_t n_data = target_z.n_elem;
    // double total_pairs = (n_data * (n_data - 1)) / 2.0;
    // Compute conditional couplings in the given view
    for (size_t u = 0; u < n_views; ++u) {
        if (u == curr_view) continue;
        const arma::uvec& view_u_z = tracker.clus_allocs[u];
        total_conditional_sum += pair_coupling_energy(compute_rand_index(target_z, view_u_z));
    }
    return total_conditional_sum;
};

double MultiViewPYSampler::compute_current_coupling(size_t v) const {
    double curr_coupling = 0.0;
    for (size_t u = 0; u < n_views; ++u) {
        if (u == v) continue;
        const double ri = 1.0 - (tracker.marginal_pair_counts[v] + tracker.marginal_pair_counts[u]
                                 - 2.0 * tracker.joint_pair_counts[v][u]) / tracker.total_pairs;
        curr_coupling += pair_coupling_energy(ri);
    }
    return curr_coupling;
};

void MultiViewPYSampler::compute_total_coupling() {
    total_coupling = 0.0;
    for (size_t v = 0; v < n_views; ++v) {
        for (size_t u = v + 1; u < n_views; ++u) {
            const double ri = 1.0 - (tracker.marginal_pair_counts[v] + tracker.marginal_pair_counts[u]
                                     - 2.0 * tracker.joint_pair_counts[v][u]) / tracker.total_pairs;
            total_coupling += pair_coupling_energy(ri);
        }
    }
};

void MultiViewPYSampler::step(size_t curr_iter) {
    for (size_t v = 0; v < n_views; ++v) {
        // Define the FullCouplingCallback functor for S&M updates
        auto snm_evaluator = [&](const arma::uvec & target) -> double {
            return this->compute_conditional_coupling(v, target);
        };
        // Define the GibbsCouplingCallback functor
        auto gibbs_evaluator = [&](int obs_i, int k_old, int k_new) -> double {
            // Return current coupling if the old and new clusters coincide
            if (k_old == k_new) { return this->compute_current_coupling(v); }
            // Compute total penalty using tracker variables
            double total_penalty = 0.0;
            // Reads go through size_of/joint_of: k_new can be a cluster id
            // that does not exist yet, hence beyond the tracker's current
            // capacity, and must read as 0 rather than index out of bounds.
            double S_v_new = tracker.marginal_pair_counts[v]
                           + tracker.size_of(v, k_new) - (tracker.size_of(v, k_old) - 1.0);
            for (size_t u = 0; u < n_views; ++u) {
                if (u == v) continue;
                int c = tracker.clus_allocs[u](obs_i);
                double S_vu_new = tracker.joint_pair_counts[v][u]
                                + tracker.joint_of(v, u, k_new, c) - (tracker.joint_of(v, u, k_old, c) - 1.0);
                const double ri = 1.0 - (S_v_new + tracker.marginal_pair_counts[u] - 2.0 * S_vu_new) / tracker.total_pairs;
                total_penalty += this->pair_coupling_energy(ri);
            }
            return total_penalty;
        };
        // Define the GibbsUpdateCallback functor
        auto gibbs_updater = [&](int obs_i, int k_old, int k_new) -> void {
            tracker.apply_move(v, obs_i, k_old, k_new);
        };

        // Execute the internal steps in the view, passing the callback functions
        model_in_view[v]->step(curr_iter, snm_evaluator, gibbs_evaluator, gibbs_updater);
        
        // Synchronize the view's tracker to guarantee correctness after Split/Merge
        tracker.sync_view(v, model_in_view[v]->get_current_state().cluster_allocs);
    }
    // Update total coupling once per full iteration
    this->compute_total_coupling();
};
