#include "multiview_py_sampler.h"

MultiViewPYSampler::MultiViewPYSampler(const std::vector<std::shared_ptr<PYSampler>> & _model_in_view,
                                       const CouplingParams& _coupling_params,
                                       const MixtureAlgorithmParams & _algo_params): 
    model_in_view(_model_in_view), coupling_params(_coupling_params), algo_params(_algo_params) {};

MultiViewMixtureMCMCOutput MultiViewPYSampler::run() {
    // Initialize the sampler
    this->init();
    // Deduce retained samples
    int n_retained = (algo_params.iterations - algo_params.burnin) / algo_params.thinning;
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
            for (size_t v = 0; v < n_views; ++v) {
                out.views[v].cluster_allocs.shed_rows(save_idx, n_retained - 1);
                out.views[v].n_clust.shed_rows(save_idx, n_retained - 1);
                out.views[v].discount.shed_rows(save_idx, n_retained - 1);
                out.views[v].concentration.shed_rows(save_idx, n_retained - 1);
                out.views[v].lpdf.shed_rows(save_idx, n_retained - 1);
            }
            out.joint_lpdf.shed_rows(save_idx, n_retained - 1);
            break;
        }
        // Single MCMC step
        this->step(i);
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
    // Initialize RNG
    rng.seed(algo_params.random_seed);
    // Initialize number of views
    n_views = model_in_view.size();        
    if (n_views == 0) {
        throw std::invalid_argument("At least one view must be provided.");
    }
    // Initialize states in each view
    for (size_t v = 0; v < n_views; ++v) {
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
        double ri = compute_rand_index(target_z, view_u_z);
        if (ri < 1e-10) ri = 1e-10;
        double dist_val = (1.0 / ri) - 1.0;
        double tricomi = gsl_sf_hyperg_U(coupling_params.strength_alpha, 1.0 - coupling_params.strength_beta, dist_val);
        total_conditional_sum += coupling_log_const + std::log(tricomi);
    }
    return total_conditional_sum;
};

double MultiViewPYSampler::compute_current_coupling(size_t v) const {
    double curr_coupling = 0.0;
    for (size_t u = 0; u < n_views; ++u) {
        if (u == v) continue;
        double ri = 1.0 - (tracker.marginal_pair_counts[v] + tracker.marginal_pair_counts[u] - 2.0 * tracker.joint_pair_counts[v][u]) / tracker.total_pairs;
        if (ri < 1e-10) ri = 1e-10;
        double dist_val = (1.0 / ri) - 1.0;
        double tricomi = gsl_sf_hyperg_U(coupling_params.strength_alpha, 1.0 - coupling_params.strength_beta, dist_val);
        curr_coupling += coupling_log_const + std::log(tricomi);
    }
    return curr_coupling;
};

void MultiViewPYSampler::compute_total_coupling() {
    total_coupling = 0.0;
    for (size_t v = 0; v < n_views; ++v) {
        for (size_t u = v + 1; u < n_views; ++u) {
            double ri = 1.0 - (tracker.marginal_pair_counts[v] + tracker.marginal_pair_counts[u] - 2.0 * tracker.joint_pair_counts[v][u]) / tracker.total_pairs;
            if (ri < 1e-10) ri = 1e-10;
            double dist_val = (1.0 / ri) - 1.0;
            double tricomi = gsl_sf_hyperg_U(coupling_params.strength_alpha, 1.0 - coupling_params.strength_beta, dist_val);
            total_coupling += coupling_log_const + std::log(tricomi);
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
            double S_v_new = tracker.marginal_pair_counts[v] + tracker.clus_sizes[v](k_new) - (tracker.clus_sizes[v](k_old) - 1);                
            for (size_t u = 0; u < n_views; ++u) {
                if (u == v) continue;
                int c = tracker.clus_allocs[u](obs_i);
                double S_vu_new = tracker.joint_pair_counts[v][u] + tracker.joint_cluster_counts[v][u](k_new, c) - (tracker.joint_cluster_counts[v][u](k_old, c) - 1);
                double ri = 1.0 - (S_v_new + tracker.marginal_pair_counts[u] - 2.0 * S_vu_new) / tracker.total_pairs;
                if (ri < 1e-10) ri = 1e-10;
                double dist_val = (1.0 / ri) - 1.0;
                double tricomi = gsl_sf_hyperg_U(coupling_params.strength_alpha, 1.0 - coupling_params.strength_beta, dist_val);
                total_penalty += coupling_log_const + std::log(tricomi);
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
