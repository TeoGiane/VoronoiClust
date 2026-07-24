#include "multiview_voronoi_sampler.h"

MultiViewVoronoiSampler::MultiViewVoronoiSampler(const std::vector<std::shared_ptr<VoronoiSampler>>& _model_in_view,
                                                 const CouplingParams& _coupling_params,
                                                 const AlgorithmParams & _algo_params):
    model_in_view(_model_in_view), coupling_params(_coupling_params), algo_params(_algo_params) {};

MultiViewMCMCOutput MultiViewVoronoiSampler::run() {
    // Initialize the sampler
    this->init();
    // Deduce retained samples
    int n_retained = (algo_params.iterations - algo_params.burnin) / algo_params.thinning;
    if (n_retained <= 0) {
        throw std::invalid_argument("Burnin is greater than iterations or thinning is too large.");
    }
    // Pre-allocate output structures
    MultiViewMCMCOutput out;
    out.views.resize(n_views);
    size_t n_data = model_in_view[0]->get_current_state().cluster_allocs.n_elem;
    for (int v = 0; v < n_views; ++v) {
        out.views[v].cluster_allocs.set_size(n_retained, n_data);
        out.views[v].centres.reserve(n_retained);
        out.views[v].n_clust.set_size(n_retained);
        out.views[v].lpdf.set_size(n_retained);
    }
    out.joint_lpdf.set_size(n_retained);
    // Print initial message (if not in debug mode)
    if (!algo_params.debug) {
        Rcpp::Rcout << "VoronoiClust: Multiview Tessellation MCMC (" << algo_params.iterations << " iterations)" << std::endl;
    }
    // Initialize progress bar (if not in debug mode)
    Progress prog_bar(algo_params.iterations, !algo_params.debug);
    int save_idx = 0;
    // Main MCMC loop
    for (size_t i = 0; i < algo_params.iterations; ++i) {
        // Check for user interrupt
        if (Progress::check_abort()) {
            Rcpp::Rcout << "\nSampling ii n_vinterrupted by user. Returning available samples..." << "\n";
            for (int v = 0; v < n_views; ++v) {
                out.views[v].cluster_allocs.shed_rows(save_idx, n_retained - 1);
                out.views[v].n_clust.shed_rows(save_idx, n_retained - 1);
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
            for (int v = 0; v < n_views; ++v) {
                TessellationState v_state = model_in_view[v]->get_current_state();
                out.views[v].cluster_allocs.row(save_idx) = v_state.cluster_allocs.t();
                out.views[v].centres.push_back(v_state.cluster_centres);
                out.views[v].n_clust(save_idx) = v_state.n_clust;
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

ConditionalCouplingResult MultiViewVoronoiSampler::compute_conditional_coupling(int curr_view, const arma::uvec& target_z) const {
    // Debug log
    if (algo_params.debug) { Rcpp::Rcout << "compute_conditional_coupling()" << std::endl; }
    // Initialize buffer
    ConditionalCouplingResult res;
    res.total_conditional_sum = 0.0;
    res.pairwise_terms.assign(n_views, 0.0);
    // Compute conditional couplings in the given view
    for (int u = 0; u < n_views; ++u) {
        if (u == curr_view) continue;
        const arma::uvec& view_u_z = model_in_view[u]->get_current_state().cluster_allocs;
        double ri = compute_rand_index(target_z, view_u_z);
        if (ri < 1e-10) ri = 1e-10; 
        double dist_val = (1.0 / ri) - 1.0;
        double tricomi_val = gsl_sf_hyperg_U(
            coupling_params.strength_alpha, 
            1.0 - coupling_params.strength_beta, 
            dist_val
        );
        double log_loss = coupling_log_const + std::log(tricomi_val);
        // Store output in res
        res.pairwise_terms[u] = log_loss;
        res.total_conditional_sum += log_loss;
    }
    // Return
    return res;
};

void MultiViewVoronoiSampler::init() {
    //Debug log
    if (algo_params.debug) { Rcpp::Rcout << "init()" << std::endl; }
    // Initialize RNG
    rng.seed(algo_params.random_seed);
    // Initialize number of views
    n_views = model_in_view.size();
    if (n_views == 0) {
        throw std::invalid_argument("At least one view must be provided.");
    }
    // Initialize states in each view
    for (int v = 0; v < n_views; ++v) {
        model_in_view[v]->init();
    }
    // Precompute the coupling normalizing constant
    coupling_log_const = R::lgammafn(coupling_params.strength_alpha) - R::lbeta(coupling_params.strength_alpha, coupling_params.strength_beta);
    // Initliaze cached pairwise couplings
    pairwise_couplings.zeros(n_views, n_views);
    total_coupling = 0.0;        
    for (int v = 0; v < n_views; ++v) {
        const arma::uvec& z_v = model_in_view[v]->get_current_state().cluster_allocs;
        for (int u = v + 1; u < n_views; ++u) {
            const arma::uvec& z_u = model_in_view[u]->get_current_state().cluster_allocs;
            double ri = compute_rand_index(z_v, z_u);
            if (ri < 1e-10) ri = 1e-10; 
            double dist_val = (1.0 / ri) - 1.0;
            double tricomi_val = gsl_sf_hyperg_U(coupling_params.strength_alpha, 1.0 - coupling_params.strength_beta, dist_val);
            double log_loss = coupling_log_const + std::log(tricomi_val);
            // Populate symmetric matrix and add to strictly upper triangle sum
            pairwise_couplings(v, u) = log_loss;
            pairwise_couplings(u, v) = log_loss;
            total_coupling += log_loss;
        }
    }
};

void MultiViewVoronoiSampler::step(size_t curr_iter) {
    // Define uniform distribution
    std::uniform_real_distribution<double> uniform_dist(0.0, 1.0);
    // Loop through the views
    for (int v = 0; v < n_views; ++v) {
        // Generate Proposalin current view
        TessellationProposal prop = model_in_view[v]->generate_proposal(curr_iter);
        // Compute proposed conditional couplings
        ConditionalCouplingResult prop_coupling = compute_conditional_coupling(v, prop.prop_cluster_allocs);
        // Get old coupling directly from cached matrix
        double coupling_old = 0.0;
        for (int u = 0; u < n_views; ++u) {
            coupling_old += pairwise_couplings(v, u);
        }
        // Compute MH Ratio
        auto view_prior = model_in_view[v]->get_prior();
        double prop_loss = prop.prop_lpdf - prop_coupling.total_conditional_sum;
        double curr_loss = model_in_view[v]->get_current_state().lpdf - coupling_old;
        double log_arate = prop_loss - curr_loss + 
            view_prior->eval_lpdf(prop.prop_n_clust) - view_prior->eval_lpdf(model_in_view[v]->get_current_state().n_clust) +
            std::log(prop.prob_old_new) - std::log(prop.prob_new_old);
        // double log_prior_ratio = view_prior->eval_lpdf(prop.prop_n_clust) - view_prior->eval_lpdf(model_in_view[v]->get_current_state().n_clust);                 
        // double log_numerator = prop.prop_lpdf + log_prior_ratio + std::log(prop.prob_old_new) + prop_coupling.total_conditional_sum;
        // double log_denominator = model_in_view[v]->get_current_state().lpdf + std::log(prop.prob_new_old) + coupling_old;
        // Accept or Reject the proposed move in the current view
        if (std::log(uniform_dist(rng)) < log_arate/*(log_numerator - log_denominator)*/) {
            // Update state
            model_in_view[v]->apply_accepted_proposal(prop);
            // Update cached couplings and total_coupling
            total_coupling = total_coupling - coupling_old + prop_coupling.total_conditional_sum;
            for (int u = 0; u < n_views; ++u) {
                if (u != v) {
                    pairwise_couplings(v, u) = prop_coupling.pairwise_terms[u];
                    pairwise_couplings(u, v) = prop_coupling.pairwise_terms[u];
                }
            }
            // Debug log
            if (algo_params.debug) { Rcpp::Rcout << "View " << v << ": Proposal accepted" << std::endl; }
        } else {
            // Debug log
            if (algo_params.debug) { Rcpp::Rcout << "View " << v << ": Proposal rejected" << std::endl; }
        }
    }
    // End updates in each view
    return;
};