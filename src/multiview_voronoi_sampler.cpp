#include "multiview_voronoi_sampler.h"

namespace {
constexpr double RI_FLOOR = 1e-10;
constexpr double RI_CEIL = 1.0 - 1e-12;
const double LN10 = std::log(10.0);
}

MultiViewVoronoiSampler::MultiViewVoronoiSampler(
    const std::vector<std::shared_ptr<VoronoiSampler>>& _model_in_view,
    const CouplingParams& _coupling_params,
    const AlgorithmParams& _algo_params)
    : model_in_view(_model_in_view), coupling_params(_coupling_params), algo_params(_algo_params) {}

MultiViewMCMCOutput MultiViewVoronoiSampler::run() {
    init();
    if (algo_params.thinning == 0) {
        throw std::invalid_argument("Thinning must be at least 1.");
    }
    if (algo_params.burnin >= algo_params.iterations) {
        throw std::invalid_argument("Burnin is greater than iterations or thinning is too large.");
    }
    const size_t n_retained = (algo_params.iterations - algo_params.burnin + algo_params.thinning - 1) / algo_params.thinning;

    MultiViewMCMCOutput out;
    out.views.resize(n_views);
    const size_t n_data = model_in_view[0]->get_current_state().cluster_allocs.n_elem;
    for (size_t view = 0; view < n_views; ++view) {
        out.views[view].cluster_allocs.set_size(n_retained, n_data);
        out.views[view].centres.reserve(n_retained);
        out.views[view].n_clust.set_size(n_retained);
        out.views[view].lpdf.set_size(n_retained);
    }
    out.joint_lpdf.set_size(n_retained);
    out.iteration_time.set_size(n_retained);

    if (!algo_params.debug) {
        Rcpp::Rcout << "VoronoiClust: Multiview Tessellation MCMC (" << algo_params.iterations << " iterations)" << std::endl;
    }
    Progress prog_bar(algo_params.iterations, !algo_params.debug);
    size_t save_idx = 0;
    for (size_t iteration = 0; iteration < algo_params.iterations; ++iteration) {
        if (Progress::check_abort()) {
            Rcpp::Rcout << "\nSampling interrupted by user. Returning available samples...\n";
            if (save_idx < n_retained) {
                for (size_t view = 0; view < n_views; ++view) {
                    out.views[view].cluster_allocs.shed_rows(save_idx, n_retained - 1);
                    out.views[view].n_clust.shed_rows(save_idx, n_retained - 1);
                    out.views[view].lpdf.shed_rows(save_idx, n_retained - 1);
                }
                out.joint_lpdf.shed_rows(save_idx, n_retained - 1);
                out.iteration_time.shed_rows(save_idx, n_retained - 1);
            }
            break;
        }

        const auto start = std::chrono::high_resolution_clock::now();
        step(iteration);
        const auto end = std::chrono::high_resolution_clock::now();
        const std::chrono::duration<double> elapsed = end - start;

        if (iteration >= algo_params.burnin && (iteration - algo_params.burnin) % algo_params.thinning == 0) {
            double joint_lpdf = -total_coupling;
            for (size_t view = 0; view < n_views; ++view) {
                const TessellationState state = model_in_view[view]->get_current_state();
                out.views[view].cluster_allocs.row(save_idx) = state.cluster_allocs.t();
                out.views[view].centres.push_back(state.cluster_centres);
                out.views[view].n_clust(save_idx) = state.n_clust;
                out.views[view].lpdf(save_idx) = state.lpdf;
                joint_lpdf += state.lpdf;
            }
            out.joint_lpdf(save_idx) = joint_lpdf;
            out.iteration_time(save_idx) = elapsed.count();
            ++save_idx;
        }
        prog_bar.increment();
    }
    return out;
}

void MultiViewVoronoiSampler::init() {
    gsl_set_error_handler_off();
    rng.seed(algo_params.random_seed);
    n_views = model_in_view.size();
    if (n_views == 0) {
        throw std::invalid_argument("At least one view must be provided.");
    }

    for (size_t view = 0; view < n_views; ++view) {
        model_in_view[view]->init();
    }
    const size_t n_data = model_in_view[0]->get_current_state().cluster_allocs.n_elem;
    tracker.init(n_views, n_data);
    for (size_t view = 0; view < n_views; ++view) {
        tracker.sync_view(view, model_in_view[view]->get_current_state().cluster_allocs);
    }
    coupling_log_const = R::lgammafn(coupling_params.strength_alpha)
                      - R::lbeta(coupling_params.strength_alpha, coupling_params.strength_beta);
    compute_total_coupling();
}

double MultiViewVoronoiSampler::pair_coupling_energy(double rand_index) const {
    double ri = rand_index;
    if (!(ri > RI_FLOOR)) ri = RI_FLOOR;
    if (ri > RI_CEIL) ri = RI_CEIL;
    const double dist_val = (1.0 / ri) - 1.0;

    gsl_sf_result_e10 result;
    const int status = gsl_sf_hyperg_U_e10_e(coupling_params.strength_alpha,
                                             1.0 - coupling_params.strength_beta,
                                             dist_val, &result);
    if (status != GSL_SUCCESS || !(result.val > 0.0)) {
        std::ostringstream msg;
        msg << "MultiViewVoronoiSampler: gsl_sf_hyperg_U_e10_e failed (status " << status
            << ": " << gsl_strerror(status) << ") at a=" << coupling_params.strength_alpha
            << ", b=" << (1.0 - coupling_params.strength_beta) << ", x=" << dist_val << ".";
        throw std::runtime_error(msg.str());
    }
    return -(coupling_log_const + std::log(result.val) + result.e10 * LN10);
}

double MultiViewVoronoiSampler::conditional_coupling_from_tracker(size_t view) const {
    if (tracker.total_pairs == 0.0) return 0.0;
    double total = 0.0;
    for (size_t other = 0; other < n_views; ++other) {
        if (other == view) continue;
        const double ri = 1.0 - (tracker.marginal_pair_counts[view]
                               + tracker.marginal_pair_counts[other]
                               - 2.0 * tracker.joint_pair_counts[view][other]) / tracker.total_pairs;
        total += pair_coupling_energy(ri);
    }
    return total;
}

void MultiViewVoronoiSampler::compute_total_coupling() {
    if (tracker.total_pairs == 0.0) {
        total_coupling = 0.0;
        return;
    }
    total_coupling = 0.0;
    for (size_t view = 0; view < n_views; ++view) {
        for (size_t other = view + 1; other < n_views; ++other) {
            const double ri = 1.0 - (tracker.marginal_pair_counts[view]
                                   + tracker.marginal_pair_counts[other]
                                   - 2.0 * tracker.joint_pair_counts[view][other]) / tracker.total_pairs;
            total_coupling += pair_coupling_energy(ri);
        }
    }
}

double MultiViewVoronoiSampler::score_proposal_coupling(size_t view, const arma::uvec& proposed_allocs) {
    struct Move { arma::uword obs; int from; int to; };
    std::vector<Move> moves;
    moves.reserve(proposed_allocs.n_elem);
    for (arma::uword obs = 0; obs < proposed_allocs.n_elem; ++obs) {
        const int from = static_cast<int>(tracker.clus_allocs[view](obs));
        const int to = static_cast<int>(proposed_allocs(obs));
        if (from != to) {
            tracker.apply_move(view, static_cast<int>(obs), from, to);
            moves.push_back({obs, from, to});
        }
    }
    const double score = conditional_coupling_from_tracker(view);
    for (auto move = moves.rbegin(); move != moves.rend(); ++move) {
        tracker.apply_move(view, static_cast<int>(move->obs), move->to, move->from);
    }
    return score;
}

void MultiViewVoronoiSampler::apply_allocation_changes(size_t view, const arma::uvec& target_allocs) {
    for (arma::uword obs = 0; obs < target_allocs.n_elem; ++obs) {
        const int from = static_cast<int>(tracker.clus_allocs[view](obs));
        const int to = static_cast<int>(target_allocs(obs));
        if (from != to) {
            tracker.apply_move(view, static_cast<int>(obs), from, to);
        }
    }
}

void MultiViewVoronoiSampler::step(size_t curr_iter) {
    std::uniform_real_distribution<double> uniform_dist(0.0, 1.0);
    for (size_t view = 0; view < n_views; ++view) {
        TessellationProposal proposal = model_in_view[view]->generate_proposal(curr_iter);
        const double proposed_coupling = score_proposal_coupling(view, proposal.prop_cluster_allocs);
        const double current_coupling = conditional_coupling_from_tracker(view);
        const TessellationState current_state = model_in_view[view]->get_current_state();
        const auto prior = model_in_view[view]->get_prior();
        const double log_arate = (proposal.prop_lpdf - proposed_coupling)
                               - (current_state.lpdf - current_coupling)
                               + prior->eval_lpdf(proposal.prop_n_clust)
                               - prior->eval_lpdf(current_state.n_clust)
                               + std::log(proposal.prob_old_new)
                               - std::log(proposal.prob_new_old);
        if (std::log(uniform_dist(rng)) < log_arate) {
            model_in_view[view]->apply_accepted_proposal(proposal);
            apply_allocation_changes(view, model_in_view[view]->get_current_state().cluster_allocs);
            compute_total_coupling();
            if (algo_params.debug) {
                Rcpp::Rcout << "View " << view << ": Proposal accepted" << std::endl;
            }
        } else if (algo_params.debug) {
            Rcpp::Rcout << "View " << view << ": Proposal rejected" << std::endl;
        }
    }
}
