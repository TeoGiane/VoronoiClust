#include "voronoi_sampler.h"

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
    //out.cluster_allocs.set_size(n_data, n_retained);
    out.cluster_allocs.set_size(n_retained, n_data);
    out.centres.reserve(n_retained);
    out.n_clust.set_size(n_retained);
    out.lpdf.set_size(n_retained);
    // Print inital message (if not in debug mode)
    if (!algo_params.debug) {
        Rcpp::Rcout << "VoronoiClust: Tessellation MCMC (" << algo_params.iterations << " iterations)" << std::endl;
    }
    // Initialize progress bar and save_idx
    Progress prog_bar(algo_params.iterations, !algo_params.debug);
    int save_idx = 0;
    // Main MCMC loop
    for (int i = 0; i < algo_params.iterations; ++i) {
        // Check for user interrupt
        if (Progress::check_abort()) {
            Rcpp::Rcout << "\nSampling interrupted by user. Returning available samples..." << "\n";
            // Truncate matrices to save_idx so you don't return blocks of zeros
            out.cluster_allocs.shed_rows(save_idx, n_retained - 1);
            out.n_clust.shed_rows(save_idx, n_retained - 1);
            out.lpdf.shed_rows(save_idx, n_retained - 1);
            break;
        }
        // Single MCMC step (birth, death, or move)
        this->step(i);
        // Store output if past burn-in and respecting thinning
        if (i >= algo_params.burnin && (i - algo_params.burnin) % algo_params.thinning == 0) {
            out.cluster_allocs.row(save_idx) = curr_state.cluster_allocs.t();
            out.centres.push_back(curr_state.cluster_centres);
            out.n_clust(save_idx) = curr_state.n_clust;
            out.lpdf(save_idx) = curr_state.lpdf;
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
            std::vector<arma::uword> cand_centers = curr_state.cluster_centres;
            auto it = std::lower_bound(cand_centers.begin(), cand_centers.end(), k);
            cand_centers.insert(it, k);
            arma::uvec cand_allocs = compute_tessellation(cand_centers);
            double cand_lpdf = likelihood->eval_lpdf(distance_matrix, cand_allocs);
            log_probs(k) = cand_lpdf + prior->eval_lpdf(cand_centers.size());
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
        // No tempering, uniform probabilities for all active centers
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
            // Build candidate by removing the target center
            std::vector<arma::uword> cand_centers = curr_state.cluster_centres;
            cand_centers.erase(
                std::remove(cand_centers.begin(), cand_centers.end(), k), 
                cand_centers.end()
            );
            // Evaluate
            arma::uvec cand_allocs = compute_tessellation(cand_centers);
            double cand_lpdf = likelihood->eval_lpdf(distance_matrix, cand_allocs);
            log_probs(k) = cand_lpdf + prior->eval_lpdf(cand_centers.size());
        }
        return apply_tempering(log_probs, algo_params.tempering, indices_to_flip);
    }
};

arma::vec VoronoiSampler::compute_move_probs(int old_center_idx) const {
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
            // Build candidate by replacing the old center and re-sorting
            std::vector<arma::uword> cand_centers = curr_state.cluster_centres;
            std::replace(cand_centers.begin(), cand_centers.end(), old_center_idx, k);
            std::sort(cand_centers.begin(), cand_centers.end());
            // Evaluate
            arma::uvec cand_allocs = compute_tessellation(cand_centers);
            double cand_lpdf = likelihood->eval_lpdf(distance_matrix, cand_allocs);
            log_probs(k) = cand_lpdf + prior->eval_lpdf(cand_centers.size());
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
    // Check: cannot have more centers than data points
    unsigned int k = algo_params.init_n_clust;
    if (k > n_data) {
        k = n_data;
    }
    // Sample initial centres
    std::uniform_int_distribution<arma::uword> ui_dist(0, n_data > 0 ? n_data - 1 : 0);
    std::vector<arma::uword> initial_centers;
    initial_centers.reserve(k);
    while(initial_centers.size() < k) {
        arma::uword cand = ui_dist(rng);
        // Only add if we haven't picked this center already
        if (std::find(initial_centers.begin(), initial_centers.end(), cand) == initial_centers.end()) {
            initial_centers.push_back(cand);
        }
    }
    // Initialize state vectors
    curr_state.n_clust = k;
    curr_state.cluster_centres = initial_centers;
    // Sort centers to guarantee deterministic ordering
    std::sort(curr_state.cluster_centres.begin(), curr_state.cluster_centres.end());
    curr_state.is_centre = arma::zeros<arma::uvec>(n_data);
    for (arma::uword c : curr_state.cluster_centres) {
        curr_state.is_centre(c) = 1;
    }
    // Compute initial tessellation and likelihood
    curr_state.cluster_allocs = compute_tessellation(curr_state.cluster_centres);
    curr_state.lpdf = likelihood->eval_lpdf(distance_matrix, curr_state.cluster_allocs);
    // Initialization complete
    if (algo_params.debug) {curr_state.print();}    
    return;
};

void VoronoiSampler::birth_step() {
    // Debug log
    if (algo_params.debug) { Rcpp::Rcout << "birth_step()" << std::endl;}
    // Choose a new center to add based on the birth probabilities
    arma::vec fwd_probs = compute_birth_probs();
    // Debug log
    if (algo_params.debug) { Rcpp::Rcout << "Birth probabilities: " << fwd_probs.t() << std::endl; }
    std::discrete_distribution<int> birth_dist(fwd_probs.begin(), fwd_probs.end());
    int new_center_idx = birth_dist(rng);
    // Generate proposal (mutate current state in place)
    curr_state.n_clust += 1;
    curr_state.is_centre(new_center_idx) = 1;
    auto it = std::lower_bound(curr_state.cluster_centres.begin(), curr_state.cluster_centres.end(), new_center_idx);
    curr_state.cluster_centres.insert(it, new_center_idx);
    // Compute the new tessellation and log-likelihood for the proposed state
    arma::uvec prop_allocs = compute_tessellation(curr_state.cluster_centres);
    double prop_lpdf = likelihood->eval_lpdf(distance_matrix, prop_allocs);
    // Compute reverse probabilities
    arma::vec rev_probs = compute_death_probs();
    double prob_new_old = fwd_probs(new_center_idx);
    double prob_old_new = rev_probs(new_center_idx);
    // Compute acceptance ratio
    double log_arate = prop_lpdf - curr_state.lpdf + 
        prior->eval_lpdf(curr_state.n_clust) - prior->eval_lpdf(curr_state.n_clust - 1) + 
        std::log(prob_old_new) - std::log(prob_new_old);
    // Accept or roll-back
    if(std::log(std::uniform_real_distribution<double>(0.0, 1.0)(rng)) < log_arate) {
        if (algo_params.debug) { Rcpp::Rcout << "Birth accepted" << std::endl;}
        curr_state.cluster_allocs = std::move(prop_allocs);
        curr_state.lpdf = prop_lpdf;
    } else {
        if (algo_params.debug) { Rcpp::Rcout << "Birth rejected" << std::endl;}
        curr_state.n_clust -= 1;
        curr_state.is_centre(new_center_idx) = 0;
        curr_state.cluster_centres.erase(std::remove(curr_state.cluster_centres.begin(), curr_state.cluster_centres.end(), new_center_idx), curr_state.cluster_centres.end());
    }
    // Debug log
    if(algo_params.debug) { curr_state.print(); }
    return;
};

void VoronoiSampler::death_step() {
    // Debug log
    if (algo_params.debug) { Rcpp::Rcout << "death_step()" << std::endl; }
    // Choose a center to remove based on the death probabilities
    arma::vec fwd_probs = compute_death_probs();
    // Debug log
    if (algo_params.debug) { Rcpp::Rcout << "Death probabilities: " << fwd_probs.t() << std::endl; }
    std::discrete_distribution<int> death_dist(fwd_probs.begin(), fwd_probs.end());
    int dead_center_idx = death_dist(rng);
    // Generate proposal (mutate current state in place)
    curr_state.n_clust -= 1;
    curr_state.is_centre(dead_center_idx) = 0;
    curr_state.cluster_centres.erase(std::remove(curr_state.cluster_centres.begin(), curr_state.cluster_centres.end(), dead_center_idx), curr_state.cluster_centres.end());
    // Compute the new tessellation and log-likelihood for the proposed state
    arma::uvec prop_allocs = compute_tessellation(curr_state.cluster_centres);
    double prop_lpdf = likelihood->eval_lpdf(distance_matrix, prop_allocs);
    // Compute reverse probabilities (the reverse of a death is a birth)
    arma::vec rev_probs = compute_birth_probs();
    double prob_new_old = fwd_probs(dead_center_idx);
    double prob_old_new = rev_probs(dead_center_idx);
    // Compute acceptance ratio (note the prior is for n_clust + 1)
    double log_arate = prop_lpdf - curr_state.lpdf + 
        prior->eval_lpdf(curr_state.n_clust) - prior->eval_lpdf(curr_state.n_clust + 1) + 
        std::log(prob_old_new) - std::log(prob_new_old);
    // Accept or roll-back
    if(std::log(std::uniform_real_distribution<double>(0.0, 1.0)(rng)) < log_arate) {
        if (algo_params.debug) { Rcpp::Rcout << "Death accepted" << std::endl;}
        curr_state.cluster_allocs = std::move(prop_allocs);
        curr_state.lpdf = prop_lpdf;
    } else {
        // Roll-back: Re-insert the center maintaining sorted order
        if (algo_params.debug) { Rcpp::Rcout << "Death rejected" << std::endl;}
        curr_state.n_clust += 1;
        curr_state.is_centre(dead_center_idx) = 1;
        auto it = std::lower_bound(curr_state.cluster_centres.begin(), curr_state.cluster_centres.end(), dead_center_idx);
        curr_state.cluster_centres.insert(it, dead_center_idx);
    }
    // Debug log
    if(algo_params.debug) { curr_state.print(); }
    return;
};

void VoronoiSampler::move_step() {
    // Debug log
    if (algo_params.debug) { Rcpp::Rcout << "move_step()" << std::endl; }
    // Choose a current center to move (uniformly from existing centers)
    std::uniform_int_distribution<int> center_dist(0, curr_state.n_clust - 1);
    int old_center_idx = curr_state.cluster_centres[center_dist(rng)];
    // Compute probabilities for where it will move
    arma::vec fwd_probs = compute_move_probs(old_center_idx);
    // Debug log
    if (algo_params.debug) { Rcpp::Rcout << "Move probabilities: " << fwd_probs.t() << std::endl;}
    std::discrete_distribution<int> move_dist(fwd_probs.begin(), fwd_probs.end());
    int new_center_idx = move_dist(rng);
    // Generate proposal (mutate current state in place)
    curr_state.is_centre(old_center_idx) = 0;
    curr_state.is_centre(new_center_idx) = 1;
    std::replace(curr_state.cluster_centres.begin(), curr_state.cluster_centres.end(), old_center_idx, new_center_idx);
    std::sort(curr_state.cluster_centres.begin(), curr_state.cluster_centres.end());
    // Compute the new tessellation and log-likelihood for the proposed state
    arma::uvec prop_allocs = compute_tessellation(curr_state.cluster_centres);
    double prop_lpdf = likelihood->eval_lpdf(distance_matrix, prop_allocs);
    // Compute reverse probabilities (moving the new center BACK to the old center)
    arma::vec rev_probs = compute_move_probs(new_center_idx);
    double prob_new_old = fwd_probs(new_center_idx);
    double prob_old_new = rev_probs(old_center_idx);
    // Compute acceptance ratio (Prior on K cancels out)
    double log_arate = prop_lpdf - curr_state.lpdf + 
        std::log(prob_old_new) - std::log(prob_new_old);
    // Accept or roll-back
    if(std::log(std::uniform_real_distribution<double>(0.0, 1.0)(rng)) < log_arate) {
        if (algo_params.debug) { Rcpp::Rcout << "Move accepted" << std::endl;}
        curr_state.cluster_allocs = std::move(prop_allocs);
        curr_state.lpdf = prop_lpdf;
    } else {
        // Roll-back: Swap them back and re-sort
        if (algo_params.debug) { Rcpp::Rcout << "Move rejected" << std::endl;}
        curr_state.is_centre(new_center_idx) = 0;
        curr_state.is_centre(old_center_idx) = 1;
        std::replace(curr_state.cluster_centres.begin(), curr_state.cluster_centres.end(), new_center_idx, old_center_idx);
        std::sort(curr_state.cluster_centres.begin(), curr_state.cluster_centres.end());
    }
    // Debug log
    if(algo_params.debug) { curr_state.print(); }
    return;
};

void VoronoiSampler::step(size_t curr_iter) {
    // Debug log
    if (algo_params.debug) { Rcpp::Rcout << "step()" << std::endl; }
    // Choose the move to perform at this iteration
    std::uniform_real_distribution<double> uniform_dist(0.0, 1.0);
    size_t n_clust = curr_state.n_clust;
    if (curr_iter % 2 == 0 || n_clust == 0 || n_clust == n_data) {
        bool do_birth = (uniform_dist(rng) < 0.5 || n_clust < 2) && (n_clust != n_data);
        if (do_birth) {
            this->birth_step();
        } else {
            this->death_step();
        }
    } else {
        this->move_step();
    }
};
