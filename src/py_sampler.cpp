#include "py_sampler.h"

PYSplitMergeSampler::PYSplitMergeSampler(const arma::mat & _distance_matrix, std::shared_ptr<AbstractLikelihood> _likelihood_ptr, std::shared_ptr<AbstractMixturePrior> _prior_ptr, const MixtureAlgorithmParams & _algo_params): distance_matrix(_distance_matrix), likelihood(std::move(_likelihood_ptr)), prior(std::move(_prior_ptr)), algo_params(_algo_params) {};

MixtureMCMCOutput PYSplitMergeSampler::run() {
    // Initialize the sampler
    this->init();
    // Deduce retained samples
    int n_retained = (algo_params.iterations - algo_params.burnin) / algo_params.thinning;
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
            // Truncate matrices to save_idx so you don't return blocks of zeros
            out.cluster_allocs.shed_rows(save_idx, n_retained - 1);
            out.n_clust.shed_rows(save_idx, n_retained - 1);
            out.discount.shed_rows(save_idx, n_retained - 1);
            out.concentration.shed_rows(save_idx, n_retained - 1);
            out.lpdf.shed_rows(save_idx, n_retained - 1);
            break;
        }
        // Single MCMC step (Jain & Neal 2004 algorithm)
        this->step(i);
        // Store output if past burn-in and respecting thinning
        if (i >= algo_params.burnin && (i - algo_params.burnin) % algo_params.thinning == 0) {
            out.cluster_allocs.row(save_idx) = curr_state.cluster_allocs.t();
            out.n_clust(save_idx) = curr_state.n_clust;
            out.discount(save_idx) = curr_state.discount;
            out.concentration(save_idx) = curr_state.concentration;
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

void PYSplitMergeSampler::init() {
    // Debug log
    if (algo_params.debug) { Rcpp::Rcout << "init()" << std::endl; }
    // Set n_data based on the distance matrix
    n_data = distance_matrix.n_rows;
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
    // Initialization complete
    if (algo_params.debug) { curr_state.print(); }    
    return;
};

void PYSplitMergeSampler::split_step(int obs_i, int obs_j) {
    // Debug log
    if (algo_params.debug) { Rcpp::Rcout << "split_step()" << std::endl; }
    // Prepare prop_allocs buffer
    arma::uvec prop_allocs = curr_state.cluster_allocs;
    // Select elements involved in the split
    int target_clust = prop_allocs(obs_i);
    int new_clust_id = curr_state.n_clust;
    arma::uvec members = arma::find(prop_allocs == target_clust);
    // Initial random split
    prop_allocs(obs_i) = target_clust;
    prop_allocs(obs_j) = new_clust_id;
    std::uniform_real_distribution<double> runif(0.0, 1.0);
    for (size_t idx = 0; idx < members.n_elem; ++idx) {
        int m = members(idx);
        if (m != obs_i && m != obs_j) {
            prop_allocs(m) = (runif(rng) < 0.5) ? target_clust : new_clust_id;
        }
    }
    // Intermediate Gibbs sweeps (t-1 sweeps)
    for (int s = 0; s < algo_params.n_sweeps - 1; ++s) {
        restricted_gibbs_sweep(prop_allocs, members, obs_i, obs_j, target_clust, new_clust_id, true, prop_allocs);
    }
    // Final sweep to accumulate proposal probability
    double log_q_split = restricted_gibbs_sweep(prop_allocs, members, obs_i, obs_j, target_clust, new_clust_id, true, prop_allocs);
    prop_allocs = standardize_allocs(prop_allocs);
    // Complete proposal
    int prop_n_clust = arma::unique(prop_allocs).eval().n_elem;
    double prop_lpdf = likelihood->eval_lpdf(distance_matrix, prop_allocs);   
    // Compute acceptance ratio (reverse merge is deterministic, so log_q_merge = 0)
    double log_arate = prop_lpdf - curr_state.lpdf + 
        prior->eval_lpdf(prop_allocs) - prior->eval_lpdf(curr_state.cluster_allocs) - log_q_split;
    // Test for acceptance
    if(std::log(runif(rng)) < log_arate) {
        if (algo_params.debug) { Rcpp::Rcout << "Split accepted" << std::endl;}
        curr_state.cluster_allocs = std::move(prop_allocs);
        curr_state.n_clust = prop_n_clust;
        curr_state.lpdf = prop_lpdf;
    } else {
        if (algo_params.debug) { Rcpp::Rcout << "Split rejected" << std::endl;}
    }
    // Debug log
    if (algo_params.debug) { curr_state.print(); }
    return;
};

void PYSplitMergeSampler::merge_step(int obs_i, int obs_j) {
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
    arma::uvec standardized_prop = standardize_allocs(prop_allocs);
    int prop_n_clust = arma::unique(standardized_prop).eval().n_elem;
    double prop_lpdf = likelihood->eval_lpdf(distance_matrix, standardized_prop);
    // Compute reverse probability (splitting the merged state back into exact curr_state)
    arma::uvec merged_members = arma::find(prop_allocs == clust_i);
    arma::uvec dummy_allocs = prop_allocs;
    dummy_allocs(obs_i) = clust_i;
    dummy_allocs(obs_j) = clust_j;
    // Initial random split
    std::uniform_real_distribution<double> runif(0.0, 1.0);
    for (size_t idx = 0; idx < merged_members.n_elem; ++idx) {
        int m = merged_members(idx);
        if (m != obs_i && m != obs_j) {
            dummy_allocs(m) = (runif(rng) < 0.5) ? clust_i : clust_j;
        }
    }
    // Intermediate sweeps (t-1 sweeps)
    for (int s = 0; s < algo_params.n_sweeps - 1; ++s) {
        restricted_gibbs_sweep(dummy_allocs, merged_members, obs_i, obs_j, clust_i, clust_j, true, dummy_allocs);
    }
    // Final sweep: do not sample, but force transition into curr_state and calculate probability
    double log_q_split_rev = restricted_gibbs_sweep(dummy_allocs, merged_members, obs_i, obs_j, clust_i, clust_j, false, curr_state.cluster_allocs);
    // Compute acceptance ratio (forward merge is deterministic, so log_q_merge = 0)
    double log_arate = prop_lpdf - curr_state.lpdf + 
        prior->eval_lpdf(standardized_prop) - prior->eval_lpdf(curr_state.cluster_allocs) + log_q_split_rev;
    // Test for acceptance
    if(std::log(runif(rng)) < log_arate) {
        if (algo_params.debug) { Rcpp::Rcout << "Merge accepted" << std::endl;}
        curr_state.cluster_allocs = std::move(standardized_prop);
        curr_state.n_clust = prop_n_clust;
        curr_state.lpdf = prop_lpdf;
    } else {
        if (algo_params.debug) { Rcpp::Rcout << "Merge rejected" << std::endl;}
    }
    // Debug log
    if (algo_params.debug) { curr_state.print(); }
    return;
};

void PYSplitMergeSampler::gibbs_step() {
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
        // Compute probabilities for existing clusters assignments
        for (arma::uword k = 0; k < sizes.n_elem; ++k) {
            int n_k_minus_i = sizes(k);
            if (n_k_minus_i > 0) {
                curr_state.cluster_allocs(i) = k; // Mutate state directly
                double log_cond_prior = py_prior->eval_pred_lpdf_existing(n_k_minus_i);
                double lp = likelihood->eval_lpdf(distance_matrix, curr_state.cluster_allocs) + log_cond_prior;
                log_probs.push_back(lp);
                clust_ids.push_back(k);
            }
        }
        // Compute probability for a new cluster (ID recycling for the new cluster label)
        int new_clust_id = 0;
        while (sizes(new_clust_id) > 0) { new_clust_id++; }
        curr_state.cluster_allocs(i) = new_clust_id;
        double log_cond_prior_new = py_prior->eval_pred_lpdf_new(K_minus_i);
        double lp_new = likelihood->eval_lpdf(distance_matrix, curr_state.cluster_allocs) + log_cond_prior_new;
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
        // Apply choice and restore the cluster count
        curr_state.cluster_allocs(i) = chosen_clust;
        if (sizes(chosen_clust)++ == 0) {
            K_minus_i++;
        }
    }
    // Update the current state after the Gibbs sampler loop through observations
    curr_state.cluster_allocs = standardize_allocs(curr_state.cluster_allocs);
    curr_state.n_clust = arma::unique(curr_state.cluster_allocs).eval().n_elem;
    curr_state.lpdf = likelihood->eval_lpdf(distance_matrix, curr_state.cluster_allocs);
    // Debug log
    if (algo_params.debug) { curr_state.print(); }
    return;
};

void PYSplitMergeSampler::sample_discount(size_t curr_iter) {
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

void PYSplitMergeSampler::sample_concentration(size_t curr_iter) {
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

void PYSplitMergeSampler::step(size_t curr_iter) {
    // Debug log
    if (algo_params.debug) { Rcpp::Rcout << "step()" << std::endl; }
    // Jain and Neal (2004) approach: Alternate standard Gibbs scans with Split-Merge proposals
    if (curr_iter % 2 == 0) {
        this->gibbs_step();
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
            this->split_step(obs_i, obs_j);
        } else {
            // Observations in different clusters: try to merge them
            this->merge_step(obs_i, obs_j);
        }
    }
    // Sample hyperparameters of the PY process
    this->sample_discount(curr_iter);
    this->sample_concentration(curr_iter);
};

arma::uvec PYSplitMergeSampler::standardize_allocs(const arma::uvec & allocs) const {
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

double PYSplitMergeSampler::restricted_gibbs_sweep(arma::uvec & allocs, const arma::uvec & members, unsigned int obs_i, unsigned int obs_j, unsigned int clust_i, unsigned int clust_j, bool sample, const arma::uvec & target_allocs) {
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
        // Evaluate probability of assigning m to clust_i
        allocs(m) = clust_i;
        double log_cond_prior_i = py_prior->eval_pred_lpdf_existing(n_i);
        double log_p_i = likelihood->eval_lpdf(distance_matrix, allocs) + log_cond_prior_i;
        // Evaluate probability of assigning m to clust_j
        allocs(m) = clust_j;
        double log_cond_prior_j = py_prior->eval_pred_lpdf_existing(n_j);
        double log_p_j = likelihood->eval_lpdf(distance_matrix, allocs) + log_cond_prior_j;
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
        // Apply choice and restore the cluster count for the next iteration
        allocs(m) = chosen_clust;
        if (chosen_clust == clust_i) n_i++;
        if (chosen_clust == clust_j) n_j++;
    }
    // Return total transition probability (used for the computation of the MH acceptance rate)
    return log_transition_prob;
};