#include "likelihoods.h"

double QuadraticTessellationLikelihood::eval_lpdf(const arma::mat& dist_matrix, const arma::uvec& cluster_allocs) const {
        
    // 1. O(N) ONE-PASS GROUPING
    // Find unique clusters (arma::unique returns a sorted vector)
    arma::uvec unique_z = arma::unique(cluster_allocs);
    int K = unique_z.n_elem;
    
    // Pre-allocate buckets for each cluster
    std::vector<std::vector<arma::uword>> clusters(K);
    for (arma::uword i = 0; i < cluster_allocs.n_elem; ++i) {
        // Fast binary search to find which bucket this index belongs to
        auto it = std::lower_bound(unique_z.begin(), unique_z.end(), cluster_allocs[i]);
        int k = std::distance(unique_z.begin(), it);
        clusters[k].push_back(i);
    }

    // 2. PRECOMPUTE CONSTANTS
    double total_lpdf = 0.0;
    double const_within = params.prior_shape_within * std::log(params.prior_rate_within) - std::lgamma(params.prior_shape_within);
    double const_between = params.prior_shape_between * std::log(params.prior_rate_between) - std::lgamma(params.prior_shape_between);

    // 3. WITHIN-CLUSTER EVALUATION
    // #pragma omp parallel for reduction(+:total_lpdf)
    for (int k = 0; k < K; ++k) {
        const auto& idx = clusters[k];
        int n = idx.size();
        
        if (n > 1) {
            double sum_d = 0.0;
            double sum_log_d = 0.0;
            int n_pairs = 0;
            
            // Read directly from matrix - ZERO heap allocations!
            for (int i = 0; i < n; ++i) {
                for (int j = i + 1; j < n; ++j) {
                    double d = dist_matrix(idx[i], idx[j]);
                    if (d > 0.0) { // Safely handles identical points (log(0) = -inf)
                        sum_d += d;
                        sum_log_d += std::log(d);
                        n_pairs++;
                    }
                }
            }
            
            if (n_pairs > 0) {
                double term = const_within + 
                    (params.shape_within - 1.0) * sum_log_d + 
                    (-params.prior_shape_within - params.shape_within * n_pairs) * std::log(params.prior_rate_within + sum_d) - 
                    n_pairs * std::lgamma(params.shape_within) + 
                    std::lgamma(params.prior_shape_within + n_pairs * params.shape_within);
                total_lpdf += term;
            }
        }
    }

    // 4. BETWEEN-CLUSTER EVALUATION
    if (K > 1 && params.repulsion) {
        // schedule(dynamic) is used because inner loop length shrinks as k increases
        // #pragma omp parallel for schedule(dynamic) reduction(+:total_lpdf)
        for (int k = 0; k < K - 1; ++k) {
            const auto& idx1 = clusters[k];
            
            for (int t = k + 1; t < K; ++t) {
                const auto& idx2 = clusters[t];
                double sum_d = 0.0;
                double sum_log_d = 0.0;
                int n_pairs = 0;
                
                // Read directly from matrix - ZERO heap allocations!
                for (size_t i = 0; i < idx1.size(); ++i) {
                    for (size_t j = 0; j < idx2.size(); ++j) {
                        double d = dist_matrix(idx1[i], idx2[j]);
                        if (d > 0.0) {
                            sum_d += d;
                            sum_log_d += std::log(d);
                            n_pairs++;
                        }
                    }
                }
                
                if (n_pairs > 0) {
                    double term = const_between + 
                        (params.shape_between - 1.0) * sum_log_d + 
                        (-params.prior_shape_between - params.shape_between * n_pairs) * std::log(params.prior_rate_between + sum_d) - 
                        n_pairs * std::lgamma(params.shape_between) + 
                        std::lgamma(params.prior_shape_between + n_pairs * params.shape_between);
                    total_lpdf += term;
                }
            }
        }
    }

    return total_lpdf;
};

// This works, maybe can be better optimized
// double QuadraticTessellationLikelihood::eval_lpdf(const arma::mat & dist_matrix, const arma::uvec & cluster_allocs) const {
        
//     // Build loglikelihood matrix
//     arma::uvec unique_z = unique(cluster_allocs);
//     int K = unique_z.n_elem;
//     arma::mat log_Lkt = arma::zeros<arma::mat>(K, K); // Matrix initialization with zeros
    
//     // Compute within-cluster contributions
//     #pragma omp parallel for
//     for (int k = 0; k < K; ++k) {
//         arma::uvec index_select = arma::find(cluster_allocs == unique_z(k));
//         if (index_select.n_elem > 1) {
//             arma::mat D_k = dist_matrix.submat(index_select, index_select);
//             arma::vec distances = D_k(trimatu_ind(size(D_k), 1));
//             // distances = distances.elem(find_finite(distances));    //remove NAs
    
//         log_Lkt(k, k)  = params.prior_shape_within * log(params.prior_rate_within) +
//             (params.shape_within - 1) * sum(log(distances)) +
//             (-params.prior_shape_within - params.shape_within * distances.size()) * log(params.prior_rate_within + sum(distances)) -
//             distances.size() * lgamma(params.shape_within) +
//             lgamma(params.prior_shape_within + distances.size() * params.shape_within ) - lgamma(params.prior_shape_within);
//         }
//     }

//     // Compute between-cluster contributions if repulsion is enabled
//     if (K > 1 && params.repulsion) {
//         #pragma omp parallel for collapse(2)
//         for (int k = 0; k < K - 1; ++k) {
//             arma::uvec index_select1 = arma::find(cluster_allocs == unique_z(k));
//             for (int t = k + 1; t < K; ++t) {
//                 arma::uvec index_select2 = arma::find(cluster_allocs == unique_z(t));
//                 arma::mat D_kt = dist_matrix.submat(index_select1, index_select2);
//                 arma::vec distances = arma::vectorise(D_kt);
//                 // distances = distances.elem(find_finite(distances));   //remove NAs 
//                 log_Lkt(k, t) = params.prior_shape_between * log(params.prior_rate_between) + (params.shape_between - 1) * sum(log(distances)) +
//                 (-params.prior_shape_between - params.shape_between * distances.size()) * log(params.prior_rate_between + sum(distances)) -
//                 distances.size() * lgamma(params.shape_between ) +
//                 lgamma(params.prior_shape_between + distances.size() * params.shape_between) - lgamma(params.prior_shape_between);
//             }
//         }
//     }

//     double lpdf = accu(log_Lkt);
//     return lpdf;
// };