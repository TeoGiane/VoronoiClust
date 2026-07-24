#include "mcmc_utils.h"


double truncated_normal_rng(double mean, double sd, double min, double max, std::mt19937 & rng) {
    // Calculate CDF values at bounds
    double Fa = (min == -arma::datum::inf) ? 0.0 : R::pnorm(min, mean, sd, 1, 0);
    double Fb = (max == arma::datum::inf) ? 1.0 : R::pnorm(max, mean, sd, 1, 0);
    // Draw from Uniform(Fa, Fb)
    std::uniform_real_distribution<double> runif(Fa, Fb);
    double u = runif(rng);
    // Prevent exact 0.0 or 1.0 edge cases for the inverse CDF
    if (u <= 0.0) u = 1e-15;
    if (u >= 1.0) u = 1.0 - 1e-15;    
    // Inverse transform
    return R::qnorm(u, mean, sd, 1, 0);
};

double truncated_normal_lpdf(double x, double mean, double sd, double min, double max) {
    // If x is outside the bounds, the log-density is -infinity
    if (x < min || x > max) {
        return -arma::datum::inf;
    }
    // Log-density of the un-truncated normal
    double norm_lpdf = R::dnorm(x, mean, sd, 1);
    // Normalization constant Z = P(min < X < max)
    double Fa = (min == -arma::datum::inf) ? 0.0 : R::pnorm(min, mean, sd, 1, 0);
    double Fb = (max == arma::datum::inf) ? 1.0 : R::pnorm(max, mean, sd, 1, 0);
    double Z = Fb - Fa;
    // Truncated log-density
    return norm_lpdf - std::log(Z);
};

double beta_lpdf(double x, double alpha, double beta) {
    return R::dbeta(x, alpha, beta, 1);
};

double gamma_lpdf(double x, double shape, double rate) {
    return R::dgamma(x, shape, 1.0 / rate, 1);
};

double compute_rand_index(const arma::uvec & clus_allocs_1, const arma::uvec& clus_allocs_2) {
    // Debug log
    // Rcpp::Rcout << "compute_rand_index()" << std::endl;
    // Deduce number of data
    size_t n_data = clus_allocs_1.n_elem;
    // Check: cannot compute Rand Index with less than two observations
    if (n_data < 2) return 1.0;
    // Deduce number of pairs
    double total_pairs = (n_data * (n_data - 1)) / 2.0;    
    // Find maximum cluster IDs to size the contingency table dynamically
    arma::uword k1 = clus_allocs_1.max() + 1;
    arma::uword k2 = clus_allocs_2.max() + 1;
    // Buffers for contingency table and marginal counts
    arma::umat cont = arma::zeros<arma::umat>(k1, k2);
    arma::uvec sum1 = arma::zeros<arma::uvec>(k1);
    arma::uvec sum2 = arma::zeros<arma::uvec>(k2);    
    // O(N) pass to build the contingency table and marginal counts
    for (size_t i = 0; i < n_data; ++i) {
        cont(clus_allocs_1[i], clus_allocs_2[i])++;
        sum1(clus_allocs_1[i])++;
        sum2(clus_allocs_2[i])++;
    }
    // Calculate S1, S2, and S12 from the tables
    double S1 = 0.0, S2 = 0.0, S12 = 0.0;
    for (arma::uword i = 0; i < k1; ++i) {
        if (sum1[i] > 1) S1 += (sum1[i] * (sum1[i] - 1)) / 2.0;
    }    
    for (arma::uword j = 0; j < k2; ++j) {
        if (sum2[j] > 1) S2 += (sum2[j] * (sum2[j] - 1)) / 2.0;
    }
    for (arma::uword i = 0; i < k1; ++i) {
        for (arma::uword j = 0; j < k2; ++j) {
            if (cont(i, j) > 1) S12 += (cont(i, j) * (cont(i, j) - 1)) / 2.0;
        }
    }
    // Return the Rand Index
    return 1.0 - (S1 + S2 - 2.0 * S12) / total_pairs;
};

// OLD RAND INDEX COMPUTATION - O(N^2)
// double compute_rand_index(const arma::uvec & clus_allocs_1, const arma::uvec& clus_allocs_2) {
//     // Debug log
//     // Rcpp::Rcout << "compute_rand_index()" << std::endl;
//     // Initialize losses
//     double agree_same = 0.0;
//     double agree_diff = 0.0;
//     // Determine number of elements
//     size_t n_data = clus_allocs_1.n_elem; 
//     double total_pairs = (n_data * (n_data - 1)) / 2.0;
//     for (size_t i = 0; i < n_data; ++i) {
//         for (size_t j = i + 1; j < n_data; ++j) {
//             bool same1 = (clus_allocs_1[i] == clus_allocs_1[j]);
//             bool same2 = (clus_allocs_2[i] == clus_allocs_2[j]);
//             if (same1 && same2) agree_same += 1.0;
//             if (!same1 && !same2) agree_diff += 1.0;
//         }
//     }
//     // Return
//     return (agree_same + agree_diff) / total_pairs;
// };

Rcpp::List wrap_multiview_output(const MultiViewMCMCOutput& out) {
    // Set list size
    int n_views = out.views.size();
    Rcpp::List views_list(n_views);
    // Populate each vew
    for (int v = 0; v < n_views; ++v) {
        const MCMCOutput& current_view = out.views[v];
        Rcpp::List r_centres(current_view.centres.size());
        for (size_t i = 0; i < current_view.centres.size(); ++i) {
            std::vector<arma::uword> cpp_centers = current_view.centres[i];
            for (auto& c : cpp_centers) { c += 1; } // 1-based indexing for R
            r_centres[i] = cpp_centers;
        }        
        Rcpp::List single_view = Rcpp::List::create(
            Rcpp::Named("cluster_allocs") = current_view.cluster_allocs + 1, 
            Rcpp::Named("centres")        = r_centres,
            Rcpp::Named("n_clust")        = current_view.n_clust,
            Rcpp::Named("lpdf")           = current_view.lpdf
        );
        views_list[v] = single_view;
    }
    // Return list of lists    
    return Rcpp::List::create(
        Rcpp::Named("views")      = views_list,
        Rcpp::Named("joint_lpdf") = out.joint_lpdf
    );
};

Rcpp::List wrap_multiview_mixture_output(const MultiViewMixtureMCMCOutput& out) {
    // Set list size
    int n_views = out.views.size();
    Rcpp::List views_list(n_views);
    // Populate each view
    for (int v = 0; v < n_views; ++v) {
        const MixtureMCMCOutput& current_view = out.views[v];
        Rcpp::List single_view = Rcpp::List::create(
            Rcpp::Named("cluster_allocs") = current_view.cluster_allocs + 1, 
            Rcpp::Named("n_clust")        = current_view.n_clust,
            Rcpp::Named("discount")       = current_view.discount,
            Rcpp::Named("concentration")  = current_view.concentration,
            Rcpp::Named("lpdf")           = current_view.lpdf
        );
        views_list[v] = single_view;
    }
    // Return list of lists
    return Rcpp::List::create(
        Rcpp::Named("views")      = views_list,
        Rcpp::Named("joint_lpdf") = out.joint_lpdf
    );
};