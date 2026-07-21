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

Rcpp::List wrap_multiview_output(const MultiViewMCMCOutput& out) {
    int n_views = out.views.size();
    Rcpp::List views_list(n_views);
    
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
    
    return Rcpp::List::create(
        Rcpp::Named("views")      = views_list,
        Rcpp::Named("joint_lpdf") = out.joint_lpdf
    );
};