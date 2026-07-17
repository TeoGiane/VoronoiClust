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