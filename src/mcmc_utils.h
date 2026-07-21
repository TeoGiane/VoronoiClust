#pragma once

// Standard library includes
#include <random>

// Rcpp includes
#include <RcppArmadillo.h>

// Local includes
#include "sampler_types.h"


// Sample from Truncated Normal N(mu, sigma) on [a, b]
double truncated_normal_rng(double mean, double sd, double min, double max, std::mt19937 & rng);

// Log-PDF of a Truncated Normal Distribution
double truncated_normal_lpdf(double x, double mean, double sd, double min, double max);

// Log-PDF of a Beta Distribution
double beta_lpdf(double x, double alpha, double beta);

// Log-PDF of a Gamma Distribution
double gamma_lpdf(double x, double shape, double rate);

// Rand index between two cluster allocation vector
double compute_rand_index(const arma::uvec & clus_allocs_1, const arma::uvec& clus_allocs_2);

// Wrap the output of a multiview sampler into a structured Rcpp list
Rcpp::List wrap_multiview_output(const MultiViewMCMCOutput& out);