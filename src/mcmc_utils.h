#pragma once

#include <random>

#include <RcppArmadillo.h>

// Sample from Truncated Normal N(mu, sigma) on [a, b]
double truncated_normal_rng(double mean, double sd, double min, double max, std::mt19937 & rng);

// Log-PDF of a Truncated Normal Distribution
double truncated_normal_lpdf(double x, double mean, double sd, double min, double max);

// Log-PDF of a Beta Distribution
double beta_lpdf(double x, double alpha, double beta);

// Log-PDF of a Gamma Distribution
double gamma_lpdf(double x, double shape, double rate);