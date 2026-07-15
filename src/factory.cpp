#include "factory.h"

std::shared_ptr<AbstractLikelihood> build_likelihood(const Rcpp::List& params) {
    // Parse type variable in the input list
    std::string type = Rcpp::as<std::string>(params["type"]);
    // Proper conversion in light of type
    if (type == "quadratic") {
        auto cfg = Rcpp::as<QuadraticLikelihoodParams>(params);
        return std::make_shared<QuadraticTessellationLikelihood>(cfg);
    } else { 
        throw std::invalid_argument("Unknown likelihood type requested: " + type);
    }
};

std::shared_ptr<AbstractPrior> build_prior(const Rcpp::List& params) {
    // Parse type variable in the input list
    std::string type = Rcpp::as<std::string>(params["type"]);
    // Proper converion in light of type
    if (type == "truncated_geometric") {
        auto cfg = Rcpp::as<TruncatedGeometricParams>(params);
        return std::make_shared<TruncatedGeometricPrior>(cfg);
    } else {
        throw std::invalid_argument("Unknown prior type requested: " + type);
    }
};