#include "factory.h"

std::shared_ptr<AbstractLikelihood> build_likelihood(const Rcpp::List& params) {
    // Parse type variable in the input list
    std::string type = Rcpp::as<std::string>(params["type"]);
    // Proper conversion in light of type
    if (type == "quadratic") {
        auto cfg = Rcpp::as<QuadraticLikelihoodParams>(params);
        return std::make_shared<QuadraticTessellationLikelihood>(cfg);
    } else if (type == "linear") {
        auto cfg = Rcpp::as<LinearLikelihoodParams>(params);
        return std::make_shared<LinearTessellationLikelihood>(cfg);
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

std::shared_ptr<AbstractMixturePrior> build_mixture_prior(const Rcpp::List & params) {
    // Parse type variable in the input list
    std::string type = Rcpp::as<std::string>(params["type"]);
    // Proper converion in light of type
    if (type == "PY-fixed") {
        auto cfg = Rcpp::as<PYFixedParams>(params);
        return std::make_shared<PYFixedPrior>(cfg);
    } else if (type == "PY-hierarchical") {
        auto cfg = Rcpp::as<PYHierarchicalParams>(params);
        return std::make_shared<PYHierarchicalPrior>(cfg);
    } else {
        throw std::invalid_argument("Unknown prior type requested: " + type);
    }
};