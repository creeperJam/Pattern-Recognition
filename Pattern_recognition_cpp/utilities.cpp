#include "utilities.h"

#include <cmath>
#include <fstream>
#include <iostream>
#include <numeric>
#include <sstream>

#include "sequential.h"

TimeSeriesSoA LoadAndPrepareData(const std::string& filepath) {
    std::ifstream file(filepath);

    if (!file.is_open()) {
        std::cerr << "Errore: Impossibile aprire " << filepath << std::endl;
        return 1;
    }

    TimeSeriesSoA data(2'000'000);

    std::string line;
    std::getline(file, line);

    while (std::getline(file, line)) {
        if (line.empty()) continue;

        std::stringstream ss(line);
        std::string token;

        std::getline(ss, token, ';');

        if (std::getline(ss, token, ';')) {
            if (!token.empty()) {
                try { data.historical_data_c1.push_back(std::stof(token)); }
                catch (...) { /* token malformato, manterrà il vecchio valore */ }
            }
        }
        if (std::getline(ss, token, ';')) {
            if (!token.empty()) {
                try { data.historical_data_c2.push_back(std::stof(token)); }
                catch (...) { /* token malformato, manterrà il vecchio valore */ }
            }
        }
        if (std::getline(ss, token, ';')) {
            if (!token.empty()) {
                try { data.historical_data_c3.push_back(std::stof(token)); }
                catch (...) { /* token malformato, manterrà il vecchio valore */ }
            }
        }
        if (std::getline(ss, token, ';')) {
            if (!token.empty()) {
                try { data.historical_data_c4.push_back(std::stof(token)); }
                catch (...) { /* token malformato, manterrà il vecchio valore */ }
            }
        }
        if (std::getline(ss, token, ';')) {
            if (!token.empty()) {
                try { data.historical_data_c5.push_back(std::stof(token)); }
                catch (...) { /* token malformato, manterrà il vecchio valore */ }
            }
        }
    }
    file.close();
    data.historical_data_c1.shrink_to_fit();
    data.historical_data_c2.shrink_to_fit();
    data.historical_data_c3.shrink_to_fit();
    data.historical_data_c4.shrink_to_fit();
    data.historical_data_c5.shrink_to_fit();
    return data;
}

double Average(const std::vector<double> &values) {
    if (values.empty()) {
        return 0.0;
    }
    return std::accumulate(values.begin(), values.end(), 0.0) / static_cast<double>(values.size());
}

bool SameResults(const SADResults &a, const SADResults &b, float atol) {
    if (a.best_indices != b.best_indices) {
        return false;
    }

    if (a.best_sads.size() != b.best_sads.size()) {
        return false;
    }

    for (std::size_t i = 0; i < a.best_sads.size(); ++i) {
        if (std::fabs(a.best_sads[i] - b.best_sads[i]) > atol) {
            return false;
        }
    }

    return true;
}
