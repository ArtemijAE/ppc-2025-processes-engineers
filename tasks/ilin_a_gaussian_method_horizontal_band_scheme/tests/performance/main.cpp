#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

#include "ilin_a_gaussian_method_horizontal_band_scheme/common/include/common.hpp"
#include "ilin_a_gaussian_method_horizontal_band_scheme/mpi/include/ops_mpi.hpp"
#include "ilin_a_gaussian_method_horizontal_band_scheme/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace ilin_a_gaussian_method_horizontal_band_scheme {

class IlinARunPerfTestProcesses : public ppc::util::BaseRunPerfTests<InType, OutType> {
  const int kSize_ = 100;
  InType input_data_{};

  void SetUp() override {
    GenerateTestData(kSize_);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return !output_data.empty() && output_data.size() == static_cast<size_t>(kSize_);
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  void GenerateTestData(int size) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(1.0, 10.0);

    int band_width = std::min(10, size / 4 + 1);
    std::vector<double> matrix(size * band_width, 0.0);
    std::vector<double> vector(size, 0.0);
    std::vector<double> solution(size);

    for (int i = 0; i < size; ++i) {
      solution[i] = dist(gen);
    }

    for (int i = 0; i < size; ++i) {
      for (int j = 0; j < size; ++j) {
        int band_idx = (j - i + band_width - 1);
        if (band_idx >= 0 && band_idx < band_width) {
          if (i == j) {
            matrix[i * band_width + band_idx] = dist(gen) + band_width * 10.0;
          } else if (std::abs(i - j) < band_width) {
            matrix[i * band_width + band_idx] = dist(gen) * 0.1;
          }
        }
      }
    }

    for (int i = 0; i < size; ++i) {
      double sum = 0.0;
      for (int j = 0; j < size; ++j) {
        int band_idx = (j - i + band_width - 1);
        if (band_idx >= 0 && band_idx < band_width) {
          sum += matrix[i * band_width + band_idx] * solution[j];
        }
      }
      vector[i] = sum;
    }

    input_data_.clear();
    input_data_.push_back(static_cast<double>(size));
    input_data_.push_back(static_cast<double>(band_width));
    input_data_.insert(input_data_.end(), matrix.begin(), matrix.end());
    input_data_.insert(input_data_.end(), vector.begin(), vector.end());
  }
};

TEST_P(IlinARunPerfTestProcesses, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, IlinAGaussianMethodMPI, IlinAGaussianMethodSEQ>(
    PPC_SETTINGS_ilin_a_gaussian_method_horizontal_band_scheme);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = IlinARunPerfTestProcesses::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, IlinARunPerfTestProcesses, kGtestValues, kPerfTestName);

}  // namespace ilin_a_gaussian_method_horizontal_band_scheme
