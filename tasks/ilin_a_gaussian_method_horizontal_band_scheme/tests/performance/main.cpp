#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <random>
#include <vector>

#include "ilin_a_gaussian_method_horizontal_band_scheme/common/include/common.hpp"
#include "ilin_a_gaussian_method_horizontal_band_scheme/mpi/include/ops_mpi.hpp"
#include "ilin_a_gaussian_method_horizontal_band_scheme/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace ilin_a_gaussian_method_horizontal_band_scheme {

class IlinARunPerfTestProcesses : public ppc::util::BaseRunPerfTests<InType, OutType> {
  const int kSize_ = 10000;
  InType input_data_;

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

    int band_width = std::max(1, size / 100);
    band_width = std::min(band_width, size);

    std::vector<double> matrix(static_cast<size_t>(size) * static_cast<size_t>(band_width), 0.0);
    std::vector<double> vector(static_cast<size_t>(size), 0.0);
    std::vector<double> expected_solution(static_cast<size_t>(size));

    for (int i = 0; i < size; ++i) {
      expected_solution[static_cast<size_t>(i)] = static_cast<double>(i + 1);
    }

    for (int i = 0; i < size; ++i) {
      double diag_sum = 0.0;
      for (int offset = 1; offset < std::min(band_width, 10); ++offset) {
        if (i - offset >= 0) {
          double val = dist(gen) * 0.1;
          int band_idx = band_width - 1 - offset;
          matrix[(static_cast<size_t>(i) * static_cast<size_t>(band_width)) + band_idx] = val;
          diag_sum += std::fabs(val);
        }
      }
      int diag_band_idx = band_width - 1;
      matrix[(static_cast<size_t>(i) * static_cast<size_t>(band_width)) + diag_band_idx] = diag_sum + dist(gen) + 10.0;
    }
    for (int i = 0; i < size; ++i) {
      double sum = matrix[(static_cast<size_t>(i) * static_cast<size_t>(band_width)) + (band_width - 1)] *
                   expected_solution[static_cast<size_t>(i)];

      for (int offset = 1; offset < std::min(band_width, 10); ++offset) {
        if (i - offset >= 0) {
          int band_idx = band_width - 1 - offset;
          sum += matrix[(static_cast<size_t>(i) * static_cast<size_t>(band_width)) + band_idx] *
                 expected_solution[static_cast<size_t>(i - offset)];
        }
      }

      vector[static_cast<size_t>(i)] = sum;
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
