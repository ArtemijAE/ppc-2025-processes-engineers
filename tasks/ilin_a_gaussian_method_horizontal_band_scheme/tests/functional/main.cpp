#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <random>
#include <string>
#include <tuple>
#include <vector>

#include "ilin_a_gaussian_method_horizontal_band_scheme/common/include/common.hpp"
#include "ilin_a_gaussian_method_horizontal_band_scheme/mpi/include/ops_mpi.hpp"
#include "ilin_a_gaussian_method_horizontal_band_scheme/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace ilin_a_gaussian_method_horizontal_band_scheme {

class IlinARunFuncTestsProcesses : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::to_string(std::get<0>(test_param)) + "_" + std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    auto params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    int size = std::get<0>(params);

    GenerateTestData(size);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (output_data.size() != expected_solution_.size()) {
      return false;
    }

    const double tolerance = 1e-6;
    for (size_t i = 0; i < output_data.size(); ++i) {
      if (std::fabs(output_data[i] - expected_solution_[i]) > tolerance) {
        return false;
      }
    }

    return true;
  }

  InType GetTestInputData() final {
    return test_input_;
  }

 private:
  InType test_input_;
  std::vector<double> expected_solution_;

  void GenerateTestData(int size) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(1.0, 10.0);

    int band_width = std::max(1, size / 4);
    band_width = std::min(band_width, size);

    std::vector<double> matrix(static_cast<size_t>(size) * static_cast<size_t>(band_width), 0.0);
    std::vector<double> vector(static_cast<size_t>(size), 0.0);
    expected_solution_.resize(static_cast<size_t>(size));

    for (int i = 0; i < size; ++i) {
      expected_solution_[static_cast<size_t>(i)] = static_cast<double>(i + 1);
    }

    for (int i = 0; i < size; ++i) {
      double diag_sum = 0.0;

      for (int j = 0; j < size; ++j) {
        if (i == j) {
          continue;
        }

        if (j <= i) {
          int band_idx = (i - j + band_width - 1);
          if (band_idx >= 0 && band_idx < band_width) {
            double val = dist(gen) * 0.1;
            matrix[(static_cast<size_t>(i) * static_cast<size_t>(band_width)) + band_idx] = val;
            diag_sum += std::fabs(val);
          }
        }
      }

      int diag_band_idx = band_width - 1;
      matrix[(static_cast<size_t>(i) * static_cast<size_t>(band_width)) + diag_band_idx] = diag_sum + dist(gen) + 10.0;
    }

    for (int i = 0; i < size; ++i) {
      double sum = 0.0;
      for (int j = 0; j < size; ++j) {
        if (j <= i) {
          int band_idx = (i - j + band_width - 1);
          if (band_idx >= 0 && band_idx < band_width) {
            sum += matrix[(static_cast<size_t>(i) * static_cast<size_t>(band_width)) + band_idx] *
                   expected_solution_[static_cast<size_t>(j)];
          }
        }
      }
      vector[static_cast<size_t>(i)] = sum;
    }

    test_input_.clear();
    test_input_.push_back(static_cast<double>(size));
    test_input_.push_back(static_cast<double>(band_width));
    test_input_.insert(test_input_.end(), matrix.begin(), matrix.end());
    test_input_.insert(test_input_.end(), vector.begin(), vector.end());
  }
};

namespace {

TEST_P(IlinARunFuncTestsProcesses, GaussianMethod) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 3> kTestParam = {std::make_tuple(10, "small"), std::make_tuple(50, "medium"),
                                            std::make_tuple(100, "large")};

const auto kTestTasksList = std::tuple_cat(ppc::util::AddFuncTask<IlinAGaussianMethodMPI, InType>(
                                               kTestParam, PPC_SETTINGS_ilin_a_gaussian_method_horizontal_band_scheme),
                                           ppc::util::AddFuncTask<IlinAGaussianMethodSEQ, InType>(
                                               kTestParam, PPC_SETTINGS_ilin_a_gaussian_method_horizontal_band_scheme));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = IlinARunFuncTestsProcesses::PrintFuncTestName<IlinARunFuncTestsProcesses>;

INSTANTIATE_TEST_SUITE_P(GaussianTests, IlinARunFuncTestsProcesses, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace ilin_a_gaussian_method_horizontal_band_scheme
