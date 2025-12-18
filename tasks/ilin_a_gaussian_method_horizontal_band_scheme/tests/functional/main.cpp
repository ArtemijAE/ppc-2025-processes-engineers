#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
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

    for (size_t i = 0; i < output_data.size(); ++i) {
      if (std::fabs(output_data[i] - expected_solution_[i]) > 1e-6) {
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
    std::uniform_real_distribution<double> dist(0.5, 2.0);

    int band_width = std::max(3, size / 4);

    std::vector<double> matrix(size * band_width, 0.0);
    std::vector<double> vector(size, 0.0);
    expected_solution_.resize(size);

    for (int i = 0; i < size; ++i) {
      expected_solution_[i] = dist(gen);
    }

    for (int i = 0; i < size; ++i) {
      double sum = 0.0;
      int start_col = std::max(0, i - band_width + 1);
      int end_col = std::min(size - 1, i + band_width - 1);

      for (int j = start_col; j <= end_col; ++j) {
        int band_index = (j - i + band_width - 1);

        if (i == j) {
          matrix[i * band_width + band_index] = dist(gen) + band_width;
        } else {
          matrix[i * band_width + band_index] = dist(gen) / (std::abs(i - j) + 1);
        }

        sum += matrix[i * band_width + band_index] * expected_solution_[j];
      }

      vector[i] = sum;
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
