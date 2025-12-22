#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <random>
#include <tuple>
#include <vector>

#include "ilin_a_strassen_algorithm/common/include/common.hpp"
#include "ilin_a_strassen_algorithm/mpi/include/ops_mpi.hpp"
#include "ilin_a_strassen_algorithm/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace ilin_a_strassen_algorithm {

class IlinARunFuncTestsProcesses
    : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::to_string(std::get<0>(test_param)) + "_" +
           std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    auto params = std::get<static_cast<std::size_t>(
        ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    matrix_size_ = std::get<0>(params);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(1.0, 10.0);

    input_data_.A.resize(matrix_size_ * matrix_size_);
    input_data_.B.resize(matrix_size_ * matrix_size_);
    input_data_.size = matrix_size_;

    gen.seed(42 + matrix_size_);
    
    for (int i = 0; i < matrix_size_ * matrix_size_; ++i) {
      input_data_.A[i] = dist(gen);
      input_data_.B[i] = dist(gen);
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (output_data.size != matrix_size_) {
      return false;
    }
    
    if (output_data.C.size() != static_cast<size_t>(matrix_size_ * matrix_size_)) {
      return false;
    }

    std::vector<double> reference(matrix_size_ * matrix_size_, 0.0);
    for (int i = 0; i < matrix_size_; ++i) {
      for (int j = 0; j < matrix_size_; ++j) {
        double sum = 0.0;
        for (int k = 0; k < matrix_size_; ++k) {
          sum += input_data_.A[i * matrix_size_ + k] * input_data_.B[k * matrix_size_ + j];
        }
        reference[i * matrix_size_ + j] = sum;
      }
    }

    double tolerance = 1e-6;
    for (int i = 0; i < matrix_size_ * matrix_size_; ++i) {
      if (std::abs(output_data.C[i] - reference[i]) > tolerance) {
        return false;
      }
    }
    
    return true;
  }

  InType GetTestInputData() final { 
    return input_data_; 
  }

 private:
  int matrix_size_ = 0;
  InType input_data_;
};

namespace {

TEST_P(IlinARunFuncTestsProcesses, StrassenAlgorithm) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 6> kTestParam = {
    std::make_tuple(16, "16"),
    std::make_tuple(32, "32"),
    std::make_tuple(64, "64"),
    std::make_tuple(65, "65"),
    std::make_tuple(128, "128"),
    std::make_tuple(127, "127")
};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<IlinAStrassenAlgorithmMPI, InType>(
        kTestParam, PPC_SETTINGS_ilin_a_strassen_algorithm),
    ppc::util::AddFuncTask<IlinAStrassenAlgorithmSEQ, InType>(
        kTestParam, PPC_SETTINGS_ilin_a_strassen_algorithm));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName =
    IlinARunFuncTestsProcesses::PrintFuncTestName<IlinARunFuncTestsProcesses>;

INSTANTIATE_TEST_SUITE_P(StrassenTests, IlinARunFuncTestsProcesses,
                         kGtestValues, kPerfTestName);

}  // namespace

}  // namespace ilin_a_strassen_algorithm