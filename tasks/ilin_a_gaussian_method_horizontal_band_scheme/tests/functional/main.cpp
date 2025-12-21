#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
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
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    int matrix_size = std::get<0>(params);
    std::string test_name = std::get<1>(params);

    int band_width = 0;
    if (test_name == "tiny") {
      band_width = 1;
    } else if (test_name == "small") {
      band_width = std::max(1, matrix_size / 4);
    } else if (test_name == "medium") {
      band_width = std::max(2, matrix_size / 3);
    } else if (test_name == "large") {
      band_width = std::max(3, matrix_size / 2);
    } else if (test_name == "singular") {
      band_width = std::max(2, matrix_size / 6);
    }

    band_width = std::min(band_width, matrix_size);
    band_width = std::max(band_width, 1);

    expected_output_.resize(static_cast<size_t>(matrix_size));
    for (int i = 0; i < matrix_size; ++i) {
      expected_output_[static_cast<size_t>(i)] = static_cast<double>(i + 1);
    }

    const auto mat_size = static_cast<size_t>(matrix_size) * static_cast<size_t>(band_width);
    std::vector<double> matrix(mat_size, 0.0);
    std::vector<double> vector(static_cast<size_t>(matrix_size), 0.0);

    if (test_name == "singular") {
      for (int i = 0; i < matrix_size; ++i) {
        int diag_band_idx = band_width - 1;
        if (i % 3 == 0) {
          matrix[(static_cast<size_t>(i) * static_cast<size_t>(band_width)) + diag_band_idx] = 1e-6;
        } else {
          matrix[(static_cast<size_t>(i) * static_cast<size_t>(band_width)) + diag_band_idx] = 1.0;
        }
      }
    } else {
      for (int i = 0; i < matrix_size; ++i) {
        double diag_sum = 0.0;

        for (int j = 0; j < matrix_size; ++j) {
          if (i == j) {
            continue;
          }

          if (j <= i) {
            int band_idx = (i - j + band_width - 1);
            if (band_idx >= 0 && band_idx < band_width) {
              double value = 0.1 / (std::abs(i - j) + 1.0);
              matrix[(static_cast<size_t>(i) * static_cast<size_t>(band_width)) + band_idx] = value;
              diag_sum += std::abs(value);
            }
          }
        }

        int diag_band_idx = band_width - 1;
        matrix[(static_cast<size_t>(i) * static_cast<size_t>(band_width)) + diag_band_idx] =
            diag_sum + matrix_size + 10.0;
      }
    }

    for (int i = 0; i < matrix_size; ++i) {
      double b = 0.0;
      for (int j = 0; j < matrix_size; ++j) {
        if (j <= i) {
          int band_idx = (i - j + band_width - 1);
          if (band_idx >= 0 && band_idx < band_width) {
            b += matrix[(static_cast<size_t>(i) * static_cast<size_t>(band_width)) + band_idx] *
                 expected_output_[static_cast<size_t>(j)];
          }
        }
      }
      vector[static_cast<size_t>(i)] = b;
    }

    input_data_.clear();
    input_data_.push_back(static_cast<double>(matrix_size));
    input_data_.push_back(static_cast<double>(band_width));
    input_data_.insert(input_data_.end(), matrix.begin(), matrix.end());
    input_data_.insert(input_data_.end(), vector.begin(), vector.end());
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (output_data.size() != expected_output_.size()) {
      return false;
    }

    const double tolerance = 1e-5;
    for (size_t i = 0; i < expected_output_.size(); ++i) {
      if (std::abs(output_data[i] - expected_output_[i]) > tolerance) {
        return false;
      }
    }
    return true;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  OutType expected_output_;
};

class IlinAValidationTests : public ::testing::Test {
 protected:
  static void TestValidation(const InType &input, bool expected_valid) {
    IlinAGaussianMethodSEQ task(input);
    bool result = task.Validation();
    EXPECT_EQ(result, expected_valid);
  }
};

TEST_F(IlinAValidationTests, ValidInput) {
  InType input;
  int size = 5;
  int band_width = 3;

  input.push_back(static_cast<double>(size));
  input.push_back(static_cast<double>(band_width));

  const auto mat_size = static_cast<size_t>(size) * static_cast<size_t>(band_width);
  std::vector<double> matrix(mat_size, 0.0);
  std::vector<double> vector(static_cast<size_t>(size), 1.0);

  for (int i = 0; i < size; ++i) {
    matrix[(i * band_width) + (band_width - 1)] = (i + 1) * 2.0;
  }

  input.insert(input.end(), matrix.begin(), matrix.end());
  input.insert(input.end(), vector.begin(), vector.end());

  TestValidation(input, true);
}

TEST_F(IlinAValidationTests, InvalidSizeZero) {
  InType input = {0.0, 3.0};
  TestValidation(input, false);
}

TEST_F(IlinAValidationTests, InvalidBandwidthZero) {
  InType input = {5.0, 0.0};
  TestValidation(input, false);
}

TEST_F(IlinAValidationTests, InvalidBandwidthLargerThanSize) {
  InType input = {5.0, 6.0};
  TestValidation(input, false);
}

TEST_F(IlinAValidationTests, InvalidInputTooSmall) {
  InType input = {5.0, 3.0};
  TestValidation(input, false);
}

TEST_F(IlinAValidationTests, BandwidthOneTest) {
  InType input;
  int size = 4;
  int band_width = 1;

  input.push_back(static_cast<double>(size));
  input.push_back(static_cast<double>(band_width));

  const auto mat_size = static_cast<size_t>(size) * static_cast<size_t>(band_width);
  std::vector<double> matrix(mat_size);
  std::vector<double> vector(static_cast<size_t>(size));

  for (int i = 0; i < size; ++i) {
    matrix[static_cast<size_t>(i)] = (i + 1) * 2.0;
    vector[static_cast<size_t>(i)] = (i + 1) * 4.0;
  }

  input.insert(input.end(), matrix.begin(), matrix.end());
  input.insert(input.end(), vector.begin(), vector.end());

  TestValidation(input, true);
}

TEST_F(IlinAValidationTests, FullBandwidthTest) {
  InType input;
  int size = 3;
  int band_width = 3;

  input.push_back(static_cast<double>(size));
  input.push_back(static_cast<double>(band_width));

  std::vector<double> matrix = {1.0, 0.0, 0.0, 2.0, 3.0, 0.0, 0.0, 4.0, 5.0};
  std::vector<double> vector = {1.0, 5.0, 9.0};

  input.insert(input.end(), matrix.begin(), matrix.end());
  input.insert(input.end(), vector.begin(), vector.end());

  TestValidation(input, true);
}

namespace {

TEST_P(IlinARunFuncTestsProcesses, GaussianMethod) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 9> kTestParam = {
    std::make_tuple(3, "tiny"),    std::make_tuple(5, "tiny"),     std::make_tuple(7, "small"),
    std::make_tuple(10, "small"),  std::make_tuple(20, "medium"),  std::make_tuple(50, "medium"),
    std::make_tuple(100, "large"), std::make_tuple(8, "singular"), std::make_tuple(15, "singular")};

const auto kTestTasksList = std::tuple_cat(ppc::util::AddFuncTask<IlinAGaussianMethodMPI, InType>(
                                               kTestParam, PPC_SETTINGS_ilin_a_gaussian_method_horizontal_band_scheme),
                                           ppc::util::AddFuncTask<IlinAGaussianMethodSEQ, InType>(
                                               kTestParam, PPC_SETTINGS_ilin_a_gaussian_method_horizontal_band_scheme));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = IlinARunFuncTestsProcesses::PrintFuncTestName<IlinARunFuncTestsProcesses>;

INSTANTIATE_TEST_SUITE_P(GaussianTests, IlinARunFuncTestsProcesses, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace ilin_a_gaussian_method_horizontal_band_scheme
