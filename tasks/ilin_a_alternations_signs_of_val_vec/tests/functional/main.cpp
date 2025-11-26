#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <random>
#include <string>
#include <tuple>
#include <vector>

#include "ilin_a_alternations_signs_of_val_vec/common/include/common.hpp"
#include "ilin_a_alternations_signs_of_val_vec/mpi/include/ops_mpi.hpp"
#include "ilin_a_alternations_signs_of_val_vec/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace ilin_a_alternations_signs_of_val_vec {

class IlinARunFuncTestsProcesses : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::to_string(std::get<0>(test_param)) + "_" + std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    auto params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    int vector_size = std::get<0>(params);
    std::string type = std::get<1>(params);

    input_data_.clear();

    if (type == "alternating") {
      // Чередующиеся знаки: +, -, +, -, ...
      for (int i = 0; i < vector_size; ++i) {
        input_data_.push_back(i % 2 == 0 ? i + 1 : -i - 1);
      }
    } else if (type == "all_positive") {
      // Все положительные
      for (int i = 0; i < vector_size; ++i) {
        input_data_.push_back(i + 1);
      }
    } else if (type == "all_negative") {
      // Все отрицательные
      for (int i = 0; i < vector_size; ++i) {
        input_data_.push_back(-i - 1);
      }
    } else if (type == "random") {
      // Случайные значения
      std::random_device rd;
      std::mt19937 gen(rd());
      std::uniform_int_distribution<> dis(-100, 100);

      for (int i = 0; i < vector_size; ++i) {
        input_data_.push_back(dis(gen));
      }
    } else if (type == "zeros") {
      // Нули
      input_data_.resize(vector_size, 0);
    } else if (type == "mixed") {
      // Смешанные с нулями
      for (int i = 0; i < vector_size; ++i) {
        if (i % 3 == 0) {
          input_data_.push_back(0);
        } else if (i % 3 == 1) {
          input_data_.push_back(i + 1);
        } else {
          input_data_.push_back(-i - 1);
        }
      }
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return output_data >= 0;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
};

namespace {

TEST_P(IlinARunFuncTestsProcesses, AlternationsSigns) {
  ExecuteTest(GetParam());
}
}  // namespace

TEST(IlinARunFuncTestsProcesses, EmptyVector) {
  std::vector<int> input = {};
  ilin_a_alternations_signs_of_val_vec::IlinAAlternationsSignsOfValVecMPI task(input);
  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());
  EXPECT_EQ(task.GetOutput(), 0);
}

TEST(IlinARunFuncTestsProcesses, SingleElement) {
  std::vector<int> input = {5};
  ilin_a_alternations_signs_of_val_vec::IlinAAlternationsSignsOfValVecMPI task(input);
  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());
  EXPECT_EQ(task.GetOutput(), 0);
}

TEST(IlinARunFuncTestsProcesses, SingleProcessNoBoundaries) {
  std::vector<int> input = {1, -1, 2, -2};
  ilin_a_alternations_signs_of_val_vec::IlinAAlternationsSignsOfValVecMPI task(input);
  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());
  EXPECT_GT(task.GetOutput(), 0);
}

TEST(IlinARunFuncTestsProcesses, NoBoundaryAlternations) {
  std::vector<int> input = {1, 2, 3, 4, 5, 6};
  ilin_a_alternations_signs_of_val_vec::IlinAAlternationsSignsOfValVecMPI task(input);
  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());
  EXPECT_EQ(task.GetOutput(), 0);
}

namespace {
const std::array<TestType, 12> kTestParam = {
    std::make_tuple(10, "alternating"),   std::make_tuple(100, "alternating"),  std::make_tuple(1000, "alternating"),
    std::make_tuple(10, "all_positive"),  std::make_tuple(100, "all_positive"), std::make_tuple(10, "all_negative"),
    std::make_tuple(100, "all_negative"), std::make_tuple(50, "random"),        std::make_tuple(500, "random"),
    std::make_tuple(10, "zeros"),         std::make_tuple(1, "all_positive"),   std::make_tuple(0, "all_positive")};

const auto kTestTasksList = std::tuple_cat(ppc::util::AddFuncTask<IlinAAlternationsSignsOfValVecMPI, InType>(
                                               kTestParam, PPC_SETTINGS_ilin_a_alternations_signs_of_val_vec),
                                           ppc::util::AddFuncTask<IlinAAlternationsSignsOfValVecSEQ, InType>(
                                               kTestParam, PPC_SETTINGS_ilin_a_alternations_signs_of_val_vec));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = IlinARunFuncTestsProcesses::PrintFuncTestName<IlinARunFuncTestsProcesses>;

INSTANTIATE_TEST_SUITE_P(AlternationsTests, IlinARunFuncTestsProcesses, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace ilin_a_alternations_signs_of_val_vec
