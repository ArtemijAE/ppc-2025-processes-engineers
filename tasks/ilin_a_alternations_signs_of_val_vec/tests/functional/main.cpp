#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <random>
#include <string>
#include <tuple>

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
      CreateAlternatingData(vector_size);
    } else if (type == "mixed_alternating") {
      CreateMixedAlternatingData(vector_size);
    } else if (type == "all_positive") {
      CreateAllPositiveData(vector_size);
    } else if (type == "with_zeros") {
      CreateWithZerosData(vector_size);
    } else if (type == "random_large") {
      CreateRandomLargeData(vector_size);
    } else if (type == "zeros") {
      CreateZerosData(vector_size);
    }
  }

 private:
  void CreateAlternatingData(int vector_size) {
    for (int i = 0; i < vector_size; ++i) {
      input_data_.push_back(i % 2 == 0 ? i + 1 : -i - 1);
    }
  }

  void CreateAllPositiveData(int vector_size) {
    for (int i = 0; i < vector_size; ++i) {
      input_data_.push_back(i + 1);
    }
  }

  void CreateZerosData(int vector_size) {
    input_data_.resize(vector_size, 0);
  }

  void CreateMixedAlternatingData(int vector_size) {
    for (int i = 0; i < vector_size; ++i) {
      if (i % 4 == 0) {
        input_data_.push_back(5 + i);
      } else if (i % 4 == 1) {
        input_data_.push_back(6 + i);
      } else if (i % 4 == 2) {
        input_data_.push_back(-7 - i);
      } else {
        input_data_.push_back(8 + i);
      }
    }
  }

  void CreateWithZerosData(int vector_size) {
    for (int i = 0; i < vector_size; ++i) {
      if (i % 6 == 0) {
        input_data_.push_back(5);
      } else if (i % 6 == 1) {
        input_data_.push_back(0);
      } else if (i % 6 == 2) {
        input_data_.push_back(-7);
      } else if (i % 6 == 3) {
        input_data_.push_back(1);
      } else if (i % 6 == 4) {
        input_data_.push_back(0);
      } else {
        input_data_.push_back(9);
      }
    }
  }

  void CreateRandomLargeData(int vector_size) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(-100, 100);

    for (int i = 0; i < vector_size; ++i) {
      input_data_.push_back(dis(gen));
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return output_data >= 0;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

  InType input_data_;
};

namespace {

TEST_P(IlinARunFuncTestsProcesses, AlternationsSigns) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 8> kTestParam = {
    std::make_tuple(4, "alternating"),  std::make_tuple(5, "mixed_alternating"), std::make_tuple(7, "all_positive"),
    std::make_tuple(6, "with_zeros"),   std::make_tuple(100, "random_large"),    std::make_tuple(0, "zeros"),
    std::make_tuple(1, "all_positive"), std::make_tuple(2, "alternating")};

const auto kTestTasksList = std::tuple_cat(ppc::util::AddFuncTask<IlinAAlternationsSignsOfValVecMPI, InType>(
                                               kTestParam, PPC_SETTINGS_ilin_a_alternations_signs_of_val_vec),
                                           ppc::util::AddFuncTask<IlinAAlternationsSignsOfValVecSEQ, InType>(
                                               kTestParam, PPC_SETTINGS_ilin_a_alternations_signs_of_val_vec));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = IlinARunFuncTestsProcesses::PrintFuncTestName<IlinARunFuncTestsProcesses>;

INSTANTIATE_TEST_SUITE_P(AlternationsTests, IlinARunFuncTestsProcesses, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace ilin_a_alternations_signs_of_val_vec
