#include <gtest/gtest.h>

#include "ilin_a_alternations_signs_of_val_vec/common/include/common.hpp"
#include "ilin_a_alternations_signs_of_val_vec/mpi/include/ops_mpi.hpp"
#include "ilin_a_alternations_signs_of_val_vec/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace ilin_a_alternations_signs_of_val_vec {

class IlinARunPerfTestProcesses : public ppc::util::BaseRunPerfTests<InType, OutType> {
  const int kVectorSize_ = 15000000;
  InType input_data_;

  void SetUp() override {
    input_data_.clear();
    input_data_.reserve(kVectorSize_);

    for (int i = 0; i < kVectorSize_; ++i) {
      input_data_.push_back(((i * 17) % 201) - 100);
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return output_data >= 0 && output_data < kVectorSize_;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(IlinARunPerfTestProcesses, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, IlinAAlternationsSignsOfValVecMPI, IlinAAlternationsSignsOfValVecSEQ>(
        PPC_SETTINGS_ilin_a_alternations_signs_of_val_vec);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = IlinARunPerfTestProcesses::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, IlinARunPerfTestProcesses, kGtestValues, kPerfTestName);

}  // namespace ilin_a_alternations_signs_of_val_vec
