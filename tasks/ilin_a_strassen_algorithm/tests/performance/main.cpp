#include <gtest/gtest.h>

#include <random>
#include <chrono>

#include "ilin_a_strassen_algorithm/common/include/common.hpp"
#include "ilin_a_strassen_algorithm/mpi/include/ops_mpi.hpp"
#include "ilin_a_strassen_algorithm/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace ilin_a_strassen_algorithm {

class IlinARunPerfTestProcesses
    : public ppc::util::BaseRunPerfTests<InType, OutType> {
 protected:
  const int kMatrixSize_ = 128;
  InType input_data_{};

  void SetUp() override {
    GenerateTestData(kMatrixSize_);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return (output_data.size == kMatrixSize_ &&
            output_data.C.size() == static_cast<size_t>(kMatrixSize_ * kMatrixSize_));
  }

  InType GetTestInputData() final { 
    return input_data_; 
  }

 private:
  void GenerateTestData(int size) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(1.0, 10.0);

    input_data_.A.resize(size * size);
    input_data_.B.resize(size * size);
    input_data_.size = size;

    gen.seed(42);
    
    for (int i = 0; i < size * size; ++i) {
      input_data_.A[i] = dist(gen);
      input_data_.B[i] = dist(gen);
    }
  }
};

TEST_P(IlinARunPerfTestProcesses, RunPerfModes) { 
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<
    InType, IlinAStrassenAlgorithmMPI, IlinAStrassenAlgorithmSEQ>(
    PPC_SETTINGS_ilin_a_strassen_algorithm);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = IlinARunPerfTestProcesses::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, IlinARunPerfTestProcesses, kGtestValues,
                         kPerfTestName);

}  // namespace ilin_a_strassen_algorithm