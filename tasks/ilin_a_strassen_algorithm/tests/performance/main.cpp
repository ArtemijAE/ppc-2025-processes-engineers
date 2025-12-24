#include <gtest/gtest.h>

#include <cstddef>
#include <random>

#include "ilin_a_strassen_algorithm/common/include/common.hpp"
#include "ilin_a_strassen_algorithm/mpi/include/ops_mpi.hpp"
#include "ilin_a_strassen_algorithm/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace ilin_a_strassen_algorithm {

class IlinARunPerfTestProcesses : public ppc::util::BaseRunPerfTests<InType, OutType> {
 protected:
  static constexpr int kMatrixSize = 1000;
  InType input_data{};

  void SetUp() override {
    GenerateTestData(kMatrixSize);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return (output_data.size == kMatrixSize &&
            output_data.C.size() == static_cast<std::size_t>(kMatrixSize) * static_cast<std::size_t>(kMatrixSize));
  }

  InType GetTestInputData() final {
    return input_data;
  }

 private:
  void GenerateTestData(const int size) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(1.0, 10.0);

    input_data.A.resize(static_cast<std::size_t>(size) * static_cast<std::size_t>(size));
    input_data.B.resize(static_cast<std::size_t>(size) * static_cast<std::size_t>(size));
    input_data.size = size;

    for (int i = 0; i < size * size; ++i) {
      input_data.A[static_cast<std::size_t>(i)] = dist(gen);
      input_data.B[static_cast<std::size_t>(i)] = dist(gen);
    }
  }
};

TEST_P(IlinARunPerfTestProcesses, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, IlinAStrassenAlgorithmMPI, IlinAStrassenAlgorithmSEQ>(
    PPC_SETTINGS_ilin_a_strassen_algorithm);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = IlinARunPerfTestProcesses::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, IlinARunPerfTestProcesses, kGtestValues, kPerfTestName);

}  // namespace ilin_a_strassen_algorithm
