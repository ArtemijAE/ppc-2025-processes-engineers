#pragma once

#include "ilin_a_gaussian_method_horizontal_band_scheme/common/include/common.hpp"

namespace ilin_a_gaussian_method_horizontal_band_scheme {

class IlinAGaussianMethodSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit IlinAGaussianMethodSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  MatrixData data_;
  std::vector<double> solution_;
};

}  // namespace ilin_a_gaussian_method_horizontal_band_scheme
