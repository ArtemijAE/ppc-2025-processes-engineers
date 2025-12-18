#include "ilin_a_gaussian_method_horizontal_band_scheme/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>

namespace ilin_a_gaussian_method_horizontal_band_scheme {

IlinAGaussianMethodSEQ::IlinAGaussianMethodSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = std::vector<double>();
}

bool IlinAGaussianMethodSEQ::ValidationImpl() {
  const auto &input = GetInput();
  if (input.size() < 4) {
    return false;
  }

  int size = static_cast<int>(input[0]);
  int band_width = static_cast<int>(input[1]);

  if (size <= 0 || band_width <= 0 || band_width > size) {
    return false;
  }

  size_t expected_count = 2 + size * band_width + size;
  return input.size() == expected_count;
}

bool IlinAGaussianMethodSEQ::PreProcessingImpl() {
  const auto &input = GetInput();

  data_.size = static_cast<int>(input[0]);
  data_.band_width = static_cast<int>(input[1]);

  int mat_size = data_.size * data_.band_width;
  int vec_size = data_.size;

  data_.matrix.resize(mat_size);
  data_.vector.resize(vec_size);

  std::copy(input.begin() + 2, input.begin() + 2 + mat_size, data_.matrix.begin());

  std::copy(input.begin() + 2 + mat_size, input.end(), data_.vector.begin());

  solution_.resize(data_.size, 0.0);
  GetOutput() = std::vector<double>(data_.size, 0.0);

  return true;
}

bool IlinAGaussianMethodSEQ::RunImpl() {
  const int n = data_.size;
  const int m = data_.band_width;

  std::vector<double> b = data_.vector;
  std::vector<double> matrix = data_.matrix;

  for (int k = 0; k < n; ++k) {
    int pivot_row = k;
    double max_val = 0.0;

    int end_search = std::min(n, k + m);
    for (int i = k; i < end_search; ++i) {
      int diag_idx = m - 1;
      double val = std::fabs(matrix[i * m + diag_idx]);
      if (val > max_val) {
        max_val = val;
        pivot_row = i;
      }
    }

    if (pivot_row != k) {
      for (int j = 0; j < m; ++j) {
        std::swap(matrix[k * m + j], matrix[pivot_row * m + j]);
      }
      std::swap(b[k], b[pivot_row]);
    }

    double pivot = matrix[k * m + m - 1];
    if (std::fabs(pivot) < 1e-12) {
      continue;
    }

    int end_row = std::min(n, k + m);
    for (int i = k + 1; i < end_row; ++i) {
      int factor_idx = m - 1 - (i - k);
      if (factor_idx < 0) {
        continue;
      }

      double factor = matrix[i * m + factor_idx] / pivot;

      for (int j = 0; j < m; ++j) {
        int src_idx = j - (i - k);
        if (src_idx >= 0 && src_idx < m) {
          matrix[i * m + j] -= factor * matrix[k * m + src_idx];
        }
      }

      b[i] -= factor * b[k];
    }
  }

  for (int i = n - 1; i >= 0; --i) {
    double sum = 0.0;

    int end_col = std::min(n, i + m);
    for (int j = i + 1; j < end_col; ++j) {
      int idx = m - 1 + (j - i);
      if (idx < m) {
        sum += matrix[i * m + idx] * solution_[j];
      }
    }

    double diag = matrix[i * m + m - 1];
    solution_[i] = (b[i] - sum) / diag;
  }

  return true;
}

bool IlinAGaussianMethodSEQ::PostProcessingImpl() {
  GetOutput() = solution_;
  return true;
}

}  // namespace ilin_a_gaussian_method_horizontal_band_scheme
