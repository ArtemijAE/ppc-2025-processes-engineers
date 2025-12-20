#include "ilin_a_gaussian_method_horizontal_band_scheme/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

#include "ilin_a_gaussian_method_horizontal_band_scheme/common/include/common.hpp"

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

  size_t expected_count = 2 + (size * band_width) + size;
  return input.size() == expected_count;
}

bool IlinAGaussianMethodSEQ::PreProcessingImpl() {
  const auto &input = GetInput();

  data_.size = static_cast<int>(input[0]);
  data_.band_width = static_cast<int>(input[1]);

  auto mat_size = static_cast<size_t>(data_.size) * static_cast<size_t>(data_.band_width);
  auto vec_size = static_cast<size_t>(data_.size);

  data_.matrix.resize(mat_size);
  data_.vector.resize(vec_size);

  std::copy(input.begin() + 2, input.begin() + 2 + static_cast<int>(mat_size), data_.matrix.begin());

  std::copy(input.begin() + 2 + static_cast<int>(mat_size), input.end(), data_.vector.begin());

  solution_.resize(vec_size, 0.0);
  GetOutput() = std::vector<double>(vec_size, 0.0);

  return true;
}

bool IlinAGaussianMethodSEQ::RunImpl() {
  const int n = data_.size;
  const int m = data_.band_width;

  std::vector<double> b = data_.vector;
  std::vector<double> matrix = data_.matrix;

  ForwardElimination(n, m, matrix, b);
  BackwardSubstitution(n, m, matrix, b);

  return true;
}

void IlinAGaussianMethodSEQ::ForwardElimination(int n, int m, std::vector<double> &matrix, std::vector<double> &b) {
  for (int k = 0; k < n; ++k) {
    int max_row = FindPivotRow(k, n, m, matrix);

    if (max_row != k) {
      SwapRows(k, max_row, m, matrix, b);
    }

    int diag_idx = m - 1;
    double pivot = matrix[(static_cast<size_t>(k) * static_cast<size_t>(m)) + diag_idx];
    if (std::fabs(pivot) < 1e-12) {
      continue;
    }

    for (int i = k + 1; i < std::min(n, k + m); ++i) {
      EliminateRow(i, k, m, matrix, b, pivot);
    }
  }
}

int IlinAGaussianMethodSEQ::FindPivotRow(int k, int n, int m, const std::vector<double> &matrix) {
  int max_row = k;
  double max_val = 0.0;

  for (int i = k; i < std::min(n, k + m); ++i) {
    int diag_idx = m - 1 - (i - k);
    if (diag_idx >= 0) {
      double val = std::fabs(matrix[(static_cast<size_t>(i) * static_cast<size_t>(m)) + diag_idx]);
      if (val > max_val) {
        max_val = val;
        max_row = i;
      }
    }
  }
  return max_row;
}

void IlinAGaussianMethodSEQ::SwapRows(int row1, int row2, int m, std::vector<double> &matrix, std::vector<double> &b) {
  for (int j = 0; j < m; ++j) {
    std::swap(matrix[(static_cast<size_t>(row1) * static_cast<size_t>(m)) + j],
              matrix[(static_cast<size_t>(row2) * static_cast<size_t>(m)) + j]);
  }
  std::swap(b[static_cast<size_t>(row1)], b[static_cast<size_t>(row2)]);
}

void IlinAGaussianMethodSEQ::EliminateRow(int i, int k, int m, std::vector<double> &matrix, std::vector<double> &b,
                                          double pivot) {
  int factor_idx = m - 1 - (i - k);
  if (factor_idx < 0) {
    return;
  }

  double factor = matrix[(static_cast<size_t>(i) * static_cast<size_t>(m)) + factor_idx] / pivot;

  for (int j = 0; j < m; ++j) {
    int src_idx = j - (i - k);
    if (src_idx >= 0 && src_idx < m) {
      matrix[(static_cast<size_t>(i) * static_cast<size_t>(m)) + j] -=
          factor * matrix[(static_cast<size_t>(k) * static_cast<size_t>(m)) + src_idx];
    }
  }

  b[static_cast<size_t>(i)] -= factor * b[static_cast<size_t>(k)];
}

void IlinAGaussianMethodSEQ::BackwardSubstitution(int n, int m, const std::vector<double> &matrix,
                                                  const std::vector<double> &b) {
  for (int i = n - 1; i >= 0; --i) {
    double sum = 0.0;

    for (int j = i + 1; j < std::min(n, i + m); ++j) {
      int idx = m - 1 + (j - i);
      if (idx < m) {
        sum += matrix[(static_cast<size_t>(i) * static_cast<size_t>(m)) + idx] * solution_[static_cast<size_t>(j)];
      }
    }

    int diag_idx = m - 1;
    double diag = matrix[(static_cast<size_t>(i) * static_cast<size_t>(m)) + diag_idx];
    if (std::fabs(diag) > 1e-12) {
      solution_[static_cast<size_t>(i)] = (b[static_cast<size_t>(i)] - sum) / diag;
    } else {
      solution_[static_cast<size_t>(i)] = 0.0;
    }
  }
}

bool IlinAGaussianMethodSEQ::PostProcessingImpl() {
  GetOutput() = solution_;
  return true;
}

}  // namespace ilin_a_gaussian_method_horizontal_band_scheme
