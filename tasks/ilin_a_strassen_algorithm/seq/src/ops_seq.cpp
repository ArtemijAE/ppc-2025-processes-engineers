#include "ilin_a_strassen_algorithm/seq/include/ops_seq.hpp"

#include <cmath>
#include <vector>

#include "ilin_a_strassen_algorithm/common/include/common.hpp"

namespace ilin_a_strassen_algorithm {

IlinAStrassenAlgorithmSEQ::IlinAStrassenAlgorithmSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput().size = 0;
  GetOutput().C.clear();
}

bool IlinAStrassenAlgorithmSEQ::ValidationImpl() {
  GetOutput().size = 0;
  GetOutput().C.clear();

  const auto &input = GetInput();

  if (input.size <= 0) {
    return false;
  }
  if (input.A.size() != static_cast<std::size_t>(input.size) * static_cast<std::size_t>(input.size)) {
    return false;
  }
  if (input.B.size() != static_cast<std::size_t>(input.size) * static_cast<std::size_t>(input.size)) {
    return false;
  }

  return true;
}

bool IlinAStrassenAlgorithmSEQ::PreProcessingImpl() {
  int n = GetInput().size;
  int power = 1;
  while (power < n) {
    power *= 2;
  }

  original_size_ = n;
  padded_size_ = power;

  return true;
}

std::vector<double> IlinAStrassenAlgorithmSEQ::NaiveMultiply(const std::vector<double> &a, const std::vector<double> &b,
                                                             int n) {
  std::vector<double> c(static_cast<std::size_t>(n) * static_cast<std::size_t>(n), 0.0);

  for (int i = 0; i < n; ++i) {
    for (int k = 0; k < n; ++k) {
      double aik = a[(i * n) + k];
      for (int j = 0; j < n; ++j) {
        c[(i * n) + j] += aik * b[(k * n) + j];
      }
    }
  }
  return c;
}

std::vector<std::vector<double>> IlinAStrassenAlgorithmSEQ::ComputeStrassenProducts(
    const std::vector<double> &a11, const std::vector<double> &a12, const std::vector<double> &a21,
    const std::vector<double> &a22, const std::vector<double> &b11, const std::vector<double> &b12,
    const std::vector<double> &b21, const std::vector<double> &b22, int half) {
  std::vector<std::vector<double>> products(7);
  std::vector<double> temp1(static_cast<std::size_t>(half) * static_cast<std::size_t>(half));
  std::vector<double> temp2(static_cast<std::size_t>(half) * static_cast<std::size_t>(half));

  AddMatrix(a11, a22, temp1, half);
  AddMatrix(b11, b22, temp2, half);
  products[0] = StrassenMultiply(temp1, temp2, half);

  AddMatrix(a21, a22, temp1, half);
  products[1] = StrassenMultiply(temp1, b11, half);

  SubtractMatrix(b12, b22, temp1, half);
  products[2] = StrassenMultiply(a11, temp1, half);

  SubtractMatrix(b21, b11, temp1, half);
  products[3] = StrassenMultiply(a22, temp1, half);

  AddMatrix(a11, a12, temp1, half);
  products[4] = StrassenMultiply(temp1, b22, half);

  SubtractMatrix(a21, a11, temp1, half);
  AddMatrix(b11, b12, temp2, half);
  products[5] = StrassenMultiply(temp1, temp2, half);

  SubtractMatrix(a12, a22, temp1, half);
  AddMatrix(b21, b22, temp2, half);
  products[6] = StrassenMultiply(temp1, temp2, half);

  return products;
}

void IlinAStrassenAlgorithmSEQ::ComputeResultSubmatrices(const std::vector<std::vector<double>> &products,
                                                         std::vector<double> &c11, std::vector<double> &c12,
                                                         std::vector<double> &c21, std::vector<double> &c22, int half) {
  const auto &p1 = products[0];
  const auto &p2 = products[1];
  const auto &p3 = products[2];
  const auto &p4 = products[3];
  const auto &p5 = products[4];
  const auto &p6 = products[5];
  const auto &p7 = products[6];

  AddMatrix(p1, p4, c11, half);
  SubtractMatrix(c11, p5, c11, half);
  AddMatrix(c11, p7, c11, half);

  AddMatrix(p3, p5, c12, half);

  AddMatrix(p2, p4, c21, half);

  AddMatrix(p1, p3, c22, half);
  SubtractMatrix(c22, p2, c22, half);
  AddMatrix(c22, p6, c22, half);
}

std::vector<double> IlinAStrassenAlgorithmSEQ::StrassenMultiply(const std::vector<double> &a,
                                                                const std::vector<double> &b, int n) {
  if (n <= kThreshold) {
    return NaiveMultiply(a, b, n);
  }

  int half = n / 2;
  std::size_t half_sq = static_cast<std::size_t>(half) * static_cast<std::size_t>(half);

  std::vector<double> a11(half_sq);
  std::vector<double> a12(half_sq);
  std::vector<double> a21(half_sq);
  std::vector<double> a22(half_sq);
  std::vector<double> b11(half_sq);
  std::vector<double> b12(half_sq);
  std::vector<double> b21(half_sq);
  std::vector<double> b22(half_sq);

  SplitMatrix(a, a11, a12, a21, a22, n);
  SplitMatrix(b, b11, b12, b21, b22, n);

  auto products = ComputeStrassenProducts(a11, a12, a21, a22, b11, b12, b21, b22, half);

  std::vector<double> c11(half_sq);
  std::vector<double> c12(half_sq);
  std::vector<double> c21(half_sq);
  std::vector<double> c22(half_sq);

  ComputeResultSubmatrices(products, c11, c12, c21, c22, half);

  std::vector<double> c(static_cast<std::size_t>(n) * static_cast<std::size_t>(n));
  JoinMatrix(c, c11, c12, c21, c22, n);

  return c;
}

bool IlinAStrassenAlgorithmSEQ::RunImpl() {
  const auto &input = GetInput();
  auto &output = GetOutput();

  int n = input.size;

  if (n <= kThreshold) {
    output.C = NaiveMultiply(input.A, input.B, n);
    output.size = n;
  } else {
    std::size_t padded_size_sq = static_cast<std::size_t>(padded_size_) * static_cast<std::size_t>(padded_size_);
    std::vector<double> a_padded(padded_size_sq, 0.0);
    std::vector<double> b_padded(padded_size_sq, 0.0);

    for (int i = 0; i < original_size_; ++i) {
      for (int j = 0; j < original_size_; ++j) {
        a_padded[(i * padded_size_) + j] = input.A[(i * n) + j];
        b_padded[(i * padded_size_) + j] = input.B[(i * n) + j];
      }
    }

    std::vector<double> c_padded = StrassenMultiply(a_padded, b_padded, padded_size_);

    output.C.resize(static_cast<std::size_t>(n) * static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < n; ++j) {
        output.C[(i * n) + j] = c_padded[(i * padded_size_) + j];
      }
    }
    output.size = n;
  }

  return true;
}

bool IlinAStrassenAlgorithmSEQ::PostProcessingImpl() {
  return true;
}

void IlinAStrassenAlgorithmSEQ::AddMatrix(const std::vector<double> &a, const std::vector<double> &b,
                                          std::vector<double> &c, int n) {
  for (int i = 0; i < n * n; ++i) {
    c[i] = a[i] + b[i];
  }
}

void IlinAStrassenAlgorithmSEQ::SubtractMatrix(const std::vector<double> &a, const std::vector<double> &b,
                                               std::vector<double> &c, int n) {
  for (int i = 0; i < n * n; ++i) {
    c[i] = a[i] - b[i];
  }
}

void IlinAStrassenAlgorithmSEQ::SplitMatrix(const std::vector<double> &a, std::vector<double> &a11,
                                            std::vector<double> &a12, std::vector<double> &a21,
                                            std::vector<double> &a22, int n) {
  int half = n / 2;
  for (int i = 0; i < half; ++i) {
    for (int j = 0; j < half; ++j) {
      a11[(i * half) + j] = a[(i * n) + j];
      a12[(i * half) + j] = a[(i * n) + (j + half)];
      a21[(i * half) + j] = a[((i + half) * n) + j];
      a22[(i * half) + j] = a[((i + half) * n) + (j + half)];
    }
  }
}

void IlinAStrassenAlgorithmSEQ::JoinMatrix(std::vector<double> &a, const std::vector<double> &a11,
                                           const std::vector<double> &a12, const std::vector<double> &a21,
                                           const std::vector<double> &a22, int n) {
  int half = n / 2;
  for (int i = 0; i < half; ++i) {
    for (int j = 0; j < half; ++j) {
      a[(i * n) + j] = a11[(i * half) + j];
      a[(i * n) + (j + half)] = a12[(i * half) + j];
      a[((i + half) * n) + j] = a21[(i * half) + j];
      a[((i + half) * n) + (j + half)] = a22[(i * half) + j];
    }
  }
}

}  // namespace ilin_a_strassen_algorithm
