#include "ilin_a_strassen_algorithm/seq/include/ops_seq.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <vector>

#include "ilin_a_strassen_algorithm/common/include/common.hpp"
#include "util/include/util.hpp"

namespace ilin_a_strassen_algorithm {

IlinAStrassenAlgorithmSEQ::IlinAStrassenAlgorithmSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput().size = 0;
  GetOutput().C.clear();
  original_size_ = 0;
  padded_size_ = 0;
}

bool IlinAStrassenAlgorithmSEQ::ValidationImpl() {
  GetOutput().size = 0;
  GetOutput().C.clear();

  const auto &input = GetInput();

  if (input.size <= 0) {
    return false;
  }
  if (input.A.size() != static_cast<size_t>(input.size * input.size)) {
    return false;
  }
  if (input.B.size() != static_cast<size_t>(input.size * input.size)) {
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

std::vector<double> IlinAStrassenAlgorithmSEQ::naiveMultiply(const std::vector<double> &A, const std::vector<double> &B,
                                                             int n) {
  std::vector<double> C(n * n, 0.0);

  for (int i = 0; i < n; ++i) {
    for (int k = 0; k < n; ++k) {
      double aik = A[i * n + k];
      for (int j = 0; j < n; ++j) {
        C[i * n + j] += aik * B[k * n + j];
      }
    }
  }
  return C;
}

std::vector<std::vector<double>> IlinAStrassenAlgorithmSEQ::computeStrassenProducts(
    const std::vector<double> &A11, const std::vector<double> &A12, const std::vector<double> &A21,
    const std::vector<double> &A22, const std::vector<double> &B11, const std::vector<double> &B12,
    const std::vector<double> &B21, const std::vector<double> &B22, int half) {
  std::vector<std::vector<double>> products(7);
  std::vector<double> temp1(half * half), temp2(half * half);

  addMatrix(A11, A22, temp1, half);
  addMatrix(B11, B22, temp2, half);
  products[0] = strassenMultiply(temp1, temp2, half);

  addMatrix(A21, A22, temp1, half);
  products[1] = strassenMultiply(temp1, B11, half);

  subtractMatrix(B12, B22, temp1, half);
  products[2] = strassenMultiply(A11, temp1, half);

  subtractMatrix(B21, B11, temp1, half);
  products[3] = strassenMultiply(A22, temp1, half);

  addMatrix(A11, A12, temp1, half);
  products[4] = strassenMultiply(temp1, B22, half);

  subtractMatrix(A21, A11, temp1, half);
  addMatrix(B11, B12, temp2, half);
  products[5] = strassenMultiply(temp1, temp2, half);

  subtractMatrix(A12, A22, temp1, half);
  addMatrix(B21, B22, temp2, half);
  products[6] = strassenMultiply(temp1, temp2, half);

  return products;
}

void IlinAStrassenAlgorithmSEQ::computeResultSubmatrices(const std::vector<std::vector<double>> &products,
                                                         std::vector<double> &C11, std::vector<double> &C12,
                                                         std::vector<double> &C21, std::vector<double> &C22, int half) {
  const auto &P1 = products[0];
  const auto &P2 = products[1];
  const auto &P3 = products[2];
  const auto &P4 = products[3];
  const auto &P5 = products[4];
  const auto &P6 = products[5];
  const auto &P7 = products[6];

  addMatrix(P1, P4, C11, half);
  subtractMatrix(C11, P5, C11, half);
  addMatrix(C11, P7, C11, half);

  addMatrix(P3, P5, C12, half);

  addMatrix(P2, P4, C21, half);

  addMatrix(P1, P3, C22, half);
  subtractMatrix(C22, P2, C22, half);
  addMatrix(C22, P6, C22, half);
}

std::vector<double> IlinAStrassenAlgorithmSEQ::strassenMultiply(const std::vector<double> &A,
                                                                const std::vector<double> &B, int n) {
  if (n <= 64) {
    return naiveMultiply(A, B, n);
  }

  int half = n / 2;

  std::vector<double> A11(half * half), A12(half * half), A21(half * half), A22(half * half);
  std::vector<double> B11(half * half), B12(half * half), B21(half * half), B22(half * half);

  splitMatrix(A, A11, A12, A21, A22, n);
  splitMatrix(B, B11, B12, B21, B22, n);

  auto products = computeStrassenProducts(A11, A12, A21, A22, B11, B12, B21, B22, half);

  std::vector<double> C11(half * half), C12(half * half), C21(half * half), C22(half * half);

  computeResultSubmatrices(products, C11, C12, C21, C22, half);

  std::vector<double> C(n * n);
  joinMatrix(C, C11, C12, C21, C22, n);

  return C;
}

bool IlinAStrassenAlgorithmSEQ::RunImpl() {
  const auto &input = GetInput();
  auto &output = GetOutput();

  int n = input.size;

  if (n <= 64) {
    output.C = naiveMultiply(input.A, input.B, n);
    output.size = n;
  } else {
    std::vector<double> A_padded(padded_size_ * padded_size_, 0.0);
    std::vector<double> B_padded(padded_size_ * padded_size_, 0.0);

    for (int i = 0; i < original_size_; ++i) {
      for (int j = 0; j < original_size_; ++j) {
        A_padded[i * padded_size_ + j] = input.A[i * n + j];
        B_padded[i * padded_size_ + j] = input.B[i * n + j];
      }
    }

    std::vector<double> C_padded = strassenMultiply(A_padded, B_padded, padded_size_);

    output.C.resize(n * n);
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < n; ++j) {
        output.C[i * n + j] = C_padded[i * padded_size_ + j];
      }
    }
    output.size = n;
  }

  return true;
}

bool IlinAStrassenAlgorithmSEQ::PostProcessingImpl() {
  return true;
}

void IlinAStrassenAlgorithmSEQ::addMatrix(const std::vector<double> &A, const std::vector<double> &B,
                                          std::vector<double> &C, int n) {
  for (int i = 0; i < n * n; ++i) {
    C[i] = A[i] + B[i];
  }
}

void IlinAStrassenAlgorithmSEQ::subtractMatrix(const std::vector<double> &A, const std::vector<double> &B,
                                               std::vector<double> &C, int n) {
  for (int i = 0; i < n * n; ++i) {
    C[i] = A[i] - B[i];
  }
}

void IlinAStrassenAlgorithmSEQ::splitMatrix(const std::vector<double> &A, std::vector<double> &A11,
                                            std::vector<double> &A12, std::vector<double> &A21,
                                            std::vector<double> &A22, int n) {
  int half = n / 2;
  for (int i = 0; i < half; ++i) {
    for (int j = 0; j < half; ++j) {
      A11[i * half + j] = A[i * n + j];
      A12[i * half + j] = A[i * n + (j + half)];
      A21[i * half + j] = A[(i + half) * n + j];
      A22[i * half + j] = A[(i + half) * n + (j + half)];
    }
  }
}

void IlinAStrassenAlgorithmSEQ::joinMatrix(std::vector<double> &A, const std::vector<double> &A11,
                                           const std::vector<double> &A12, const std::vector<double> &A21,
                                           const std::vector<double> &A22, int n) {
  int half = n / 2;
  for (int i = 0; i < half; ++i) {
    for (int j = 0; j < half; ++j) {
      A[i * n + j] = A11[i * half + j];
      A[i * n + (j + half)] = A12[i * half + j];
      A[(i + half) * n + j] = A21[i * half + j];
      A[(i + half) * n + (j + half)] = A22[i * half + j];
    }
  }
}

}  // namespace ilin_a_strassen_algorithm
