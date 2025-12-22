#include "ilin_a_strassen_algorithm/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <tuple>
#include <vector>

#include "ilin_a_strassen_algorithm/common/include/common.hpp"
#include "util/include/util.hpp"

namespace ilin_a_strassen_algorithm {

IlinAStrassenAlgorithmMPI::IlinAStrassenAlgorithmMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput().size = 0;
  GetOutput().C.clear();

  world_size_ = 0;
  world_rank_ = 0;
  original_size_ = 0;
  padded_size_ = 0;
  threshold_ = 64;
}

bool IlinAStrassenAlgorithmMPI::ValidationImpl() {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  GetOutput().size = 0;
  GetOutput().C.clear();

  if (rank == 0) {
    const auto &input = GetInput();

    int validation_result = 1;
    int size_to_bcast = input.size;

    if (input.size <= 0) {
      validation_result = 0;
    }
    if (input.A.size() != static_cast<size_t>(input.size * input.size)) {
      validation_result = 0;
    }
    if (input.B.size() != static_cast<size_t>(input.size * input.size)) {
      validation_result = 0;
    }

    MPI_Bcast(&validation_result, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (validation_result) {
      MPI_Bcast(&size_to_bcast, 1, MPI_INT, 0, MPI_COMM_WORLD);
      return true;
    } else {
      int error_size = -1;
      MPI_Bcast(&error_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
      return false;
    }
  } else {
    int validation_result;
    MPI_Bcast(&validation_result, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (validation_result) {
      int size;
      MPI_Bcast(&size, 1, MPI_INT, 0, MPI_COMM_WORLD);
      return size > 0;
    } else {
      int error_size;
      MPI_Bcast(&error_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
      return false;
    }
  }
}

bool IlinAStrassenAlgorithmMPI::PreProcessingImpl() {
  MPI_Comm_size(MPI_COMM_WORLD, &world_size_);
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank_);

  const auto &input = GetInput();

  if (world_rank_ == 0) {
    original_size_ = input.size;

    int power = 1;
    while (power < original_size_) {
      power *= 2;
    }
    padded_size_ = power;
  }

  MPI_Bcast(&original_size_, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&padded_size_, 1, MPI_INT, 0, MPI_COMM_WORLD);

  return true;
}

void IlinAStrassenAlgorithmMPI::addMatrix(const std::vector<double> &A, const std::vector<double> &B,
                                          std::vector<double> &C, int n) {
  for (int i = 0; i < n * n; ++i) {
    C[i] = A[i] + B[i];
  }
}

void IlinAStrassenAlgorithmMPI::subtractMatrix(const std::vector<double> &A, const std::vector<double> &B,
                                               std::vector<double> &C, int n) {
  for (int i = 0; i < n * n; ++i) {
    C[i] = A[i] - B[i];
  }
}

void IlinAStrassenAlgorithmMPI::splitMatrix(const std::vector<double> &A, std::vector<double> &A11,
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

void IlinAStrassenAlgorithmMPI::joinMatrix(std::vector<double> &A, const std::vector<double> &A11,
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

std::vector<double> IlinAStrassenAlgorithmMPI::naiveMultiplySeq(const std::vector<double> &A,
                                                                const std::vector<double> &B, int n) {
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

std::vector<double> IlinAStrassenAlgorithmMPI::strassenSequential(const std::vector<double> &A,
                                                                  const std::vector<double> &B, int n) {
  if (n <= threshold_) {
    return naiveMultiplySeq(A, B, n);
  }

  int half = n / 2;

  std::vector<double> A11(half * half), A12(half * half), A21(half * half), A22(half * half);
  std::vector<double> B11(half * half), B12(half * half), B21(half * half), B22(half * half);

  splitMatrix(A, A11, A12, A21, A22, n);
  splitMatrix(B, B11, B12, B21, B22, n);

  std::vector<double> P1, P2, P3, P4, P5, P6, P7;
  std::vector<double> temp1(half * half), temp2(half * half);

  addMatrix(A11, A22, temp1, half);
  addMatrix(B11, B22, temp2, half);
  P1 = strassenSequential(temp1, temp2, half);

  addMatrix(A21, A22, temp1, half);
  P2 = strassenSequential(temp1, B11, half);

  subtractMatrix(B12, B22, temp1, half);
  P3 = strassenSequential(A11, temp1, half);

  subtractMatrix(B21, B11, temp1, half);
  P4 = strassenSequential(A22, temp1, half);

  addMatrix(A11, A12, temp1, half);
  P5 = strassenSequential(temp1, B22, half);

  subtractMatrix(A21, A11, temp1, half);
  addMatrix(B11, B12, temp2, half);
  P6 = strassenSequential(temp1, temp2, half);

  subtractMatrix(A12, A22, temp1, half);
  addMatrix(B21, B22, temp2, half);
  P7 = strassenSequential(temp1, temp2, half);

  std::vector<double> C11(half * half), C12(half * half), C21(half * half), C22(half * half);

  addMatrix(P1, P4, C11, half);
  subtractMatrix(C11, P5, C11, half);
  addMatrix(C11, P7, C11, half);

  addMatrix(P3, P5, C12, half);

  addMatrix(P2, P4, C21, half);

  addMatrix(P1, P3, C22, half);
  subtractMatrix(C22, P2, C22, half);
  addMatrix(C22, P6, C22, half);

  std::vector<double> C(n * n);
  joinMatrix(C, C11, C12, C21, C22, n);

  return C;
}

std::tuple<int, int> IlinAStrassenAlgorithmMPI::calculateMatrixRange(int total_matrices) const {
  int matrices_per_process = total_matrices / world_size_;
  int extra_matrices = total_matrices % world_size_;

  int matrices_to_compute = matrices_per_process;
  if (world_rank_ < extra_matrices) {
    matrices_to_compute++;
  }

  int start_matrix = 0;
  for (int i = 0; i < world_rank_; i++) {
    int matrices_for_i = matrices_per_process + (i < extra_matrices ? 1 : 0);
    start_matrix += matrices_for_i;
  }

  int end_matrix = start_matrix + matrices_to_compute;
  return std::make_tuple(start_matrix, end_matrix);
}

void IlinAStrassenAlgorithmMPI::computeSingleProduct(int product_idx, const std::vector<double> &A11,
                                                     const std::vector<double> &A12, const std::vector<double> &A21,
                                                     const std::vector<double> &A22, const std::vector<double> &B11,
                                                     const std::vector<double> &B12, const std::vector<double> &B21,
                                                     const std::vector<double> &B22, int half,
                                                     std::vector<double> &result) {
  std::vector<double> temp1(half * half), temp2(half * half);

  switch (product_idx) {
    case 0:
      addMatrix(A11, A22, temp1, half);
      addMatrix(B11, B22, temp2, half);
      result = parallelStrassenRecursive(temp1, temp2, half);
      break;
    case 1:
      addMatrix(A21, A22, temp1, half);
      result = parallelStrassenRecursive(temp1, B11, half);
      break;
    case 2:
      subtractMatrix(B12, B22, temp1, half);
      result = parallelStrassenRecursive(A11, temp1, half);
      break;
    case 3:
      subtractMatrix(B21, B11, temp1, half);
      result = parallelStrassenRecursive(A22, temp1, half);
      break;
    case 4:
      addMatrix(A11, A12, temp1, half);
      result = parallelStrassenRecursive(temp1, B22, half);
      break;
    case 5:
      subtractMatrix(A21, A11, temp1, half);
      addMatrix(B11, B12, temp2, half);
      result = parallelStrassenRecursive(temp1, temp2, half);
      break;
    case 6:
      subtractMatrix(A12, A22, temp1, half);
      addMatrix(B21, B22, temp2, half);
      result = parallelStrassenRecursive(temp1, temp2, half);
      break;
  }
}

void IlinAStrassenAlgorithmMPI::gatherProductMatrix(const std::vector<double> &local_product,
                                                    std::vector<double> &buffer) {
  int matrix_size = static_cast<int>(local_product.size());
  buffer.resize(matrix_size * world_size_);
  MPI_Allgather(local_product.data(), matrix_size, MPI_DOUBLE, buffer.data(), matrix_size, MPI_DOUBLE, MPI_COMM_WORLD);
}

void IlinAStrassenAlgorithmMPI::mergeProductFromBuffer(const std::vector<double> &buffer, int proc_count,
                                                       int matrix_size, std::vector<double> &product) {
  for (int proc = 0; proc < proc_count; proc++) {
    int offset = proc * matrix_size;
    bool has_data = false;

    for (int i = 0; i < matrix_size; i++) {
      if (buffer[offset + i] != 0.0) {
        has_data = true;
        break;
      }
    }

    if (has_data && product[0] == 0.0) {
      std::copy(buffer.begin() + offset, buffer.begin() + offset + matrix_size, product.begin());
      break;
    }
  }
}

void IlinAStrassenAlgorithmMPI::computeResultFromProducts(const std::vector<double> &P1, const std::vector<double> &P2,
                                                          const std::vector<double> &P3, const std::vector<double> &P4,
                                                          const std::vector<double> &P5, const std::vector<double> &P6,
                                                          const std::vector<double> &P7, int half,
                                                          std::vector<double> &C11, std::vector<double> &C12,
                                                          std::vector<double> &C21, std::vector<double> &C22) {
  addMatrix(P1, P4, C11, half);
  subtractMatrix(C11, P5, C11, half);
  addMatrix(C11, P7, C11, half);

  addMatrix(P3, P5, C12, half);

  addMatrix(P2, P4, C21, half);

  addMatrix(P1, P3, C22, half);
  subtractMatrix(C22, P2, C22, half);
  addMatrix(C22, P6, C22, half);
}

std::vector<double> IlinAStrassenAlgorithmMPI::parallelStrassenRecursive(const std::vector<double> &A,
                                                                         const std::vector<double> &B, int n) {
  if (n <= threshold_) {
    return strassenSequential(A, B, n);
  }

  int half = n / 2;

  std::vector<double> A11(half * half), A12(half * half), A21(half * half), A22(half * half);
  std::vector<double> B11(half * half), B12(half * half), B21(half * half), B22(half * half);

  splitMatrix(A, A11, A12, A21, A22, n);
  splitMatrix(B, B11, B12, B21, B22, n);

  std::vector<double> P1, P2, P3, P4, P5, P6, P7;

  auto [start_matrix, end_matrix] = calculateMatrixRange(7);

  for (int matrix_idx = start_matrix; matrix_idx < end_matrix; matrix_idx++) {
    switch (matrix_idx) {
      case 0:
        computeSingleProduct(0, A11, A12, A21, A22, B11, B12, B21, B22, half, P1);
        break;
      case 1:
        computeSingleProduct(1, A11, A12, A21, A22, B11, B12, B21, B22, half, P2);
        break;
      case 2:
        computeSingleProduct(2, A11, A12, A21, A22, B11, B12, B21, B22, half, P3);
        break;
      case 3:
        computeSingleProduct(3, A11, A12, A21, A22, B11, B12, B21, B22, half, P4);
        break;
      case 4:
        computeSingleProduct(4, A11, A12, A21, A22, B11, B12, B21, B22, half, P5);
        break;
      case 5:
        computeSingleProduct(5, A11, A12, A21, A22, B11, B12, B21, B22, half, P6);
        break;
      case 6:
        computeSingleProduct(6, A11, A12, A21, A22, B11, B12, B21, B22, half, P7);
        break;
    }
  }

  int matrix_size = half * half;

  if (P1.empty()) {
    P1.resize(matrix_size, 0.0);
  }
  if (P2.empty()) {
    P2.resize(matrix_size, 0.0);
  }
  if (P3.empty()) {
    P3.resize(matrix_size, 0.0);
  }
  if (P4.empty()) {
    P4.resize(matrix_size, 0.0);
  }
  if (P5.empty()) {
    P5.resize(matrix_size, 0.0);
  }
  if (P6.empty()) {
    P6.resize(matrix_size, 0.0);
  }
  if (P7.empty()) {
    P7.resize(matrix_size, 0.0);
  }

  std::vector<double> P1_buffer, P2_buffer, P3_buffer, P4_buffer, P5_buffer, P6_buffer, P7_buffer;

  gatherProductMatrix(P1, P1_buffer);
  gatherProductMatrix(P2, P2_buffer);
  gatherProductMatrix(P3, P3_buffer);
  gatherProductMatrix(P4, P4_buffer);
  gatherProductMatrix(P5, P5_buffer);
  gatherProductMatrix(P6, P6_buffer);
  gatherProductMatrix(P7, P7_buffer);

  mergeProductFromBuffer(P1_buffer, world_size_, matrix_size, P1);
  mergeProductFromBuffer(P2_buffer, world_size_, matrix_size, P2);
  mergeProductFromBuffer(P3_buffer, world_size_, matrix_size, P3);
  mergeProductFromBuffer(P4_buffer, world_size_, matrix_size, P4);
  mergeProductFromBuffer(P5_buffer, world_size_, matrix_size, P5);
  mergeProductFromBuffer(P6_buffer, world_size_, matrix_size, P6);
  mergeProductFromBuffer(P7_buffer, world_size_, matrix_size, P7);

  std::vector<double> C11(half * half), C12(half * half), C21(half * half), C22(half * half);
  computeResultFromProducts(P1, P2, P3, P4, P5, P6, P7, half, C11, C12, C21, C22);

  std::vector<double> C(n * n);
  joinMatrix(C, C11, C12, C21, C22, n);

  return C;
}

std::tuple<int, int, int> IlinAStrassenAlgorithmMPI::calculateRowDistribution(int n) const {
  int rows_per_proc = n / world_size_;
  int remainder = n % world_size_;
  int local_rows = rows_per_proc + (world_rank_ < remainder ? 1 : 0);
  return std::make_tuple(rows_per_proc, remainder, local_rows);
}

std::vector<double> IlinAStrassenAlgorithmMPI::computeLocalRows(const std::vector<double> &A,
                                                                const std::vector<double> &B, int n, int start_row,
                                                                int local_rows) {
  std::vector<double> local_result(local_rows * n, 0.0);

  for (int i = 0; i < local_rows; ++i) {
    int global_i = start_row + i;
    for (int j = 0; j < n; ++j) {
      double sum = 0.0;
      for (int k = 0; k < n; ++k) {
        sum += A[global_i * n + k] * B[k * n + j];
      }
      local_result[i * n + j] = sum;
    }
  }

  return local_result;
}

void IlinAStrassenAlgorithmMPI::setupGatherParameters(int n, std::vector<int> &recvcounts,
                                                      std::vector<int> &displs) const {
  if (world_rank_ != 0) {
    return;
  }

  auto [rows_per_proc, remainder, _] = calculateRowDistribution(n);
  recvcounts.resize(world_size_);
  displs.resize(world_size_);

  int offset = 0;
  for (int i = 0; i < world_size_; ++i) {
    int rows_for_i = rows_per_proc + (i < remainder ? 1 : 0);
    recvcounts[i] = rows_for_i * n;
    displs[i] = offset;
    offset += recvcounts[i];
  }
}

std::vector<double> IlinAStrassenAlgorithmMPI::gatherLocalResults(const std::vector<double> &local_result, int n,
                                                                  const std::vector<int> &recvcounts,
                                                                  const std::vector<int> &displs) {
  std::vector<double> final_result;

  if (world_rank_ == 0) {
    final_result.resize(n * n);
  }

  int send_count = static_cast<int>(local_result.size());
  const double *send_data = local_result.empty() ? nullptr : local_result.data();
  double *recv_data = world_rank_ == 0 ? final_result.data() : nullptr;
  const int *recvcounts_ptr = world_rank_ == 0 ? recvcounts.data() : nullptr;
  const int *displs_ptr = world_rank_ == 0 ? displs.data() : nullptr;

  MPI_Gatherv(send_data, send_count, MPI_DOUBLE, recv_data, recvcounts_ptr, displs_ptr, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  return final_result;
}

std::vector<double> IlinAStrassenAlgorithmMPI::reorderGatheredResults(const std::vector<double> &gathered_data, int n,
                                                                      const std::vector<int> &recvcounts,
                                                                      const std::vector<int> &displs) {
  if (world_rank_ != 0) {
    return std::vector<double>();
  }

  (void)recvcounts;
  (void)displs;

  auto [rows_per_proc, remainder, _] = calculateRowDistribution(n);
  std::vector<double> reordered_result(n * n);

  int offset = 0;
  for (int proc = 0; proc < world_size_; ++proc) {
    int rows_for_proc = rows_per_proc + (proc < remainder ? 1 : 0);
    int proc_start_row = 0;

    for (int i = 0; i < proc; ++i) {
      proc_start_row += rows_per_proc + (i < remainder ? 1 : 0);
    }

    for (int i = 0; i < rows_for_proc; ++i) {
      int global_row = proc_start_row + i;
      for (int j = 0; j < n; ++j) {
        reordered_result[global_row * n + j] = gathered_data[offset + i * n + j];
      }
    }
    offset += rows_for_proc * n;
  }

  return reordered_result;
}

std::vector<double> IlinAStrassenAlgorithmMPI::distributedNaiveMultiply(const std::vector<double> &A,
                                                                        const std::vector<double> &B, int n) {
  auto [rows_per_proc, remainder, local_rows] = calculateRowDistribution(n);

  int start_row = 0;
  for (int i = 0; i < world_rank_; ++i) {
    int rows_for_i = rows_per_proc + (i < remainder ? 1 : 0);
    start_row += rows_for_i;
  }

  std::vector<double> local_result;
  if (local_rows > 0) {
    local_result = computeLocalRows(A, B, n, start_row, local_rows);
  }

  std::vector<int> recvcounts, displs;
  setupGatherParameters(n, recvcounts, displs);

  std::vector<double> gathered = gatherLocalResults(local_result, n, recvcounts, displs);

  if (world_rank_ == 0) {
    return reorderGatheredResults(gathered, n, recvcounts, displs);
  }

  return std::vector<double>();
}

std::vector<double> IlinAStrassenAlgorithmMPI::parallelStrassen(const std::vector<double> &A,
                                                                const std::vector<double> &B, int n) {
  return parallelStrassenRecursive(A, B, n);
}

std::vector<double> IlinAStrassenAlgorithmMPI::multiplyMatrices(const std::vector<double> &A,
                                                                const std::vector<double> &B, int n) {
  if (n <= threshold_) {
    return distributedNaiveMultiply(A, B, n);
  } else {
    return parallelStrassen(A, B, n);
  }
}

void IlinAStrassenAlgorithmMPI::prepareSmallMatricesCase(std::vector<double> &A_full, std::vector<double> &B_full,
                                                         std::vector<double> &final_result) {
  if (world_rank_ != 0) {
    A_full.resize(original_size_ * original_size_);
    B_full.resize(original_size_ * original_size_);
  }

  MPI_Bcast(A_full.data(), original_size_ * original_size_, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(B_full.data(), original_size_ * original_size_, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  final_result = distributedNaiveMultiply(A_full, B_full, original_size_);
}

void IlinAStrassenAlgorithmMPI::prepareLargeMatricesCase(std::vector<double> &A_full, std::vector<double> &B_full,
                                                         std::vector<double> &final_result) {
  std::vector<double> A_padded(padded_size_ * padded_size_, 0.0);
  std::vector<double> B_padded(padded_size_ * padded_size_, 0.0);

  if (world_rank_ == 0) {
    for (int i = 0; i < original_size_; ++i) {
      for (int j = 0; j < original_size_; ++j) {
        A_padded[i * padded_size_ + j] = A_full[i * original_size_ + j];
        B_padded[i * padded_size_ + j] = B_full[i * original_size_ + j];
      }
    }
  }

  if (world_rank_ != 0) {
    A_padded.resize(padded_size_ * padded_size_);
    B_padded.resize(padded_size_ * padded_size_);
  }

  MPI_Bcast(A_padded.data(), padded_size_ * padded_size_, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(B_padded.data(), padded_size_ * padded_size_, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  auto C_padded = multiplyMatrices(A_padded, B_padded, padded_size_);

  if (world_rank_ == 0 && !C_padded.empty()) {
    final_result.resize(original_size_ * original_size_);
    for (int i = 0; i < original_size_; ++i) {
      for (int j = 0; j < original_size_; ++j) {
        final_result[i * original_size_ + j] = C_padded[i * padded_size_ + j];
      }
    }
  }
}

void IlinAStrassenAlgorithmMPI::distributeFinalResult(std::vector<double> &final_result) {
  auto &output = GetOutput();

  if (world_rank_ == 0) {
    if (!final_result.empty()) {
      output.C = final_result;
      output.size = original_size_;
      MPI_Bcast(final_result.data(), original_size_ * original_size_, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
  } else {
    final_result.resize(original_size_ * original_size_);
    MPI_Bcast(final_result.data(), original_size_ * original_size_, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    output.C = final_result;
    output.size = original_size_;
  }
}

bool IlinAStrassenAlgorithmMPI::RunImpl() {
  std::vector<double> A_full, B_full;

  if (world_rank_ == 0) {
    auto &input = GetInput();
    A_full = input.A;
    B_full = input.B;
  }

  std::vector<double> final_result;

  if (original_size_ <= threshold_) {
    prepareSmallMatricesCase(A_full, B_full, final_result);
  } else {
    prepareLargeMatricesCase(A_full, B_full, final_result);
  }

  distributeFinalResult(final_result);

  return true;
}

bool IlinAStrassenAlgorithmMPI::PostProcessingImpl() {
  return true;
}

}  // namespace ilin_a_strassen_algorithm
