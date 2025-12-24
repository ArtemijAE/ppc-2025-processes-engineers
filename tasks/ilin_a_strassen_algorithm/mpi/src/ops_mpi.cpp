#include "ilin_a_strassen_algorithm/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <tuple>
#include <utility>
#include <vector>

#include "ilin_a_strassen_algorithm/common/include/common.hpp"

namespace ilin_a_strassen_algorithm {

IlinAStrassenAlgorithmMPI::IlinAStrassenAlgorithmMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput().size = 0;
  GetOutput().C.clear();
}

bool IlinAStrassenAlgorithmMPI::ValidationImpl() {
  int rank = 0;
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
    if (input.A.size() != static_cast<std::size_t>(input.size) * static_cast<std::size_t>(input.size)) {
      validation_result = 0;
    }
    if (input.B.size() != static_cast<std::size_t>(input.size) * static_cast<std::size_t>(input.size)) {
      validation_result = 0;
    }

    MPI_Bcast(&validation_result, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (validation_result != 0) {
      MPI_Bcast(&size_to_bcast, 1, MPI_INT, 0, MPI_COMM_WORLD);
      return true;
    }

    int error_size = -1;
    MPI_Bcast(&error_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
    return false;
  }

  int validation_result = 0;
  MPI_Bcast(&validation_result, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (validation_result != 0) {
    int size = 0;
    MPI_Bcast(&size, 1, MPI_INT, 0, MPI_COMM_WORLD);
    return size > 0;
  }

  int error_size = 0;
  MPI_Bcast(&error_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
  return false;
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

void IlinAStrassenAlgorithmMPI::AddMatrix(const std::vector<double> &a, const std::vector<double> &b,
                                          std::vector<double> &c, int n) {
  for (int i = 0; i < n * n; ++i) {
    c[i] = a[i] + b[i];
  }
}

void IlinAStrassenAlgorithmMPI::SubtractMatrix(const std::vector<double> &a, const std::vector<double> &b,
                                               std::vector<double> &c, int n) {
  for (int i = 0; i < n * n; ++i) {
    c[i] = a[i] - b[i];
  }
}

void IlinAStrassenAlgorithmMPI::SplitMatrix(const std::vector<double> &a, std::vector<double> &a11,
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

void IlinAStrassenAlgorithmMPI::JoinMatrix(std::vector<double> &a, const std::vector<double> &a11,
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

std::vector<double> IlinAStrassenAlgorithmMPI::NaiveMultiplySeq(const std::vector<double> &a,
                                                                const std::vector<double> &b, int n) {
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

std::vector<double> IlinAStrassenAlgorithmMPI::StrassenSequential(const std::vector<double> &a,
                                                                  const std::vector<double> &b, int n) {
  if (n <= kThreshold) {
    return NaiveMultiplySeq(a, b, n);
  }

  struct MatrixTask {
    std::vector<double> a;
    std::vector<double> b;
    std::vector<double> result;
    int size;
    int stage;
  };

  std::vector<MatrixTask> stack;
  std::vector<MatrixTask> results;

  stack.push_back({a, b, {}, n, 0});

  while (!stack.empty()) {
    MatrixTask current = stack.back();
    stack.pop_back();

    if (current.stage == 0) {
      if (current.size <= kThreshold) {
        current.result = NaiveMultiplySeq(current.a, current.b, current.size);
        results.push_back(current);
        continue;
      }

      int half = current.size / 2;
      std::size_t half_sq = static_cast<std::size_t>(half) * static_cast<std::size_t>(half);

      std::vector<double> a11(half_sq);
      std::vector<double> a12(half_sq);
      std::vector<double> a21(half_sq);
      std::vector<double> a22(half_sq);
      std::vector<double> b11(half_sq);
      std::vector<double> b12(half_sq);
      std::vector<double> b21(half_sq);
      std::vector<double> b22(half_sq);

      SplitMatrix(current.a, a11, a12, a21, a22, current.size);
      SplitMatrix(current.b, b11, b12, b21, b22, current.size);

      std::vector<double> temp1(half_sq);
      std::vector<double> temp2(half_sq);

      AddMatrix(a11, a22, temp1, half);
      AddMatrix(b11, b22, temp2, half);
      stack.push_back({temp1, temp2, {}, half, 0});

      AddMatrix(a21, a22, temp1, half);
      stack.push_back({temp1, b11, {}, half, 0});

      SubtractMatrix(b12, b22, temp1, half);
      stack.push_back({a11, temp1, {}, half, 0});

      SubtractMatrix(b21, b11, temp1, half);
      stack.push_back({a22, temp1, {}, half, 0});

      AddMatrix(a11, a12, temp1, half);
      stack.push_back({temp1, b22, {}, half, 0});

      SubtractMatrix(a21, a11, temp1, half);
      AddMatrix(b11, b12, temp2, half);
      stack.push_back({temp1, temp2, {}, half, 0});

      SubtractMatrix(a12, a22, temp1, half);
      AddMatrix(b21, b22, temp2, half);
      stack.push_back({temp1, temp2, {}, half, 0});

      current.stage = 1;
      stack.push_back(current);
    } else {
      bool has_results = results.size() >= 7;

      if (has_results) {
        std::vector<std::vector<double>> products(7);
        for (int i = 0; i < 7; ++i) {
          products[6 - i] = results.back().result;
          results.pop_back();
        }

        int half = current.size / 2;
        std::size_t half_sq = static_cast<std::size_t>(half) * static_cast<std::size_t>(half);

        std::vector<double> c11(half_sq);
        std::vector<double> c12(half_sq);
        std::vector<double> c21(half_sq);
        std::vector<double> c22(half_sq);

        ComputeResultFromProducts(products[0], products[1], products[2], products[3], products[4], products[5],
                                  products[6], half, c11, c12, c21, c22);

        std::vector<double> c(static_cast<std::size_t>(current.size) * static_cast<std::size_t>(current.size));
        JoinMatrix(c, c11, c12, c21, c22, current.size);

        if (current.size == n) {
          return c;
        }
        results.push_back({current.a, current.b, c, current.size, 2});
      } else {
        stack.push_back(current);
      }
    }
  }

  if (!results.empty()) {
    return results.back().result;
  }

  return {};
}

std::tuple<int, int> IlinAStrassenAlgorithmMPI::CalculateMatrixRange(int total_matrices) const {
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

void IlinAStrassenAlgorithmMPI::ComputeSingleProduct(int product_idx, const std::vector<double> &a11,
                                                     const std::vector<double> &a12, const std::vector<double> &a21,
                                                     const std::vector<double> &a22, const std::vector<double> &b11,
                                                     const std::vector<double> &b12, const std::vector<double> &b21,
                                                     const std::vector<double> &b22, int half,
                                                     std::vector<double> &result) {
  std::size_t half_sq = static_cast<std::size_t>(half) * static_cast<std::size_t>(half);
  std::vector<double> temp1(half_sq);
  std::vector<double> temp2(half_sq);

  switch (product_idx) {
    case 0:
      AddMatrix(a11, a22, temp1, half);
      AddMatrix(b11, b22, temp2, half);
      result = StrassenSequential(temp1, temp2, half);
      break;
    case 1:
      AddMatrix(a21, a22, temp1, half);
      result = StrassenSequential(temp1, b11, half);
      break;
    case 2:
      SubtractMatrix(b12, b22, temp1, half);
      result = StrassenSequential(a11, temp1, half);
      break;
    case 3:
      SubtractMatrix(b21, b11, temp1, half);
      result = StrassenSequential(a22, temp1, half);
      break;
    case 4:
      AddMatrix(a11, a12, temp1, half);
      result = StrassenSequential(temp1, b22, half);
      break;
    case 5:
      SubtractMatrix(a21, a11, temp1, half);
      AddMatrix(b11, b12, temp2, half);
      result = StrassenSequential(temp1, temp2, half);
      break;
    case 6:
      SubtractMatrix(a12, a22, temp1, half);
      AddMatrix(b21, b22, temp2, half);
      result = StrassenSequential(temp1, temp2, half);
      break;
    default:
      result = std::vector<double>(half_sq);
      break;
  }
}

void IlinAStrassenAlgorithmMPI::GatherProductMatrix(const std::vector<double> &local_product,
                                                    std::vector<double> &buffer) const {
  int matrix_size = static_cast<int>(local_product.size());
  buffer.resize(static_cast<std::size_t>(matrix_size) * static_cast<std::size_t>(world_size_));
  MPI_Allgather(local_product.data(), matrix_size, MPI_DOUBLE, buffer.data(), matrix_size, MPI_DOUBLE, MPI_COMM_WORLD);
}

void IlinAStrassenAlgorithmMPI::MergeProductFromBuffer(const std::vector<double> &buffer, int proc_count,
                                                       int matrix_size, std::vector<double> &product) {
  for (int proc = 0; proc < proc_count; proc++) {
    int offset = proc * matrix_size;
    bool has_data = false;

    for (int i = 0; i < matrix_size; i++) {
      if (buffer[static_cast<std::size_t>(offset) + i] != 0.0) {
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

void IlinAStrassenAlgorithmMPI::ComputeResultFromProducts(const std::vector<double> &p1, const std::vector<double> &p2,
                                                          const std::vector<double> &p3, const std::vector<double> &p4,
                                                          const std::vector<double> &p5, const std::vector<double> &p6,
                                                          const std::vector<double> &p7, int half,
                                                          std::vector<double> &c11, std::vector<double> &c12,
                                                          std::vector<double> &c21, std::vector<double> &c22) {
  AddMatrix(p1, p4, c11, half);
  SubtractMatrix(c11, p5, c11, half);
  AddMatrix(c11, p7, c11, half);

  AddMatrix(p3, p5, c12, half);

  AddMatrix(p2, p4, c21, half);

  AddMatrix(p1, p3, c22, half);
  SubtractMatrix(c22, p2, c22, half);
  AddMatrix(c22, p6, c22, half);
}

std::vector<double> IlinAStrassenAlgorithmMPI::ParallelStrassenIterative(const std::vector<double> &a,
                                                                         const std::vector<double> &b, int n) {
  if (n <= kThreshold) {
    return StrassenSequential(a, b, n);
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

  std::vector<double> p1;
  std::vector<double> p2;
  std::vector<double> p3;
  std::vector<double> p4;
  std::vector<double> p5;
  std::vector<double> p6;
  std::vector<double> p7;

  auto [start_matrix, end_matrix] = CalculateMatrixRange(7);

  for (int matrix_idx = start_matrix; matrix_idx < end_matrix; matrix_idx++) {
    std::vector<double> result;
    switch (matrix_idx) {
      case 0:
        ComputeSingleProduct(0, a11, a12, a21, a22, b11, b12, b21, b22, half, result);
        p1 = std::move(result);
        break;
      case 1:
        ComputeSingleProduct(1, a11, a12, a21, a22, b11, b12, b21, b22, half, result);
        p2 = std::move(result);
        break;
      case 2:
        ComputeSingleProduct(2, a11, a12, a21, a22, b11, b12, b21, b22, half, result);
        p3 = std::move(result);
        break;
      case 3:
        ComputeSingleProduct(3, a11, a12, a21, a22, b11, b12, b21, b22, half, result);
        p4 = std::move(result);
        break;
      case 4:
        ComputeSingleProduct(4, a11, a12, a21, a22, b11, b12, b21, b22, half, result);
        p5 = std::move(result);
        break;
      case 5:
        ComputeSingleProduct(5, a11, a12, a21, a22, b11, b12, b21, b22, half, result);
        p6 = std::move(result);
        break;
      case 6:
        ComputeSingleProduct(6, a11, a12, a21, a22, b11, b12, b21, b22, half, result);
        p7 = std::move(result);
        break;
      default:
        break;
    }
  }

  int matrix_size = half * half;

  if (p1.empty()) {
    p1.resize(static_cast<std::size_t>(matrix_size), 0.0);
  }
  if (p2.empty()) {
    p2.resize(static_cast<std::size_t>(matrix_size), 0.0);
  }
  if (p3.empty()) {
    p3.resize(static_cast<std::size_t>(matrix_size), 0.0);
  }
  if (p4.empty()) {
    p4.resize(static_cast<std::size_t>(matrix_size), 0.0);
  }
  if (p5.empty()) {
    p5.resize(static_cast<std::size_t>(matrix_size), 0.0);
  }
  if (p6.empty()) {
    p6.resize(static_cast<std::size_t>(matrix_size), 0.0);
  }
  if (p7.empty()) {
    p7.resize(static_cast<std::size_t>(matrix_size), 0.0);
  }

  std::vector<double> p1_buffer;
  std::vector<double> p2_buffer;
  std::vector<double> p3_buffer;
  std::vector<double> p4_buffer;
  std::vector<double> p5_buffer;
  std::vector<double> p6_buffer;
  std::vector<double> p7_buffer;

  GatherProductMatrix(p1, p1_buffer);
  GatherProductMatrix(p2, p2_buffer);
  GatherProductMatrix(p3, p3_buffer);
  GatherProductMatrix(p4, p4_buffer);
  GatherProductMatrix(p5, p5_buffer);
  GatherProductMatrix(p6, p6_buffer);
  GatherProductMatrix(p7, p7_buffer);

  MergeProductFromBuffer(p1_buffer, world_size_, matrix_size, p1);
  MergeProductFromBuffer(p2_buffer, world_size_, matrix_size, p2);
  MergeProductFromBuffer(p3_buffer, world_size_, matrix_size, p3);
  MergeProductFromBuffer(p4_buffer, world_size_, matrix_size, p4);
  MergeProductFromBuffer(p5_buffer, world_size_, matrix_size, p5);
  MergeProductFromBuffer(p6_buffer, world_size_, matrix_size, p6);
  MergeProductFromBuffer(p7_buffer, world_size_, matrix_size, p7);

  std::vector<double> c11(half_sq);
  std::vector<double> c12(half_sq);
  std::vector<double> c21(half_sq);
  std::vector<double> c22(half_sq);
  ComputeResultFromProducts(p1, p2, p3, p4, p5, p6, p7, half, c11, c12, c21, c22);

  std::vector<double> c(static_cast<std::size_t>(n) * static_cast<std::size_t>(n));
  JoinMatrix(c, c11, c12, c21, c22, n);

  return c;
}

std::tuple<int, int, int> IlinAStrassenAlgorithmMPI::CalculateRowDistribution(int n) const {
  int rows_per_proc = n / world_size_;
  int remainder = n % world_size_;
  int local_rows = rows_per_proc + (world_rank_ < remainder ? 1 : 0);
  return std::make_tuple(rows_per_proc, remainder, local_rows);
}

std::vector<double> IlinAStrassenAlgorithmMPI::ComputeLocalRows(const std::vector<double> &a,
                                                                const std::vector<double> &b, int n, int start_row,
                                                                int local_rows) {
  std::vector<double> local_result(static_cast<std::size_t>(local_rows) * static_cast<std::size_t>(n), 0.0);

  for (int i = 0; i < local_rows; ++i) {
    int global_i = start_row + i;
    for (int j = 0; j < n; ++j) {
      double sum = 0.0;
      for (int k = 0; k < n; ++k) {
        sum += a[(global_i * n) + k] * b[(k * n) + j];
      }
      local_result[(i * n) + j] = sum;
    }
  }

  return local_result;
}

void IlinAStrassenAlgorithmMPI::SetupGatherParameters(int n, std::vector<int> &recvcounts,
                                                      std::vector<int> &displs) const {
  if (world_rank_ != 0) {
    return;
  }

  auto [rows_per_proc, remainder, _] = CalculateRowDistribution(n);
  recvcounts.resize(static_cast<std::size_t>(world_size_));
  displs.resize(static_cast<std::size_t>(world_size_));

  int offset = 0;
  for (int i = 0; i < world_size_; ++i) {
    int rows_for_i = rows_per_proc + (i < remainder ? 1 : 0);
    recvcounts[static_cast<std::size_t>(i)] = rows_for_i * n;
    displs[static_cast<std::size_t>(i)] = offset;
    offset += recvcounts[static_cast<std::size_t>(i)];
  }
}

std::vector<double> IlinAStrassenAlgorithmMPI::GatherLocalResults(const std::vector<double> &local_result, int n,
                                                                  const std::vector<int> &recvcounts,
                                                                  const std::vector<int> &displs) const {
  std::vector<double> final_result;

  if (world_rank_ == 0) {
    final_result.resize(static_cast<std::size_t>(n) * static_cast<std::size_t>(n));
  }

  int send_count = static_cast<int>(local_result.size());
  const double *send_data = local_result.empty() ? nullptr : local_result.data();
  double *recv_data = world_rank_ == 0 ? final_result.data() : nullptr;
  const int *recvcounts_ptr = world_rank_ == 0 ? recvcounts.data() : nullptr;
  const int *displs_ptr = world_rank_ == 0 ? displs.data() : nullptr;

  MPI_Gatherv(send_data, send_count, MPI_DOUBLE, recv_data, recvcounts_ptr, displs_ptr, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  return final_result;
}

std::vector<double> IlinAStrassenAlgorithmMPI::ReorderGatheredResults(const std::vector<double> &gathered_data,
                                                                      int n) const {
  if (world_rank_ != 0) {
    return {};
  }

  auto [rows_per_proc, remainder, _] = CalculateRowDistribution(n);
  std::vector<double> reordered_result(static_cast<std::size_t>(n) * static_cast<std::size_t>(n));

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
        reordered_result[(global_row * n) + j] = gathered_data[offset + (i * n) + j];
      }
    }
    offset += rows_for_proc * n;
  }

  return reordered_result;
}

std::vector<double> IlinAStrassenAlgorithmMPI::DistributedNaiveMultiply(const std::vector<double> &a,
                                                                        const std::vector<double> &b, int n) {
  auto [rows_per_proc, remainder, local_rows] = CalculateRowDistribution(n);

  int start_row = 0;
  for (int i = 0; i < world_rank_; ++i) {
    int rows_for_i = rows_per_proc + (i < remainder ? 1 : 0);
    start_row += rows_for_i;
  }

  std::vector<double> local_result;
  if (local_rows > 0) {
    local_result = ComputeLocalRows(a, b, n, start_row, local_rows);
  }

  std::vector<int> recvcounts;
  std::vector<int> displs;
  SetupGatherParameters(n, recvcounts, displs);

  std::vector<double> gathered = GatherLocalResults(local_result, n, recvcounts, displs);

  if (world_rank_ == 0) {
    return ReorderGatheredResults(gathered, n);
  }

  return {};
}

std::vector<double> IlinAStrassenAlgorithmMPI::ParallelStrassen(const std::vector<double> &a,
                                                                const std::vector<double> &b, int n) {
  return ParallelStrassenIterative(a, b, n);
}

std::vector<double> IlinAStrassenAlgorithmMPI::MultiplyMatrices(const std::vector<double> &a,
                                                                const std::vector<double> &b, int n) {
  if (n <= kThreshold) {
    return DistributedNaiveMultiply(a, b, n);
  }
  return ParallelStrassen(a, b, n);
}

void IlinAStrassenAlgorithmMPI::PrepareSmallMatricesCase(std::vector<double> &a_full, std::vector<double> &b_full,
                                                         std::vector<double> &final_result) {
  if (world_rank_ != 0) {
    a_full.resize(static_cast<std::size_t>(original_size_) * static_cast<std::size_t>(original_size_));
    b_full.resize(static_cast<std::size_t>(original_size_) * static_cast<std::size_t>(original_size_));
  }

  MPI_Bcast(a_full.data(), original_size_ * original_size_, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(b_full.data(), original_size_ * original_size_, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  final_result = DistributedNaiveMultiply(a_full, b_full, original_size_);
}

void IlinAStrassenAlgorithmMPI::PrepareLargeMatricesCase(std::vector<double> &a_full, std::vector<double> &b_full,
                                                         std::vector<double> &final_result) {
  std::size_t padded_size_sq = static_cast<std::size_t>(padded_size_) * static_cast<std::size_t>(padded_size_);
  std::vector<double> a_padded(padded_size_sq, 0.0);
  std::vector<double> b_padded(padded_size_sq, 0.0);

  if (world_rank_ == 0) {
    for (int i = 0; i < original_size_; ++i) {
      for (int j = 0; j < original_size_; ++j) {
        a_padded[(i * padded_size_) + j] = a_full[(i * original_size_) + j];
        b_padded[(i * padded_size_) + j] = b_full[(i * original_size_) + j];
      }
    }
  }

  if (world_rank_ != 0) {
    a_padded.resize(padded_size_sq);
    b_padded.resize(padded_size_sq);
  }

  MPI_Bcast(a_padded.data(), padded_size_ * padded_size_, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(b_padded.data(), padded_size_ * padded_size_, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  auto c_padded = MultiplyMatrices(a_padded, b_padded, padded_size_);

  if (world_rank_ == 0 && !c_padded.empty()) {
    final_result.resize(static_cast<std::size_t>(original_size_) * static_cast<std::size_t>(original_size_));
    for (int i = 0; i < original_size_; ++i) {
      for (int j = 0; j < original_size_; ++j) {
        final_result[(i * original_size_) + j] = c_padded[(i * padded_size_) + j];
      }
    }
  }
}

void IlinAStrassenAlgorithmMPI::DistributeFinalResult(std::vector<double> &final_result) {
  auto &output = GetOutput();

  if (world_rank_ == 0) {
    if (!final_result.empty()) {
      output.C = final_result;
      output.size = original_size_;
      MPI_Bcast(final_result.data(), original_size_ * original_size_, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
  } else {
    final_result.resize(static_cast<std::size_t>(original_size_) * static_cast<std::size_t>(original_size_));
    MPI_Bcast(final_result.data(), original_size_ * original_size_, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    output.C = final_result;
    output.size = original_size_;
  }
}

bool IlinAStrassenAlgorithmMPI::RunImpl() {
  std::vector<double> a_full;
  std::vector<double> b_full;

  if (world_rank_ == 0) {
    auto &input = GetInput();
    a_full = input.A;
    b_full = input.B;
  }

  std::vector<double> final_result;

  if (original_size_ <= kThreshold) {
    PrepareSmallMatricesCase(a_full, b_full, final_result);
  } else {
    PrepareLargeMatricesCase(a_full, b_full, final_result);
  }

  DistributeFinalResult(final_result);

  return true;
}

bool IlinAStrassenAlgorithmMPI::PostProcessingImpl() {
  return true;
}

}  // namespace ilin_a_strassen_algorithm
