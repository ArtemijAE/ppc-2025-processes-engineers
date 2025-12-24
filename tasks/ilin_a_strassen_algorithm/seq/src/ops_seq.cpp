#include "ilin_a_strassen_algorithm/seq/include/ops_seq.hpp"

#include <cmath>
#include <cstddef>
#include <utility>
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

struct TaskState {
  std::vector<double> a_part;
  std::vector<double> b_part;
  std::vector<double> result;
  int size;
  int parent_idx;
  int child_count;
  int stage;
  bool completed;
};

namespace {

void ProcessStrassenTaskStage0Naive(TaskState &current, std::vector<TaskState> &tasks, std::vector<int> &ready_stack) {
  std::vector<double> result = IlinAStrassenAlgorithmSEQ::NaiveMultiply(current.a_part, current.b_part, current.size);
  current.result = std::move(result);
  current.stage = 2;
  current.completed = true;

  if (current.parent_idx != -1) {
    tasks[current.parent_idx].child_count--;
    if (tasks[current.parent_idx].child_count == 0) {
      ready_stack.push_back(current.parent_idx);
    }
  }
}

void ProcessStrassenTaskStage0Split(TaskState &current, int current_idx, std::vector<TaskState> &tasks,
                                    std::vector<int> &ready_stack) {
  current.stage = 1;
  current.child_count = 7;

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

  IlinAStrassenAlgorithmSEQ::SplitMatrix(current.a_part, a11, a12, a21, a22, current.size);
  IlinAStrassenAlgorithmSEQ::SplitMatrix(current.b_part, b11, b12, b21, b22, current.size);

  std::vector<double> temp1(half_sq);
  std::vector<double> temp2(half_sq);

  IlinAStrassenAlgorithmSEQ::AddMatrix(a11, a22, temp1, half);
  IlinAStrassenAlgorithmSEQ::AddMatrix(b11, b22, temp2, half);
  tasks.push_back({temp1, temp2, {}, half, current_idx, 0, 0, false});

  IlinAStrassenAlgorithmSEQ::AddMatrix(a21, a22, temp1, half);
  tasks.push_back({temp1, b11, {}, half, current_idx, 0, 0, false});

  IlinAStrassenAlgorithmSEQ::SubtractMatrix(b12, b22, temp1, half);
  tasks.push_back({a11, temp1, {}, half, current_idx, 0, 0, false});

  IlinAStrassenAlgorithmSEQ::SubtractMatrix(b21, b11, temp1, half);
  tasks.push_back({a22, temp1, {}, half, current_idx, 0, 0, false});

  IlinAStrassenAlgorithmSEQ::AddMatrix(a11, a12, temp1, half);
  tasks.push_back({temp1, b22, {}, half, current_idx, 0, 0, false});

  IlinAStrassenAlgorithmSEQ::SubtractMatrix(a21, a11, temp1, half);
  IlinAStrassenAlgorithmSEQ::AddMatrix(b11, b12, temp2, half);
  tasks.push_back({temp1, temp2, {}, half, current_idx, 0, 0, false});

  IlinAStrassenAlgorithmSEQ::SubtractMatrix(a12, a22, temp1, half);
  IlinAStrassenAlgorithmSEQ::AddMatrix(b21, b22, temp2, half);
  tasks.push_back({temp1, temp2, {}, half, current_idx, 0, 0, false});

  for (int i = 0; i < 7; ++i) {
    int child_idx = static_cast<int>(tasks.size()) - 7 + i;
    ready_stack.push_back(child_idx);
  }
}

void ProcessStrassenTaskStage0(TaskState &current, int current_idx, std::vector<TaskState> &tasks,
                               std::vector<int> &ready_stack) {
  if (current.size <= 64) {
    ProcessStrassenTaskStage0Naive(current, tasks, ready_stack);
  } else {
    ProcessStrassenTaskStage0Split(current, current_idx, tasks, ready_stack);
  }
}

void ProcessStrassenTaskStage1Collect(TaskState &current, int current_idx, std::vector<TaskState> &tasks,
                                      std::vector<int> &ready_stack) {
  std::vector<std::vector<double>> products(7);
  int found = 0;

  for (const auto &task : tasks) {
    if (task.parent_idx == current_idx && task.completed) {
      products[found] = task.result;
      found++;
      if (found == 7) {
        break;
      }
    }
  }

  if (found == 7) {
    int half = current.size / 2;
    std::size_t half_sq = static_cast<std::size_t>(half) * static_cast<std::size_t>(half);

    std::vector<double> c11(half_sq);
    std::vector<double> c12(half_sq);
    std::vector<double> c21(half_sq);
    std::vector<double> c22(half_sq);

    IlinAStrassenAlgorithmSEQ::ComputeResultSubmatrices(products, c11, c12, c21, c22, half);

    std::vector<double> c(static_cast<std::size_t>(current.size) * static_cast<std::size_t>(current.size));
    IlinAStrassenAlgorithmSEQ::JoinMatrix(c, c11, c12, c21, c22, current.size);

    current.result = c;
    current.stage = 2;
    current.completed = true;

    if (current.parent_idx != -1) {
      tasks[current.parent_idx].child_count--;
      if (tasks[current.parent_idx].child_count == 0) {
        ready_stack.push_back(current.parent_idx);
      }
    }
  } else {
    ready_stack.push_back(current_idx);
  }
}

void ProcessStrassenTaskStage1Wait(int current_idx, std::vector<int> &ready_stack) {
  ready_stack.push_back(current_idx);
}

void ProcessStrassenTaskStage1(TaskState &current, int current_idx, std::vector<TaskState> &tasks,
                               std::vector<int> &ready_stack) {
  if (current.child_count == 0) {
    ProcessStrassenTaskStage1Collect(current, current_idx, tasks, ready_stack);
  } else {
    ProcessStrassenTaskStage1Wait(current_idx, ready_stack);
  }
}

}  // namespace

std::vector<double> IlinAStrassenAlgorithmSEQ::StrassenMultiply(const std::vector<double> &a,
                                                                const std::vector<double> &b, int n) {
  if (n <= kThreshold) {
    return NaiveMultiply(a, b, n);
  }

  std::vector<TaskState> tasks;
  std::vector<int> ready_stack;

  tasks.push_back({a, b, {}, n, -1, 0, 0, false});
  ready_stack.push_back(0);

  int iteration = 0;
  const int max_iterations = 10000;

  while (!ready_stack.empty() && iteration++ < max_iterations) {
    int current_idx = ready_stack.back();
    ready_stack.pop_back();

    TaskState &current = tasks[current_idx];

    if (current.stage == 0) {
      ProcessStrassenTaskStage0(current, current_idx, tasks, ready_stack);
    } else if (current.stage == 1) {
      ProcessStrassenTaskStage1(current, current_idx, tasks, ready_stack);
    }
  }

  for (const auto &task : tasks) {
    if (task.completed && task.parent_idx == -1) {
      return task.result;
    }
  }

  return {};
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

    if (c_padded.empty()) {
      return false;
    }

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

}  // namespace ilin_a_strassen_algorithm
