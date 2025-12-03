# Нахождение числа чередований знаков значений соседних элементов вектора

- Студент: Ильин Артемий Александрович, группа 3823Б1ПР1
- Технология: SEQ, MPI
- Вариант: 5

## 1. Введение
Задача подсчета чередований знаков в векторах может явиться важной задачей анализа данных. Эта задача является типичным показательным примером улучшения производительности вычислений при использовании методов параллельного программирования.
Целью данной работы является реализация и сравнение последовательного и параллельного алгоритма с использованием MPI для демонстрации эффективности распараллеливания задачи нахождения числа чередований знаков значений соседних элементов вектора.

## 2. Постановка задачи
**Описание задачи:** Для заданного вектора целых чисел найти количество пар соседних элементов, у которых знаки различны.

**Входные данные:** `vector<int>` - исходный вектор  
**Выходные данные:** `int` - количество чередований знаков

**Ограничения:**
- вектор может содержать положительные, отрицательные числа и нули;
- результирующее число чередований знаков вектора последовательной и параллельной реализаций алгоритма не должны различаться;
- для реализации параллельного алгоритма должен быть использован MPI;
- ноль считается неотрицательным числом;
- для вектора размером < 2 результат равен 0;

## 3. Описание базового алгоритма

```cpp
int alternation_count = 0;
for (size_t i = 0; i < vec.size() - 1; ++i) {
    if ((vec[i] < 0 && vec[i + 1] >= 0) || 
        (vec[i] >= 0 && vec[i + 1] < 0)) {
        alternation_count++;
    }
}
```

**Шаги алгоритма:**

1. **Инициализация счетчика** - устанавливается начальное значение `alternation_count = 0`
2. **Последовательное прохождение по элементам массива** - цикл от первого до предпоследнего элемента:
   - На каждой итерации цикла в проверке участвует текущий элемент `vec[i]` и следующий (соседний) `vec[i + 1]`
3. **Проверка условия чередования знаков** 
   - Отрицательный -> Неотрицательный (`vec[i] < 0 && vec[i + 1] >= 0`)
   - Неотрицательный -> Отрицательный (`vec[i] >= 0 && vec[i + 1] < 0`)
4. **Увеличение счетчика** - при выполнении любого из условий чередования значение `alternation_count` увеличивается на 1
5. **Завершение обработки** - после прохода всех пар соседних элементов в счетчике содержится общее количество чередований знаков между соседними элементами вектора

**Сложность:** O(n), где n - размер вектора

## 4. Схема распараллеливания

**Описание принципа распределения данных**

Использовано блочное распределение вектора между процессами. Вектор делится на непрерывные блоки почти равного размера (размер отличается на 1). Для вектора размером `n` и `p` процессов:
- Первые `n % p` процессов получают блоки размером `⌊n/p⌋ + 1`
- Остальные процессы получают блоки размером `⌊n/p⌋`

**Коммуникация между процессами**

**Распределение размера вектора**
```cpp
MPI_Bcast(&data_size, 1, MPI_INT, kRootRank, MPI_COMM_WORLD);
```
Все процессы должны знать общий размер вектора для вычисления размеров своих блоков.

**Распределение данных**
В реализации используется схема с ручным распределением данных:
```cpp
void DistributeData(const std::vector<int> &global_data,
                   std::vector<int> &local_data, const int world_rank,
                   const int world_size) {
  if (world_rank == kRootRank) {
    for (int process_index = 0; process_index < world_size; ++process_index) {
      if (process_index == kRootRank) continue;
      MPI_Send(send_data, send_size, MPI_INT, process_index, 0, MPI_COMM_WORLD);
    }
  } else {
    MPI_Recv(local_data.data(), local_data.size(), MPI_INT, kRootRank, 0, 
             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }
}
```

**Локальный подсчет чередований**
```cpp
int CountLocalSignChanges(const std::vector<int> &segment) {
  int count = 0;
  for (size_t index = 0; index < segment.size() - 1; ++index) {
    const bool is_negative_current = segment[index] < 0;
    const bool is_negative_next = segment[index + 1] < 0;
    if (is_negative_current != is_negative_next) {
      ++count;
    }
  }
  return count;
}
```
Каждый процесс независимо подсчитывает чередования в своем блоке.

**Сбор граничных элементов**
```cpp
BoundaryInfo GatherEdgeValues(const std::vector<int> &segment) {
  const int left_val = segment.empty() ? 0 : segment.front();
  const int right_val = segment.empty() ? 0 : segment.back();
  MPI_Gather(&left_val, 1, MPI_INT, info.all_edges.data(), 1, MPI_INT, 
             kRootRank, MPI_COMM_WORLD);
  MPI_Gather(&right_val, 1, MPI_INT, 
             info.all_edges.data() + total_processes, 1, MPI_INT,
             kRootRank, MPI_COMM_WORLD);
  return info;
}
```

**Суммирование результатов**
```cpp
MPI_Reduce(&local_changes, &total_changes, 1, MPI_INT, MPI_SUM, kRootRank, 
           MPI_COMM_WORLD);
```
Локальные подсчеты суммируются на процессе 0.

**Добавление граничных чередований**
```cpp
int CountEdgeAlternations(const BoundaryInfo &edges, const int total_processes) {
  int count = 0;
  for (int process_index = 0; process_index < total_processes - 1; ++process_index) {
    const int right_edge = edges.all_edges[total_processes + process_index];
    const int left_edge = edges.all_edges[process_index + 1];
    const bool is_negative_right = right_edge < 0;
    const bool is_negative_left = left_edge < 0;
    if (is_negative_right != is_negative_left) {
      ++count;
    }
  }
  return count;
}
```

**Полный код параллельного алгоритма (RunImpl):**
```cpp
bool IlinAAlternationsSignsOfValVecMPI::RunImpl() {
  int world_rank = 0, world_size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  const std::vector<int> &input_data = GetInput();
  int data_size = static_cast<int>(input_data.size());
  MPI_Bcast(&data_size, 1, MPI_INT, kRootRank, MPI_COMM_WORLD);
  if (data_size < kMinDataSize) {
    if (world_rank == kRootRank) GetOutput() = 0;
    return true;
  }
  std::vector<int> counts(world_size), offsets(world_size);
  if (world_rank == kRootRank) {
    CalculateDistribution(data_size, world_size, counts, offsets);
  }
  MPI_Bcast(counts.data(), world_size, MPI_INT, kRootRank, MPI_COMM_WORLD);
  MPI_Bcast(offsets.data(), world_size, MPI_INT, kRootRank, MPI_COMM_WORLD);
  const int local_size = counts[world_rank];
  std::vector<int> local_data(static_cast<size_t>(local_size));
  DistributeData(input_data, local_data, world_rank, world_size);
  const int local_changes = CountLocalSignChanges(local_data);
  BoundaryInfo edges = GatherEdgeValues(local_data);
  int total_changes = 0;
  MPI_Reduce(&local_changes, &total_changes, 1, MPI_INT, MPI_SUM, kRootRank, 
             MPI_COMM_WORLD);
  if (world_rank == kRootRank) {
    total_changes += CountEdgeAlternations(edges, world_size);
    GetOutput() = total_changes;
  }
  return true;
}
```

## 5. Детали реализации

**Структура проекта**
```
ilin_a_alternations_signs_of_val_vec/
├── common/
│   └── include/
│       └── common.hpp
├── mpi/
│   ├── include/
│   │   └── ops_mpi.hpp
│   └── src/
│       └── ops_mpi.cpp
├── seq/
│   ├── include/
│   │   └── ops_seq.hpp
│   └── src/
│       └── ops_seq.cpp
├── tests/
│   ├── functional/
│   │   └── main.cpp
│   └── performance/
│       └── main.cpp
├── info.json
├── report.md
└── settings.json
```

**Общие компоненты (common/include/common.hpp)**

Определение типов данных и пространства имен:
```cpp
namespace ilin_a_alternations_signs_of_val_vec {
    using InType = std::vector<int>;        
    using OutType = int;                    
    using TestType = std::tuple<int, std::string>;  
    using BaseTask = ppc::task::Task<InType, OutType>;
}
```

**Последовательная реализация (seq/)**
**ops_seq.hpp** - объявление класса:
```cpp
class IlinAAlternationsSignsOfValVecSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit IlinAAlternationsSignsOfValVecSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};
```

**ops_seq.cpp** - реализация методов:

- **Конструктор** - инициализация задачи:
```cpp
IlinAAlternationsSignsOfValVecSEQ::IlinAAlternationsSignsOfValVecSEQ(const InType &in) {
    SetTypeOfTask(GetStaticTypeOfTask());
    GetInput() = in;
    GetOutput() = 0;
}
```

- **ValidationImpl()** - проверка корректности входных данных:
```cpp
bool IlinAAlternationsSignsOfValVecSEQ::ValidationImpl() {
    return !GetInput().empty() && (GetOutput() == 0);
}
```

- **RunImpl()** - основной алгоритм:
```cpp
bool IlinAAlternationsSignsOfValVecSEQ::RunImpl() {
    const std::vector<int>& vec = GetInput();
    int alternation_count = 0;
    if (vec.size() < 2) {
        GetOutput() = 0;
        return true;
    }
    for (size_t i = 0; i < vec.size() - 1; ++i) {
        if ((vec[i] < 0 && vec[i + 1] >= 0) || 
            (vec[i] >= 0 && vec[i + 1] < 0)) {
            alternation_count++;
        }
    }
    GetOutput() = alternation_count;
    return true;
}
```

**MPI реализация (mpi/)**

**ops_mpi.hpp** - объявление класса со вспомогательными методами:
```cpp
class IlinAAlternationsSignsOfValVecMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit IlinAAlternationsSignsOfValVecMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  static int CountLocalSignChanges(const std::vector<int> &segment);
  static BoundaryInfo GatherEdgeValues(const std::vector<int> &segment);
  static int CountEdgeAlternations(const BoundaryInfo &edges, int total_processes);
  static void CalculateDistribution(int data_size, int world_size, 
                                   std::vector<int> &counts, std::vector<int> &offsets);
  static void DistributeData(const std::vector<int> &global_data, 
                            std::vector<int> &local_data, int world_rank,
                            int world_size);
};
```

**ops_mpi.cpp** - содержит полную реализацию параллельного алгоритма, описанного в разделе 4.

**Функциональные тесты (tests/functional/main.cpp)**
Генерация тестовых данных различных типов (16 тестовых случаев):
```cpp
const std::array<TestType, 16> kTestParam = {
    std::make_tuple(10, "alternating"),   std::make_tuple(100, "alternating"),
    std::make_tuple(1000, "alternating"), std::make_tuple(10, "all_positive"),
    std::make_tuple(100, "all_positive"), std::make_tuple(10, "all_negative"),
    std::make_tuple(100, "all_negative"), std::make_tuple(50, "random"),
    std::make_tuple(500, "random"),       std::make_tuple(10, "zeros"),
    std::make_tuple(1, "all_positive"),   std::make_tuple(0, "zeros"),
    std::make_tuple(1, "zeros"),          std::make_tuple(2, "alternating"),
    std::make_tuple(3, "all_positive"),   std::make_tuple(4, "alternating")
};
```

**Тесты производительности (tests/performance/main.cpp)**

Генерация большого вектора для тестирования производительности:
```cpp
class IlinARunPerfTestProcesses : public ppc::util::BaseRunPerfTests<InType, OutType> {
  const int kVectorSize_ = 15000000;
  InType input_data_;
  void SetUp() override {
    input_data_.clear();
    input_data_.reserve(kVectorSize_);
    for (int i = 0; i < kVectorSize_; ++i) {
      input_data_.push_back(((i * 17) % 201) - 100);
    }
  }
};
```

## 6. Результаты экспериментов
**Окружение:**
- Процессор: 11th Gen Intel(R) Core(TM) i5-1135G7 @ 2.40GHz, 2419 МГц, ядер: 4, логических процессоров: 8
- Архитектура: AMD64
- Ядра: 4
- Оперативная память: 16 GB
- Операционная система: Windows 10 (базовая) / Ubuntu 24.04.3 LTS (сборочная)
- Подсистема: WSL2 (Windows Subsystem for Linux)

**Инструменты:**
- Компилятор: GCC 13.3.0 (Ubuntu 13.3.0-6ubuntu2~24.04)
- MPI реализация: Open MPI 4.1.6
- Тип сборки: Release 

**Переменные окружения:**
```
PPC_NUM_THREADS=1
PPC_NUM_PROC=2,4,6,8
```

**Тестовые данные:**
- Размер вектора: 15,000,000 элементов.
- Для получения значимых результатов каждый тест выполнялся несколько раз, для конечного подсчета использовались средние значения за 4 запуска.

## 7. Результаты и обсуждение

### 7.1 Корректность
Корректность проверена через 16 функциональных тестов:
- Чередующиеся знаки
- Все положительные/отрицательные элементы
- Случайные значения
- Нулевые элементы
- Граничные случаи (пустой вектор, 1 элемент, 2 элемента)

Все функциональные тесты пройдены для обеих реализаций.

### 7.2 Производительность

**Методика измерений:**
- Используются значения `task_run` из вывода тестов (время выполнения основного алгоритма)
- Время измеряется в секундах
- Для каждого количества процессов выполнено 4 запуска, взяты средние значения
- SEQ время взято среднее за 4 запуска

**Результаты для вектора 15,000,000 элементов:**
| Технология | Кол-во процессов | Время, сек | Ускорение | Эффективность |
|------------|------------------|------------|-----------|---------------|
| SEQ        | 1                | 0.02618840 | 1.00      | N/A           |
| MPI        | 2                | 0.02238310 | 1.17      | 58.5%         |
| MPI        | 4                | 0.01819460 | 1.44      | 36.0%         |
| MPI        | 6                | 0.01685246 | 1.55      | 25.8%         |
| MPI        | 8                | 0.01987937 | 1.32      | 16.5%         |

**Расчеты ускорения:**
- SEQ : 0.02618840 сек
- MPI 2 процесса: Speedup = 0.02618840 / 0.02238310 = 1.17
- MPI 4 процесса: Speedup = 0.02618840 / 0.01819460 = 1.44  
- MPI 6 процессов: Speedup = 0.02618840 / 0.01685246 = 1.55
- MPI 8 процессов: Speedup = 0.02618840 / 0.01987937 = 1.32

**Расчет эффективности:**
- MPI 2 процесса: (1.17 / 2) * 100% = 58.5%
- MPI 4 процесса: (1.44 / 4) * 100% = 36.0%
- MPI 6 процессов: (1.55 / 6) * 100% = 25.8%
- MPI 8 процессов: (1.32 / 8) * 100% = 16.5%

**Анализ результатов:**
- На 2 процессах достигается ускорение 1.17 раза с эффективностью 58.5%. Именно на двух процессах достигается баланс между полезными вычислениями и накладными расходами на коммуникацию.
- На 4 процессах ускорение увеличивается до 1.44 раза, но эффективность уже снижается до 36.0%. С увеличением числа процессов растут коммуникационные затраты.
- На 6 процессах наблюдается максимальное ускорение 1.55 раза, это соответствует оптимальному использованию физических ядер процессора. Эффективность составляет 25.8%.
- На 8 процессах производительность снижается до 1.32 раза, а эффективность - до 16.5%. Скорее всего возникает конкуренция за вычислительные ресурсы и увеличиваются накладные расходы на синхронизацию, из-за чего показатели метрик снижаются.

- Ускорение растет до 6 процессов, достигая максимума в 1.55 раза
- На 8 процессах наблюдается ухудшение производительности
- Эффективность монотонно снижается с увеличением числа процессов

**Вывод:** Алгоритм демонстрирует положительное ускорение на всех конфигурациях, достигая максимума 1.55 раза на 6 процессах, но эффективность использования вычислительных ресурсов снижается с ростом числа процессов, так как наша задача характеризуется низкой вычислительной сложностью и значительными коммуникационными затратами. Оптимальной конфигурацией для данной задачи является использование 4-6 процессов MPI.

## 8. Заключение

Успешно реализованы последовательный и параллельный алгоритмы подсчета чередований знаков и было проведено сравнение оных, что подтвердило практическую ценность MPI для задач анализа данных. Эффективность распараллеливания доказана экспериментально: на 6 процессах достигается ускорение 1.55 раза по сравнению с последовательной версией, что демонстрирует преимущества параллельных вычислений для обработки больших объемов данных, а оптимальная конфигурация для данной задачи составляет 4-6 процесса MPI.
Значимость работы заключается в демонстрации того, что даже для простых алгоритмов анализа данных правильно организованное распараллеливание позволяет сократить время вычислений, что особенно важно при обработке огромных пластов информации в реальных задачах статистики, математики и других областях научных исследований.

## 9. Список литературы
1. Chandra R. Parallel programming in OpenMP. – Morgan kaufmann, 2001.
2. R. L. Graham, G. M. Shipman, B. W. Barrett, R. H. Castain, G. Bosilca and A. Lumsdaine, "Open MPI: A High-Performance, Heterogeneous MPI," 2006 IEEE International Conference on Cluster Computing, Barcelona, Spain, 2006, pp. 1-9, doi: 10.1109/CLUSTR.2006.311904.
3. В.П. Гергель. Учебный курс "Введение в методы параллельного программирования". Раздел "Параллельное программирование с использованием OpenMP" // URL: http://www.hpcc.unn.ru/multicore/materials/tb/mc_ppr04.pdf, 2007.
4. Документация по курсу «Параллельное программирование» // URL: https://learning-process.github.io/parallel_programming_course/ru/index.html, 2025.