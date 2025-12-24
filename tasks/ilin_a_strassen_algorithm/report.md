# Умножение плотных матриц. Элементы типа double. Алгоритм Штрассена.

- Студент: Ильин Артемий Александрович, группа 3823Б1ПР1
- Технология: SEQ, MPI
- Вариант: 3

## 1. Введение
Умножение матриц является одной из фундаментальных операций в вычислительной математике, широко используемой в научных расчетах, машинном обучении и компьютерной графике. Классический алгоритм умножения матриц имеет кубическую сложность O(n³), что становится проблематичным вычислением при работе с матрицами больших размеров.
Алгоритм Штрассена, позволяет снизить асимптотическую сложность до O(n^log₂7) за счет использования рекурсивного подхода и сокращения количества умножений подматриц. В данной работе реализованы последовательная и параллельная версии алгоритма с использованием MPI для умножения матриц размера 1000×1000.

Целью данной работы является реализация последовательного и параллельного алгоритмов Штрассена для умножения плотных матриц типа double с использованием MPI и горизонтальной схемы распределения данных между процессами.

## 2. Постановка задачи
**Описание задачи:** Умножение двух квадратных плотных матриц A и B размерности n×n, где элементы матриц имеют тип double.

**Входные данные:**
- Целое n - размер матриц
- Вектор коэффициентов матрицы A размером n×n
- Вектор коэффициентов матрицы B размером n×n

**Выходные данные:** Вектор коэффициентов матрицы C = A × B размером n×n

**Ограничения:**
- Реализация должна использовать алгоритм Штрассена
- Для небольших подматриц (ниже порогового значения) используется классический алгоритм умножения
- В параллельной реализации должна использоваться горизонтальная схема распределения строк матриц между процессами
- Алгоритм должен корректно обрабатывать матрицы произвольного размера 

## 3. Описание базового алгоритма

### 3.1 Алгоритм Штрассена

Алгоритм Штрассена основан на рекурсивном разбиении матриц на подматрицы и выполнении семи рекурсивных умножений вместо восьми, требуемых классическим алгоритмом.

**Основные шаги алгоритма:**

1. Разбиение исходных матриц A и B на четыре подматрицы одинакового размера
2. Вычисление семи промежуточных произведений:
   - P1 = (A11 + A22) × (B11 + B22)
   - P2 = (A21 + A22) × B11
   - P3 = A11 × (B12 - B22)
   - P4 = A22 × (B21 - B11)
   - P5 = (A11 + A12) × B22
   - P6 = (A21 - A11) × (B11 + B12)
   - P7 = (A12 - A22) × (B21 + B22)
3. Вычисление результирующих подматриц:
   - C11 = P1 + P4 - P5 + P7
   - C12 = P3 + P5
   - C21 = P2 + P4
   - C22 = P1 + P3 - P2 + P6
4. Объединение подматриц в результирующую матрицу C

### 3.2 Особенности реализации

1. **Пороговое значение:** Используется константа `kThreshold = 64`. При размере матрицы меньше или равном этому значению алгоритм переключается на классическое умножение.

2. **Обработка произвольных размеров:** Для матриц, размер которых не является степенью числа два, выполняется дополнение до ближайшей степени двойки нулевыми элементами.

3. **Сложность:** 
   - Теоретическая сложность: O(n^log₂7)
   - Классический алгоритм: O(n³)

## 4. Схема распараллеливания

### 4.1 Общая архитектура MPI реализации

Параллельная реализация использует гибридный подход:
- Для матриц размером ≤ kThreshold (64) применяется классическое умножение
- Для больших матриц используется параллельная версия алгоритма Штрассена

### 4.2 Ключевые компоненты MPI реализации

**Основной метод RunImpl:**
```cpp
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
```

### 4.3 Параллельный алгоритм Штрассена

**Итеративная параллельная версия (ParallelStrassenIterative):**
```cpp
std::vector<double> IlinAStrassenAlgorithmMPI::ParallelStrassenIterative(
    const std::vector<double> &a, const std::vector<double> &b, int n) {
  MPI_Barrier(MPI_COMM_WORLD);

  if (n <= kThreshold) {
    return StrassenSequential(a, b, n);
  }

  int half = n / 2;
  std::size_t half_sq = static_cast<std::size_t>(half) * static_cast<std::size_t>(half);

  // Разбиение матриц на подматрицы
  std::vector<double> a11(half_sq), a12(half_sq), a21(half_sq), a22(half_sq);
  std::vector<double> b11(half_sq), b12(half_sq), b21(half_sq), b22(half_sq);

  if (world_rank_ == 0) {
    SplitMatrix(a, a11, a12, a21, a22, n);
    SplitMatrix(b, b11, b12, b21, b22, n);
  }

  // Рассылка подматриц всем процессам
  MPI_Bcast(a11.data(), half_sq, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(a12.data(), half_sq, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(a21.data(), half_sq, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(a22.data(), half_sq, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(b11.data(), half_sq, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(b12.data(), half_sq, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(b21.data(), half_sq, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(b22.data(), half_sq, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  auto [start_matrix, end_matrix] = CalculateMatrixRange(7);
  
  // Каждый процесс вычисляет свою часть произведений
  std::vector<double> p1(half_sq, 0.0), p2(half_sq, 0.0), p3(half_sq, 0.0),
                     p4(half_sq, 0.0), p5(half_sq, 0.0), p6(half_sq, 0.0),
                     p7(half_sq, 0.0);
  for (int matrix_idx = start_matrix; matrix_idx < end_matrix; matrix_idx++) {
    std::vector<double> result;
    ComputeSingleProduct(matrix_idx, a11, a12, a21, a22, 
                        b11, b12, b21, b22, half, result);
    switch (matrix_idx) {
      case 0: p1 = std::move(result); break;
      case 1: p2 = std::move(result); break;
      case 2: p3 = std::move(result); break;
      case 3: p4 = std::move(result); break;
      case 4: p5 = std::move(result); break;
      case 5: p6 = std::move(result); break;
      case 6: p7 = std::move(result); break;
    }
  }

  MPI_Barrier(MPI_COMM_WORLD);

  // Сбор результатов со всех процессов
  int matrix_size = half * half;
  
  std::vector<double> p1_total(half_sq, 0.0);
  std::vector<double> p2_total(half_sq, 0.0);
  std::vector<double> p3_total(half_sq, 0.0);
  std::vector<double> p4_total(half_sq, 0.0);
  std::vector<double> p5_total(half_sq, 0.0);
  std::vector<double> p6_total(half_sq, 0.0);
  std::vector<double> p7_total(half_sq, 0.0);

  MPI_Reduce(p1.data(), p1_total.data(), matrix_size, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(p2.data(), p2_total.data(), matrix_size, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(p3.data(), p3_total.data(), matrix_size, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(p4.data(), p4_total.data(), matrix_size, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(p5.data(), p5_total.data(), matrix_size, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(p6.data(), p6_total.data(), matrix_size, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(p7.data(), p7_total.data(), matrix_size, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

  if (world_rank_ == 0) {
    p1 = std::move(p1_total);
    p2 = std::move(p2_total);
    p3 = std::move(p3_total);
    p4 = std::move(p4_total);
    p5 = std::move(p5_total);
    p6 = std::move(p6_total);
    p7 = std::move(p7_total);
  }

  // Рассылка финальных значений всем процессам
  MPI_Bcast(p1.data(), matrix_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(p2.data(), matrix_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(p3.data(), matrix_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(p4.data(), matrix_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(p5.data(), matrix_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(p6.data(), matrix_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(p7.data(), matrix_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  // Вычисление результирующих подматриц
  std::vector<double> c11(half_sq), c12(half_sq), c21(half_sq), c22(half_sq);
  ComputeResultFromProducts(p1, p2, p3, p4, p5, p6, p7, half, c11, c12, c21, c22);

  // Объединение подматриц
  std::vector<double> c(static_cast<std::size_t>(n) * static_cast<std::size_t>(n));
  JoinMatrix(c, c11, c12, c21, c22, n);

  return c;
}
```

### 4.4 Распределение вычислений

**Метод CalculateMatrixRange для балансировки нагрузки:**
```cpp
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
```

### 4.5 Особенности коммуникаций

1. **Широковещательная рассылка (MPI_Bcast):** Используется для распространения подматриц среди всех процессов.

2. **Редукция (MPI_Reduce):** Применяется для суммирования частично вычисленных промежуточных произведений с операцией MPI_SUM.

3. **Барьерная синхронизация (MPI_Barrier):** Обеспечивает корректность выполнения на разных этапах алгоритма.

## 5. Детали реализации

### 5.1 Структура проекта
```
ilin_a_strassen_algorithm/
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
│   ├── functional/main.cpp     
│   └── performance/main.cpp    
└── settings.json
```

### 5.2 Основные структуры данных
```cpp
struct MatrixData {
  std::vector<double> A;    // Матрица A
  std::vector<double> B;    // Матрица B
  int size;                 // Размер матриц
};

struct ResultData {
  std::vector<double> C;    // Результирующая матрица
  int size;                 // Размер матрицы
};
```

### 5.3 Пороговые значения и оптимизации
- `kThreshold = 64` - переход на классическое умножение
- Дополнение матриц до степени двойки для работы алгоритма Штрассена
- Гибридный подход: классическое умножение для маленьких подматриц

## 6. Результаты экспериментов

### 6.1 Окружение
- **Процессор:** 11th Gen Intel(R) Core(TM) i5-1135G7 @ 2.40GHz (4 ядра, 8 потоков)
- **Оперативная память:** 16 GB
- **ОС:** Windows 10 / Ubuntu 24.04.3 LTS (WSL2)
- **Компилятор:** GCC 13.3.0
- **MPI:** Open MPI 4.1.6
- **Тип сборки:** Release

### 6.2 Тестовые данные
- **Размер матриц:** 1000×1000
- **Тип элементов:** double
- **Объем данных:** ~16 MB на матрицу (1000×1000×8 байт)
- **Матрицы:** заполнены случайными числами от 1.0 до 10.0

### 6.3 Методика измерений
- Измеряется время выполнения основного алгоритма 
- Для MPI реализаций 4 раза выполнены запуски с 2, 4, 6, и 8 процессами и взято среднее значение
- Для SEQ реализации 4 раза выполнены запуски и взято среднее

## 7. Результаты и обсуждение

### 7.1 Корректность
Корректность проверена через 6 функциональных тестов с матрицами различных размеров (16×16, 32×32, 64×64, 65×65, 127×127, 128×128). Все тесты пройдены.
## 7. Результаты и обсуждение

### 7.1 Корректность
Корректность проверена через 6 функциональных тестов с матрицами различных размеров (16×16, 32×32, 64×64, 65×65, 127×127, 128×128). Все тесты пройдены с точностью 10⁻⁶.

### 7.2 Производительность

**Результаты измерений MPI и SEQ реализаций:**

| Технология | Кол-во процессов | Время, сек | Ускорение | Эффективность |
|------------|------------------|------------|-----------|---------------|
| SEQ        | 1                | 1.213      | 1.00      | 100.0%        |
| MPI        | 2                | 0.413      | 2.94      | 147.0%        |
| MPI        | 4                | 0.297      | 4.08      | 102.0%        |
| MPI        | 6                | 0.371      | 3.27      | 54.5%         |
| MPI        | 8                | 0.397      | 3.06      | 38.2%         |

**Расчеты ускорения:**
- SEQ: 1.213 сек
- MPI 2 процесса: Speedup = 1.213 / 0.413 = 2.94 раза
- MPI 4 процесса: Speedup = 1.213 / 0.297 = 4.08 раза
- MPI 6 процессов: Speedup = 1.213 / 0.371 = 3.27 раза
- MPI 8 процессов: Speedup = 1.213 / 0.397 = 3.06 раза

**Расчет эффективности:**
- MPI 2 процесса: (2.94 / 2) * 100% = 147.0%
- MPI 4 процесса: (4.08 / 4) * 100% = 102.0%
- MPI 6 процессов: (3.27 / 6) * 100% = 54.5%
- MPI 8 процессов: (3.06 / 8) * 100% = 38.2%

**Анализ результатов:**

1. Наблюдается значительное ускорение 2.94× при использовании 2 процессов с эффективностью 147.0%. Это объясняется тем, что распределение данных между процессами позволяет более эффективно использовать кэш-память процессора, а параллельная обработка снижает затраты на управление рекурсивными вызовами. Алгоритм эффективно распределяет вычисления между двумя процессами.

2. При использовании 4 процессов достигается ускорение 4.08× с эффективностью 102.0%, что близко к идеальному линейному ускорению. Это оптимальная конфигурация для данной задачи.

3. На 6 и 8 процессах эффективность значительно снижается до 54.5% и 38.2% соответственно. Основными причинами является то, что на верхнем уровне рекурсии доступно только 7 независимых умножений, что ограничивает максимальное количество эффективно используемых процессов, а с ростом числа процессов возрастают накладные расходы на передачу данных. При распределении 7 задач между 6-8 процессами возникает дисбаланс вычислительной нагрузки, тестовая система имеет 4 физических ядра, что ограничивает реальный параллелизм.

4. **Оптимальное количество процессов**: Для данной задачи с матрицами 1000×1000 оптимальным является использование **4 процессов**, где достигается максимальное ускорение 4.08× с высокой эффективностью 102.0%.

5. **Влияние размера матриц**: Для матриц 1000×1000 алгоритм Штрассена демонстрирует хорошую эффективность, так как:
   - Глубокая рекурсия (1000 → 500 → 250 → 125 → переход на классическое умножение) позволяет эффективно использовать параллелизм
   - Относительные коммуникационные затраты меньше, чем для маленьких матриц
   - Становится заметным преимущество асимптотически меньшей сложности алгоритма Штрассена 

**Ключевые выводы:**
- Параллельная реализация алгоритма Штрассена демонстрирует высокую эффективность на 2-4 процессах.
- Максимальное ускорение 4.08× достигается при использовании 4 процессов..
- Алгоритм хорошо подходит для умножения больших матриц (1000×1000 и более).
- Эффективность использования вычислительных ресурсов тестовой системы максимальна при 4 процессах.

## 8. Заключение

Успешно реализованы последовательный и параллельный алгоритмы Штрассена для умножения плотных матриц. Особенностью реализации является поддержка матриц произвольного размера через дополнение до степени числа два и использование MPI для параллельных вычислений.
Реализованный алгоритм может быть использован для решения практических задач, требующих умножения больших плотных матриц, и может послужить основой для дальнейшей оптимизации.

## 9. Список литературы
1. Chandra R. Parallel programming in OpenMP. – Morgan kaufmann, 2001.
2. R. L. Graham, G. M. Shipman, B. W. Barrett, R. H. Castain, G. Bosilca and A. Lumsdaine, "Open MPI: A High-Performance, Heterogeneous MPI," 2006 IEEE International Conference on Cluster Computing, Barcelona, Spain, 2006, pp. 1-9, doi: 10.1109/CLUSTR.2006.311904.
3. В.П. Гергель. Учебный курс "Введение в методы параллельного программирования". Раздел "Параллельное программирование с использованием OpenMP" // URL: http://www.hpcc.unn.ru/multicore/materials/tb/mc_ppr04.pdf, 2007.
4. Документация по курсу «Параллельное программирование» // URL: https://learning-process.github.io/parallel_programming_course/ru/index.html, 2025.