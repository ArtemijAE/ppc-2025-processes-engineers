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

``` cpp
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
MPI_Bcast(&global_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
```
Все процессы должны знать общий размер вектора для вычисления размеров своих блоков.

**Распределение данных**
```cpp
MPI_Scatterv(global_size > 0 ? const_cast<int*>(global_vec.data()) : nullptr,
             counts.data(), displs.data(), MPI_INT,
             local_vec.data(), static_cast<int>(local_vec.size()), MPI_INT,
             0, MPI_COMM_WORLD);
```
Процесс 0 распределяет части вектора между всеми процессами. Массивы `counts` и `displs` определяют размер и смещение для каждого процесса.

**Локальный подсчет чередований**
```cpp
int local_alternations = 0;
for (size_t i = 0; i < local_vec.size() - 1; ++i) {
    if ((local_vec[i] < 0 && local_vec[i + 1] >= 0) || 
        (local_vec[i] >= 0 && local_vec[i + 1] < 0)) {
        local_alternations++;
    }
}
```
Каждый процесс независимо подсчитывает чередования в своем блоке.

**Сбор граничных элементов**
```cpp
int left_boundary = local_vec.empty() ? 0 : local_vec.front();
int right_boundary = local_vec.empty() ? 0 : local_vec.back();

std::vector<int> boundaries(2 * world_size);
MPI_Gather(&left_boundary, 1, MPI_INT, boundaries.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
MPI_Gather(&right_boundary, 1, MPI_INT, boundaries.data() + world_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
```
Процессы передают первый и последний элемент своего блока процессу 0 для проверки чередований на границах.

**Суммирование результатов**
```cpp
int total_alternations = 0;
MPI_Reduce(&local_alternations, &total_alternations, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
```
Локальные подсчеты суммируются на процессе 0.

**Добавление граничных чередований**
```cpp
if (world_rank == 0) {
    for (int i = 0; i < world_size - 1; ++i) {
        int right_of_i = boundaries[world_size + i];
        int left_of_next = boundaries[i + 1];
        if ((right_of_i < 0 && left_of_next >= 0) || 
            (right_of_i >= 0 && left_of_next < 0)) {
            total_alternations++;
        }
    }
    GetOutput() = total_alternations;
}
```
Процесс 0 проверяет чередования между соседними блоками и добавляет их к общему результату.

**Синхронизация**
```cpp
MPI_Barrier(MPI_COMM_WORLD);
```
Все процессы синхронизируются перед завершением.

**Полный код параллельного алгоритма:**
```cpp
bool IlinAAlternationsSignsOfValVecMPI::RunImpl() {
    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    const std::vector<int>& global_vec = GetInput();
    int global_size = static_cast<int>(global_vec.size());
    MPI_Bcast(&global_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
    int local_size = global_size / world_size;
    int remainder = global_size % world_size;
    std::vector<int> local_vec(local_size + (world_rank < remainder ? 1 : 0));
    std::vector<int> counts(world_size), displs(world_size);
    if (world_rank == 0) {
        for (int i = 0; i < world_size; ++i) {
            counts[i] = local_size + (i < remainder ? 1 : 0);
            displs[i] = (i == 0) ? 0 : displs[i - 1] + counts[i - 1];
        }
    }
    MPI_Scatterv(global_vec.data(), counts.data(), displs.data(), MPI_INT,
                 local_vec.data(), local_vec.size(), MPI_INT, 0, MPI_COMM_WORLD);
    int local_alternations = 0;
    for (size_t i = 0; i < local_vec.size() - 1; ++i) {
        if ((local_vec[i] < 0 && local_vec[i + 1] >= 0) || 
            (local_vec[i] >= 0 && local_vec[i + 1] < 0)) {
            local_alternations++;
        }
    }
    int left = local_vec.empty() ? 0 : local_vec.front();
    int right = local_vec.empty() ? 0 : local_vec.back();
    std::vector<int> boundaries(2 * world_size);
    MPI_Gather(&left, 1, MPI_INT, boundaries.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Gather(&right, 1, MPI_INT, boundaries.data() + world_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
    int total = 0;
    MPI_Reduce(&local_alternations, &total, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    if (world_rank == 0) {
        for (int i = 0; i < world_size - 1; ++i) {
            int right_of_i = boundaries[world_size + i];
            int left_of_next = boundaries[i + 1];
            if ((right_of_i < 0 && left_of_next >= 0) || 
                (right_of_i >= 0 && left_of_next < 0)) {
                total++;
            }
        }
        GetOutput() = total;
    }
    MPI_Barrier(MPI_COMM_WORLD);
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

**ops_mpi.hpp** - объявление класса:
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
};
```

**ops_mpi.cpp** - содержит полную реализацию параллельного алгоритма, описанного в разделе 4.

**Функциональные тесты (tests/functional/main.cpp)**

Генерация тестовых данных различных типов:
```cpp
const std::array<TestType, 12> kTestParam = {
    std::make_tuple(10, "alternating"),    
    std::make_tuple(100, "alternating"),
    std::make_tuple(1000, "alternating"),
    std::make_tuple(10, "all_positive"),   
    std::make_tuple(100, "all_positive"),
    std::make_tuple(10, "all_negative"),  
    std::make_tuple(100, "all_negative"),
    std::make_tuple(50, "random"),         
    std::make_tuple(500, "random"),
    std::make_tuple(10, "zeros"),         
    std::make_tuple(1, "all_positive"),    
    std::make_tuple(0, "all_positive")   
};
```

**Тесты производительности (tests/performance/main.cpp)**

Генерация большого вектора для тестирования производительности:
```cpp
class IlinARunPerfTestProcesses : public ppc::util::BaseRunPerfTests<InType, OutType> {
    const int kVectorSize_ = 15000000;  
    InType input_data_{};
    void SetUp() override {
        input_data_.clear();
        input_data_.reserve(kVectorSize_);
        for (int i = 0; i < kVectorSize_; ++i) {
            input_data_.push_back((i * 17) % 201 - 100);
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

**Инструменты**:
- Компилятор: GCC 13.3.0 (Ubuntu 13.3.0-6ubuntu2~24.04)
- MPI реализация: Open MPI 4.1.6
- Тип сборки: Release 

**Переменные окружения:**
```
PPC_NUM_THREADS=1
PPC_NUM_PROC=2,4,8
```

**Тестовые данные:**
- Размер вектора: 15,000,000 элементов

## 7. Результаты и обсуждение

### 7.1 Корректность
Корректность проверена через 12 функциональных тестов:
- Чередующиеся знаки
- Все положительные/отрицательные элементы
- Случайные значения
- Нулевые элементы
- Граничные случаи (пустой вектор, 1 элемент)

Все функциональные тесты пройдены для обеих реализаций.

### 7.2 Производительность

**Методика измерений:**
- Используются значения `task_run` из вывода тестов (основной алгоритм)
- Время измеряется в секундах

**Результаты для вектора 15,000,000 элементов:**

| Технология | Кол-во процессов | Время, сек | Ускорение | Эффективность |
|------------|------------------|------------|-----------|---------------|
| SEQ        | 1                | 0.03931    | 1.00      | N/A           |
| MPI        | 2                | 0.02834    | 1.39      | 69.5%         |
| MPI        | 4                | 0.02350    | 1.67      | 41.8%         |
| MPI        | 8                | 0.02109    | 1.86      | 23.3%         |

**Расчеты:**
- SEQ: 0.03931 сек
- MPI 2: 0.02834 сек  Speedup = 0.03931/0.02834 = 1.39
- MPI 4: 0.02350 сек  Speedup = 0.03931/0.02350 = 1.67  
- MPI 8: 0.02109 сек  Speedup = 0.03931/0.02109 = 1.86

**Эффективность:**
- MPI 2: (1.39 / 2) * 100% = 69.5%
- MPI 4: (1.67 / 4) * 100% = 41.8%
- MPI 8: (1.86 / 8) * 100% = 23.3%

**Анализ результатов:**
- На 2 процессах достигается ускорение 1.39x с высокой эффективностью 69.5%
- На 4 процессах максимальное ускорение 1.67x при эффективности 41.8%
- На 8 процессах ускорение 1.86x, но эффективность падает до 23.3%
- Оптимальное количество процессов для данной задачи - 2-4

**Вывод:** Алгоритм демонстрирует эффективное распараллеливание на 2-4 процессах, однако дальнейшее увеличение числа процессов не приводит к значительному улучшению производительности. Связано это с преобладанием коммуникационных затрат над вычислительной нагрузкой при большом количестве процессов. Для вектора из 15 миллионов элементов максимальное ускорение в 1.86 раза достигается на 8 процессах, но с низкой эффективностью (23.3%), а на 2 процессах наблюдается наиболее сбалансированное соотношение ускорения (1.39x) и эффективности (69.5%).

## 8. Заключение

Успешно реализованы последовательный и параллельный алгоритмы подсчета чередований знаков и проведено сравнение оных, что подтвердило практическую ценность MPI для задач анализа данных. Эффективность распараллеливания доказана экспериментально: на 4 процессах достигается ускорение 1.67x по сравнению с последовательной версией, что демонстрирует преимущества параллельных вычислений для обработки больших объемов данных, а оптимальная конфигурация для данной задачи составляет 2-4 процесса MPI.
Значимость работы заключается в демонстрации того, что даже для простых алгоритмов анализа данных правильно организованное распараллеливание позволяет сократить время вычислений, что особенно важно при обработке огромных пластов информации в реальных задачах статистики, математики и других областях научных исследований.

## 9. Список литературы

1. Chandra R. Parallel programming in OpenMP. – Morgan kaufmann, 2001.
2. R. L. Graham, G. M. Shipman, B. W. Barrett, R. H. Castain, G. Bosilca and A. Lumsdaine, "Open MPI: A High-Performance, Heterogeneous MPI," 2006 IEEE International Conference on Cluster Computing, Barcelona, Spain, 2006, pp. 1-9, doi: 10.1109/CLUSTR.2006.311904.
3. В.П. Гергель. Учебный курс "Введение в методы параллельного программирования". Раздел "Параллельное программирование с использованием OpenMP" // URL: http://www.hpcc.unn.ru/multicore/materials/tb/mc_ppr04.pdf, 2007.
4. Документация по курсу «Параллельное программирование» // URL: https://learning-process.github.io/parallel_programming_course/ru/index.html, 2025.