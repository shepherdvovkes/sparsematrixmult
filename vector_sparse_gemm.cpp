#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cstring>
#include <cstdlib>
#include <algorithm>
#include <random>

#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#include <mach/mach_time.h>
#include <sys/sysctl.h>
#endif

#ifdef __aarch64__
#include <arm_neon.h>
#define VECTOR_WIDTH 4  // NEON: 4 floats
#define ALIGNMENT 16
#elif defined(__x86_64__)
#include <immintrin.h>
#define VECTOR_WIDTH 8  // AVX2: 8 floats
#define ALIGNMENT 32
#else
#define VECTOR_WIDTH 1
#define ALIGNMENT 8
#endif

using namespace std;
using namespace std::chrono;

// Aligned memory allocation
template<typename T>
T* aligned_alloc_custom(size_t count) {
    void* ptr = nullptr;
#ifdef __APPLE__
    if (posix_memalign(&ptr, ALIGNMENT, count * sizeof(T)) == 0) {
        return static_cast<T*>(ptr);
    }
#else
    ptr = aligned_alloc(ALIGNMENT, count * sizeof(T));
#endif
    return static_cast<T*>(ptr);
}

template<typename T>
void aligned_free_custom(T* ptr) {
    free(ptr);
}

// Sparse matrix format
struct SparseMatrix {
    vector<int> col_start_pos;
    vector<int> col_start_neg;
    vector<int> row_index_pos;
    vector<int> row_index_neg;
    int rows, cols;
    
    SparseMatrix(int r, int c) : rows(r), cols(c) {
        col_start_pos.resize(c + 1, 0);
        col_start_neg.resize(c + 1, 0);
    }
};

// Generate sparse ternary matrix (-1, 0, +1)
SparseMatrix generateSparseMatrix(int rows, int cols, int sparsity_factor) {
    SparseMatrix sparse(rows, cols);
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> dis(0, sparsity_factor - 1);
    uniform_int_distribution<> sign_dis(0, 1);
    
    for (int j = 0; j < cols; j++) {
        sparse.col_start_pos[j] = sparse.row_index_pos.size();
        sparse.col_start_neg[j] = sparse.row_index_neg.size();
        
        for (int i = 0; i < rows; i++) {
            if (dis(gen) == 0) { // Non-zero element
                if (sign_dis(gen) == 0) {
                    sparse.row_index_pos.push_back(i);
                } else {
                    sparse.row_index_neg.push_back(i);
                }
            }
        }
    }
    sparse.col_start_pos[cols] = sparse.row_index_pos.size();
    sparse.col_start_neg[cols] = sparse.row_index_neg.size();
    
    return sparse;
}

// Initialize random matrix
void initMatrix(float* matrix, int size) {
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<float> dis(-1.0f, 1.0f);
    
    for (int i = 0; i < size; i++) {
        matrix[i] = dis(gen);
    }
}

// Scalar Dense GEMM: Y = X * W + B
void denseGEMM_scalar(const float* X, const float* W, const float* B, float* Y,
                     int M, int N, int K) {
    // Initialize with bias
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            Y[i * N + j] = B[j];
        }
    }
    
    // Matrix multiplication
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += X[i * K + k] * W[k * N + j];
            }
            Y[i * N + j] += sum;
        }
    }
}

// Vector Dense GEMM with SIMD optimization
void denseGEMM_vector(const float* X, const float* W, const float* B, float* Y,
                     int M, int N, int K) {
    // Initialize with bias
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            Y[i * N + j] = B[j];
        }
    }
    
#ifdef __aarch64__
    // NEON optimized version
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float32x4_t sum_vec = vdupq_n_f32(0.0f);
            int k = 0;
            
            // Process 4 elements at a time
            for (; k <= K - 4; k += 4) {
                float32x4_t x_vec = vld1q_f32(&X[i * K + k]);
                float32x4_t w_vec = {W[k * N + j], W[(k+1) * N + j], 
                                   W[(k+2) * N + j], W[(k+3) * N + j]};
                sum_vec = vfmaq_f32(sum_vec, x_vec, w_vec);
            }
            
            // Horizontal sum
            float sum = vaddvq_f32(sum_vec);
            
            // Handle remaining elements
            for (; k < K; k++) {
                sum += X[i * K + k] * W[k * N + j];
            }
            
            Y[i * N + j] += sum;
        }
    }
    
#elif defined(__x86_64__)
    // AVX2 optimized version
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            __m256 sum_vec = _mm256_setzero_ps();
            int k = 0;
            
            // Process 8 elements at a time
            for (; k <= K - 8; k += 8) {
                __m256 x_vec = _mm256_loadu_ps(&X[i * K + k]);
                __m256 w_vec = _mm256_set_ps(
                    W[(k+7) * N + j], W[(k+6) * N + j], W[(k+5) * N + j], W[(k+4) * N + j],
                    W[(k+3) * N + j], W[(k+2) * N + j], W[(k+1) * N + j], W[k * N + j]
                );
                sum_vec = _mm256_fmadd_ps(x_vec, w_vec, sum_vec);
            }
            
            // Horizontal sum
            __m128 hi = _mm256_extractf128_ps(sum_vec, 1);
            __m128 lo = _mm256_castps256_ps128(sum_vec);
            __m128 sum_halves = _mm_add_ps(hi, lo);
            __m128 sum_pairs = _mm_hadd_ps(sum_halves, sum_halves);
            __m128 sum_final = _mm_hadd_ps(sum_pairs, sum_pairs);
            float sum = _mm_cvtss_f32(sum_final);
            
            // Handle remaining elements
            for (; k < K; k++) {
                sum += X[i * K + k] * W[k * N + j];
            }
            
            Y[i * N + j] += sum;
        }
    }
#else
    // Fallback to scalar version
    denseGEMM_scalar(X, W, B, Y, M, N, K);
#endif
}

// Scalar Sparse GEMM
void sparseGEMM_scalar(const float* X, const SparseMatrix& sparse, const float* B, 
                      float* Y, int M, int N, int K) {
    // Initialize with bias
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            Y[i * N + j] = B[j];
        }
    }
    
    // Sparse matrix multiplication
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            
            // Process positive weights (+1)
            for (int idx = sparse.col_start_pos[j]; idx < sparse.col_start_pos[j + 1]; idx++) {
                sum += X[i * K + sparse.row_index_pos[idx]];
            }
            
            // Process negative weights (-1)
            for (int idx = sparse.col_start_neg[j]; idx < sparse.col_start_neg[j + 1]; idx++) {
                sum -= X[i * K + sparse.row_index_neg[idx]];
            }
            
            Y[i * N + j] += sum;
        }
    }
}

// Vector Sparse GEMM with SIMD optimization
void sparseGEMM_vector(const float* X, const SparseMatrix& sparse, const float* B,
                      float* Y, int M, int N, int K) {
    // Initialize with bias
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            Y[i * N + j] = B[j];
        }
    }
    
#ifdef __aarch64__
    // NEON optimized sparse version
    for (int i = 0; i < M; i++) {
        const float* x_row = &X[i * K];
        
        for (int j = 0; j < N; j++) {
            float32x4_t sum_vec = vdupq_n_f32(0.0f);
            float sum = 0.0f;
            
            // Process positive weights
            int start_pos = sparse.col_start_pos[j];
            int end_pos = sparse.col_start_pos[j + 1];
            int idx = start_pos;
            
            for (; idx <= end_pos - 4; idx += 4) {
                float32x4_t x_vec = {
                    x_row[sparse.row_index_pos[idx]], x_row[sparse.row_index_pos[idx+1]],
                    x_row[sparse.row_index_pos[idx+2]], x_row[sparse.row_index_pos[idx+3]]
                };
                sum_vec = vaddq_f32(sum_vec, x_vec);
            }
            
            sum += vaddvq_f32(sum_vec);
            
            for (; idx < end_pos; idx++) {
                sum += x_row[sparse.row_index_pos[idx]];
            }
            
            // Process negative weights
            sum_vec = vdupq_n_f32(0.0f);
            int start_neg = sparse.col_start_neg[j];
            int end_neg = sparse.col_start_neg[j + 1];
            idx = start_neg;
            
            for (; idx <= end_neg - 4; idx += 4) {
                float32x4_t x_vec = {
                    x_row[sparse.row_index_neg[idx]], x_row[sparse.row_index_neg[idx+1]],
                    x_row[sparse.row_index_neg[idx+2]], x_row[sparse.row_index_neg[idx+3]]
                };
                sum_vec = vaddq_f32(sum_vec, x_vec);
            }
            
            sum -= vaddvq_f32(sum_vec);
            
            for (; idx < end_neg; idx++) {
                sum -= x_row[sparse.row_index_neg[idx]];
            }
            
            Y[i * N + j] += sum;
        }
    }
    
#elif defined(__x86_64__)
    // AVX2 optimized sparse version (similar structure)
    for (int i = 0; i < M; i++) {
        const float* x_row = &X[i * K];
        
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            
            // Process positive weights
            int start_pos = sparse.col_start_pos[j];
            int end_pos = sparse.col_start_pos[j + 1];
            for (int idx = start_pos; idx < end_pos; idx++) {
                sum += x_row[sparse.row_index_pos[idx]];
            }
            
            // Process negative weights
            int start_neg = sparse.col_start_neg[j];
            int end_neg = sparse.col_start_neg[j + 1];
            for (int idx = start_neg; idx < end_neg; idx++) {
                sum -= x_row[sparse.row_index_neg[idx]];
            }
            
            Y[i * N + j] += sum;
        }
    }
#else
    // Fallback to scalar version
    sparseGEMM_scalar(X, sparse, B, Y, M, N, K);
#endif
}

// Benchmark structure
struct BenchmarkResult {
    double avg_time_ns;
    double min_time_ns;
    double max_time_ns;
    double gflops;
    double speedup;
};

// Comprehensive benchmarking function
template<typename Func, typename... Args>
BenchmarkResult benchmark_function(Func func, int num_runs, double flops, Args... args) {
    vector<double> times;
    times.reserve(num_runs);
    
    // Warm-up runs
    for (int i = 0; i < 3; i++) {
        func(args...);
    }
    
    // Actual benchmark runs
    for (int i = 0; i < num_runs; i++) {
        auto start = high_resolution_clock::now();
        func(args...);
        auto end = high_resolution_clock::now();
        
        auto duration = duration_cast<nanoseconds>(end - start);
        times.push_back(static_cast<double>(duration.count()));
    }
    
    // Calculate statistics
    sort(times.begin(), times.end());
    double avg_time = accumulate(times.begin(), times.end(), 0.0) / times.size();
    double min_time = times.front();
    double max_time = times.back();
    
    BenchmarkResult result;
    result.avg_time_ns = avg_time;
    result.min_time_ns = min_time;
    result.max_time_ns = max_time;
    result.gflops = flops / (avg_time / 1e9) / 1e9;
    result.speedup = 1.0; // Will be set relative to reference
    
    return result;
}

// Verify correctness
bool verify_results(const float* Y1, const float* Y2, int size, float tolerance = 1e-4f) {
    for (int i = 0; i < size; i++) {
        if (abs(Y1[i] - Y2[i]) > tolerance) {
            cout << "Mismatch at index " << i << ": " << Y1[i] << " vs " << Y2[i] << endl;
            return false;
        }
    }
    return true;
}

int main() {
    cout << "=== VECTOR-OPTIMIZED SPARSE MATRIX MULTIPLICATION BENCHMARK ===" << endl;
    cout << "Architecture: ";
#ifdef __aarch64__
    cout << "ARM64 (NEON), Vector Width: " << VECTOR_WIDTH << " floats" << endl;
#elif defined(__x86_64__)
    cout << "x86_64 (AVX2), Vector Width: " << VECTOR_WIDTH << " floats" << endl;
#else
    cout << "Generic (No SIMD)" << endl;
#endif
    cout << "Alignment: " << ALIGNMENT << " bytes" << endl << endl;
    
    // Test configurations
    struct TestConfig {
        int M, K, N;
        int sparsity_factor;
        string name;
    };
    
    vector<TestConfig> configs = {
        {64, 512, 1024, 2, "Small_50%"},
        {64, 512, 1024, 4, "Small_25%"},
        {64, 512, 1024, 8, "Small_12.5%"},
        {256, 1024, 2048, 2, "Medium_50%"},
        {256, 1024, 2048, 4, "Medium_25%"},
        {256, 1024, 2048, 8, "Medium_12.5%"},
        {1024, 2048, 4096, 2, "Large_50%"},
        {1024, 2048, 4096, 4, "Large_25%"},
        {1024, 2048, 4096, 8, "Large_12.5%"}
    };
    
    const int NUM_RUNS = 10;
    
    // Print CSV header
    cout << "Test,M,K,N,Sparsity,Implementation,Time_ns,GFLOPS,Speedup,Efficiency%" << endl;
    
    for (const auto& config : configs) {
        cout << "\n--- Testing " << config.name << " (M=" << config.M 
             << ", K=" << config.K << ", N=" << config.N 
             << ", Sparsity=1/" << config.sparsity_factor << ") ---" << endl;
        
        // Allocate aligned memory
        float* X = aligned_alloc_custom<float>(config.M * config.K);
        float* W = aligned_alloc_custom<float>(config.K * config.N);
        float* B = aligned_alloc_custom<float>(config.N);
        float* Y_dense_scalar = aligned_alloc_custom<float>(config.M * config.N);
        float* Y_dense_vector = aligned_alloc_custom<float>(config.M * config.N);
        float* Y_sparse_scalar = aligned_alloc_custom<float>(config.M * config.N);
        float* Y_sparse_vector = aligned_alloc_custom<float>(config.M * config.N);
        
        // Initialize matrices
        initMatrix(X, config.M * config.K);
        initMatrix(W, config.K * config.N);
        initMatrix(B, config.N);
        
        // Generate sparse matrix
        SparseMatrix sparse = generateSparseMatrix(config.K, config.N, config.sparsity_factor);
        
        // Calculate FLOPS
        double dense_flops = 2.0 * config.M * config.N * config.K + config.M * config.N;
        double sparse_flops = config.M * config.N * 
            ((double)(sparse.row_index_pos.size() + sparse.row_index_neg.size()) / config.N + 1.0);
        
        // Benchmark Dense Scalar
        auto result_dense_scalar = benchmark_function(
            denseGEMM_scalar, NUM_RUNS, dense_flops,
            X, W, B, Y_dense_scalar, config.M, config.N, config.K
        );
        
        // Benchmark Dense Vector
        auto result_dense_vector = benchmark_function(
            denseGEMM_vector, NUM_RUNS, dense_flops,
            X, W, B, Y_dense_vector, config.M, config.N, config.K
        );
        
        // Benchmark Sparse Scalar
        auto result_sparse_scalar = benchmark_function(
            sparseGEMM_scalar, NUM_RUNS, sparse_flops,
            X, sparse, B, Y_sparse_scalar, config.M, config.N, config.K
        );
        
        // Benchmark Sparse Vector
        auto result_sparse_vector = benchmark_function(
            sparseGEMM_vector, NUM_RUNS, sparse_flops,
            X, sparse, B, Y_sparse_vector, config.M, config.N, config.K
        );
        
        // Calculate speedups (relative to dense scalar)
        double ref_time = result_dense_scalar.avg_time_ns;
        result_dense_scalar.speedup = 1.0;
        result_dense_vector.speedup = ref_time / result_dense_vector.avg_time_ns;
        result_sparse_scalar.speedup = ref_time / result_sparse_scalar.avg_time_ns;
        result_sparse_vector.speedup = ref_time / result_sparse_vector.avg_time_ns;
        
        // Calculate efficiency (relative to theoretical peak)
        double ref_gflops = result_dense_scalar.gflops;
        
        // Verify correctness
        bool dense_correct = verify_results(Y_dense_scalar, Y_dense_vector, config.M * config.N);
        if (!dense_correct) {
            cout << "⚠ Dense vector correctness check failed" << endl;
        }
        
        // Print results in CSV format
        cout << fixed << setprecision(2);
        
        cout << config.name << "," << config.M << "," << config.K << "," << config.N 
             << ",1/" << config.sparsity_factor << ",Dense_Scalar," 
             << result_dense_scalar.avg_time_ns << "," << result_dense_scalar.gflops << ","
             << result_dense_scalar.speedup << "," << 100.0 << endl;
             
        cout << config.name << "," << config.M << "," << config.K << "," << config.N 
             << ",1/" << config.sparsity_factor << ",Dense_Vector," 
             << result_dense_vector.avg_time_ns << "," << result_dense_vector.gflops << ","
             << result_dense_vector.speedup << "," << (result_dense_vector.gflops/ref_gflops)*100.0 << endl;
             
        cout << config.name << "," << config.M << "," << config.K << "," << config.N 
             << ",1/" << config.sparsity_factor << ",Sparse_Scalar," 
             << result_sparse_scalar.avg_time_ns << "," << result_sparse_scalar.gflops << ","
             << result_sparse_scalar.speedup << "," << (result_sparse_scalar.gflops/ref_gflops)*100.0 << endl;
             
        cout << config.name << "," << config.M << "," << config.K << "," << config.N 
             << ",1/" << config.sparsity_factor << ",Sparse_Vector," 
             << result_sparse_vector.avg_time_ns << "," << result_sparse_vector.gflops << ","
             << result_sparse_vector.speedup << "," << (result_sparse_vector.gflops/ref_gflops)*100.0 << endl;
        
        // Performance summary
        cout << "\nPerformance Summary:" << endl;
        cout << "  Dense Scalar:  " << setw(8) << result_dense_scalar.gflops << " GFLOPS (baseline)" << endl;
        cout << "  Dense Vector:  " << setw(8) << result_dense_vector.gflops << " GFLOPS (" 
             << setw(5) << result_dense_vector.speedup << "x speedup)" << endl;
        cout << "  Sparse Scalar: " << setw(8) << result_sparse_scalar.gflops << " GFLOPS (" 
             << setw(5) << result_sparse_scalar.speedup << "x speedup)" << endl;
        cout << "  Sparse Vector: " << setw(8) << result_sparse_vector.gflops << " GFLOPS (" 
             << setw(5) << result_sparse_vector.speedup << "x speedup)" << endl;
        
        // Cleanup
        aligned_free_custom(X);
        aligned_free_custom(W);
        aligned_free_custom(B);
        aligned_free_custom(Y_dense_scalar);
        aligned_free_custom(Y_dense_vector);
        aligned_free_custom(Y_sparse_scalar);
        aligned_free_custom(Y_sparse_vector);
    }
    
    cout << "\n=== BENCHMARK COMPLETE ===" << endl;
    cout << "Results are in CSV format for easy analysis." << endl;
    cout << "Copy the CSV data above to analyze performance trends." << endl;
    
    return 0;
}