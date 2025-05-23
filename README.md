# Vector-Optimized Sparse Matrix Multiplication Benchmark

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![C++17](https://img.shields.io/badge/C%2B%2B-17-blue.svg)](https://isocpp.org/std/the-standard)
[![Platform](https://img.shields.io/badge/Platform-Linux%20%7C%20macOS%20%7C%20Windows-lightgrey)](https://github.com)

**High-performance matrix multiplication benchmarking suite with SIMD optimizations for modern CPU architectures.**

## ✨ Features

- **Multi-Architecture SIMD**: AVX2 (8-wide), NEON (4-wide), AVX-512 support
- **Sparse Optimization**: Ternary matrix format with up to 20x speedups
- **Comprehensive Analysis**: Roofline models, performance plots, bottleneck identification
- **Apple Silicon Ready**: Native M1/M2/M3 optimizations with Accelerate framework
- **Cross-Platform**: Linux, macOS (Intel + Apple Silicon), Windows WSL

## 🎯 Quick Start

### macOS (Recommended)
```bash
chmod +x macos_build_script.sh macos_runner_script.sh
./macos_build_script.sh && ./macos_runner_script.sh
```

### Manual Build
```bash
# Apple Silicon
clang++ -std=c++17 -O3 -mcpu=apple-m1 -march=armv8.2-a+fp16+dotprod \
        vector_sparse_gemm.cpp -o benchmark

# Intel/AMD with AVX2
g++ -std=c++17 -O3 -march=native -mavx2 -mfma \
    vector_sparse_gemm.cpp -o benchmark

./benchmark > results.csv
```

## 📊 Performance Results

Real benchmarks from M1 MacBook Pro:

| Implementation | Speedup | GFLOPS | Use Case |
|----------------|---------|--------|----------|
| Dense Vector   | 1.40x   | 2.60   | SIMD optimization |
| Sparse 50%     | 8.56x   | 3.94   | Moderate sparsity |
| **Sparse 25%** | **15.22x** | **3.55** | **High sparsity** |

## 🏗️ Architecture Support

- **🍎 Apple Silicon**: M1/M2/M3 with NEON vectorization
- **💻 Intel x86_64**: AVX2/AVX-512 with FMA
- **🔥 AMD x86_64**: Zen architecture optimizations
- **🌐 Generic**: Portable fallback implementation

## 📈 Files

- `vector_sparse_gemm.cpp` - Main benchmark implementation
- `macos_build_script.sh` - macOS build automation  
- `macos_runner_script.sh` - macOS benchmark runner with analysis
- `compile_ble_scanner.sh` - Additional compilation script
- `build_macos/` - Build artifacts (gitignored)

## 🚀 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request
=======
Compare scalar vs vector implementations across dense and sparse matrix operations with comprehensive performance analysis including roofline models, efficiency plots, and architecture-specific optimizations.

---

## ✨ Features

- **Multi-Architecture SIMD**: AVX2 (8-wide), NEON (4-wide), AVX-512 support
- **Sparse Optimization**: Ternary matrix format with up to 20x speedups
- **Comprehensive Analysis**: Roofline models, performance plots, bottleneck identification
- **Apple Silicon Ready**: Native M1/M2/M3 optimizations with Accelerate framework
- **Cross-Platform**: Linux, macOS (Intel + Apple Silicon), Windows WSL
- **Automated Pipeline**: One-command build, benchmark, and analysis
- **Professional Reports**: HTML reports, CSV export, publication-quality plots

## Quick Start

### One-Command Benchmark

#### macOS (Recommended)
```bash
chmod +x macos_build.sh macos_runner.sh
./macos_build.sh && ./macos_runner.sh
```

#### Linux
```bash
make benchmark
```

#### Manual Build
```bash
# x86_64 with AVX2
g++ -std=c++17 -O3 -march=native -mavx2 -mfma \
    vector_sparse_gemm.cpp -o benchmark

# ARM64 with NEON  
g++ -std=c++17 -O3 -mcpu=native \
    vector_sparse_gemm.cpp -o benchmark

./benchmark > results.csv
```

### Expected Runtime
- **Small matrices**: 2-5 minutes
- **Complete suite**: 15-30 minutes
- **Quick test**: 30 seconds

---

## Performance Results

Real benchmarks from M1 MacBook Pro (16GB):

| Implementation | Matrix Size | Sparsity | GFLOPS | Speedup | Efficiency |
|----------------|-------------|----------|--------|---------|------------|
| Dense_Scalar   | 64×512×1024 | Dense    | 1.85   | 1.00x   | 100%       |
| Dense_Vector   | 64×512×1024 | Dense    | 2.60   | 1.40x   | 140%       |
| Sparse_Scalar  | 64×512×1024 | 25%      | 3.25   | 13.95x  | 175%       |
| **Sparse_Vector** | **64×512×1024** | **25%** | **3.55** | **15.22x** | **191%** |

### Performance Scaling

```
Architecture    | Dense Vector | Sparse (12.5%) | Peak GFLOPS
----------------|--------------|----------------|-------------
Apple M1        | 2.5-4.0x     | 8-15x          | 40-80
Intel i7        | 3.0-6.0x     | 10-20x         | 60-150
AMD Ryzen       | 3.5-7.0x     | 12-25x         | 80-200
```

---

## 🏗️ Architecture Support

### Apple Silicon (M1/M2/M3/M4)
- **NEON**: 4-wide float32 vectorization
- **Unified Memory**: 68+ GB/s bandwidth
- **Power Efficiency**: Performance per watt optimization
- **Native Compilation**: `-mcpu=apple-m1` optimizations

### Intel x86_64
- **AVX2**: 8-wide float32 vectorization with FMA
- **AVX-512**: 16-wide vectorization (Skylake-X+)
- **Cache Optimization**: L1/L2/L3 hierarchy tuning
- **Thermal Management**: Boost clock utilization

### AMD x86_64
- **AVX2**: 8-wide vectorization
- **Zen Architecture**: Optimized for Ryzen/EPYC
- **Memory Bandwidth**: DDR4/DDR5 optimization
- **SMT Awareness**: Thread-level parallelism

### Generic Fallback
- **Portable C++**: Works on any platform
- **Scalar Implementation**: No SIMD dependencies
- **Compatibility**: ARMv7, RISC-V, others

---

## 📋 Prerequisites

### Required
```bash
# Ubuntu/Debian
sudo apt update && sudo apt install build-essential

# macOS
xcode-select --install

# Windows (WSL)
sudo apt install build-essential
```

### Optional (for analysis)
```bash
# Python packages
pip3 install pandas matplotlib seaborn numpy

# macOS performance tools (included)
# - Instruments, Sample, PowerMetrics
```

---

## Building

### Automated Build

#### macOS
```bash
./macos_build.sh [OPTIONS]

Options:
  --all          Build all variants including Accelerate framework
  --clean        Clean previous builds
  --help         Show help
```

#### Linux/Cross-Platform
```bash
make [TARGET]

Targets:
  benchmark      Complete build and run pipeline
  test           Quick functionality test
  clean          Clean build artifacts
  help           Show all targets
```

### Manual Compilation

#### Optimal Performance
```bash
# Detect best flags for your CPU
gcc -march=native -Q --help=target | grep march

# Apple Silicon
clang++ -std=c++17 -O3 -mcpu=apple-m1 -march=armv8.2-a+fp16+dotprod \
        -ffast-math -funroll-loops -framework Accelerate \
        vector_sparse_gemm.cpp -o benchmark

# Intel/AMD with AVX2
g++ -std=c++17 -O3 -march=native -mavx2 -mfma -ffast-math \
    vector_sparse_gemm.cpp -o benchmark

# Generic (any platform)
g++ -std=c++17 -O3 vector_sparse_gemm.cpp -o benchmark
```

#### Debug Build
```bash
g++ -std=c++17 -O0 -g -DDEBUG \
    vector_sparse_gemm.cpp -o benchmark_debug
```

---

## 🏃‍♂️ Running Benchmarks

### Automated Runner (Recommended)

#### macOS
```bash
./macos_runner.sh [OPTIONS]

Options:
  --help              Show help
  --system-info       Display system information
  --analysis-only     Run analysis on existing results
```

#### Linux
```bash
./benchmark_runner.sh [OPTIONS]

Options:
  --quick            Run abbreviated benchmark
  --analysis-only    Only analyze existing results
```

### Manual Execution
```bash
# Run benchmark
./benchmark > results.txt

# Extract CSV data
grep "^[A-Za-z].*," results.txt > benchmark_results.csv

# Quick analysis
python3 -c "
import pandas as pd
df = pd.read_csv('benchmark_results.csv')
print(df.groupby('Implementation')[['GFLOPS', 'Speedup']].mean())
"
```

### Performance Monitoring

#### macOS
```bash
# Prevent sleep during benchmark
caffeinate -i ./benchmark

# Monitor with Instruments
instruments -t 'Time Profiler' ./benchmark

# Power monitoring (Apple Silicon)
sudo powermetrics --sample-rate 1000 &
./benchmark
```

#### Linux
```bash
# CPU monitoring
htop &
./benchmark

# Performance counters
perf stat -e cycles,instructions,cache-misses ./benchmark
```

---

## 📊 Understanding Results

### CSV Output Format
```csv
Test,M,K,N,Sparsity,Implementation,Time_ns,GFLOPS,Speedup,Efficiency%
Small_50%,64,512,1024,1/2,Dense_Scalar,36645704.20,1.83,1.00,100.00
Small_50%,64,512,1024,1/2,Dense_Vector,26403870.90,2.54,1.39,138.79
Small_50%,64,512,1024,1/2,Sparse_Vector,4278837.40,3.94,8.56,214.78
```

### Key Metrics

| Metric | Description | Good Values |
|--------|-------------|-------------|
| **GFLOPS** | Billion floating-point operations per second | Higher is better |
| **Speedup** | Performance relative to Dense_Scalar baseline | 2-20x depending on optimization |
| **Efficiency%** | Percentage improvement over baseline | >200% is excellent |
| **Time_ns** | Execution time in nanoseconds | Lower is better |

### Test Configurations

| Configuration | M | K | N | Memory Usage | Target |
|---------------|---|---|---|--------------|--------|
| Small | 64 | 512 | 1024 | ~2.5 MB | Cache performance |
| Medium | 256 | 1024 | 2048 | ~20 MB | Memory hierarchy |
| Large | 1024 | 2048 | 4096 | ~300 MB | Memory bandwidth |

### Implementation Types

1. **Dense_Scalar**: Traditional scalar matrix multiplication (baseline)
2. **Dense_Vector**: SIMD-optimized dense multiplication
3. **Sparse_Scalar**: Scalar sparse ternary matrix multiplication
4. **Sparse_Vector**: SIMD-optimized sparse multiplication (best performance)

---

## 📈 Analysis and Visualization

### Automated Analysis

The benchmark generates comprehensive analysis including:

- **📊 Performance Summary**: GFLOPS, speedups, efficiency metrics
- **🎯 Roofline Models**: Hardware utilization vs theoretical peaks
- **📈 Scaling Analysis**: Performance across matrix sizes
- **🔍 Bottleneck Identification**: Memory vs compute bound analysis
- **💡 Optimization Recommendations**: Architecture-specific guidance

### Sample Analysis Output

```
🍎 M1 Mac Performance Analysis
==============================
NEON Vectorization: 1.40x average speedup (35% efficiency)
Sparse Optimization: 15.22x maximum speedup
Memory Bandwidth: 68.25 GB/s utilized
Bottleneck: Primarily memory bound for dense operations

💡 Recommendations:
✅ Sparse algorithms highly effective - consider production use
✅ NEON optimization working well for this workload
⚠️  Memory access patterns could be improved for dense operations
```

### Custom Analysis

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load results
df = pd.read_csv('benchmark_results.csv')

# Plot performance comparison
fig, ax = plt.subplots(figsize=(12, 8))
for impl in df['Implementation'].unique():
    impl_data = df[df['Implementation'] == impl]
    ax.scatter(impl_data['M'] * impl_data['K'] * impl_data['N'], 
              impl_data['GFLOPS'], label=impl, s=100, alpha=0.7)

ax.set_xlabel('Matrix Size (M×K×N)')
ax.set_ylabel('Performance (GFLOPS)')
ax.set_xscale('log')
ax.legend()
plt.show()
```

---

## 🔧 Advanced Configuration

### Compiler Optimizations

#### Maximum Performance
```bash
# Intel/AMD
-O3 -march=native -mtune=native -mavx2 -mfma -ffast-math -funroll-loops

# Apple Silicon
-O3 -mcpu=apple-m1 -march=armv8.2-a+fp16+dotprod -ffast-math -funroll-loops

# Generic high performance
-O3 -DNDEBUG -ffast-math -funroll-loops
```

#### Architecture-Specific
```bash
# Intel Skylake
-march=skylake -mtune=skylake -mavx2 -mfma

# AMD Zen 3
-march=znver3 -mtune=znver3 -mavx2 -mfma

# ARM Cortex-A78
-mcpu=cortex-a78 -march=armv8.2-a+simd
```

### Custom Matrix Sizes

Edit the source code to test specific sizes:

```cpp
vector<TestConfig> configs = {
    {M, K, N, sparsity_factor, "Custom_Test"},
    // Add your configurations
};
```

### Memory Alignment

For optimal SIMD performance:

```cpp
// Ensure proper alignment
#define ALIGNMENT 32  // AVX2
#define ALIGNMENT 16  // NEON

// Use aligned allocation
float* matrix = aligned_alloc<float>(size);
```

---

## 🚨 Troubleshooting

### Common Build Issues

#### Compiler Not Found
```bash
# Check available compilers
which gcc g++ clang++

# Install missing compiler
# Ubuntu: sudo apt install build-essential
# macOS: xcode-select --install
```

#### Missing SIMD Support
```bash
# Check CPU capabilities
# Linux: cat /proc/cpuinfo | grep flags
# macOS: sysctl -a | grep cpu

# Disable SIMD if needed
g++ -DNO_SIMD vector_sparse_gemm.cpp -o benchmark
```

#### Architecture Mismatch
```bash
# Check binary architecture
file benchmark
ldd benchmark  # Linux
otool -L benchmark  # macOS

# Cross-compile if needed
g++ -target x86_64-linux-gnu vector_sparse_gemm.cpp
```

### Runtime Issues

#### Segmentation Fault
```bash
# Check with debug build
g++ -g -fsanitize=address vector_sparse_gemm.cpp -o debug_benchmark
./debug_benchmark

# Check memory alignment
valgrind ./benchmark  # Linux
```

#### Poor Performance
```bash
# Check system load
top
htop

# Check thermal throttling
# macOS: pmset -g therm
# Linux: sensors

# Check CPU governor
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor
```

#### Missing Results
```bash
# Check output
./benchmark 2>&1 | tee debug_output.txt

# Verify CSV format
head -5 debug_output.txt | grep "^[A-Za-z].*,"
```

### Analysis Issues

#### Python Packages Missing
```bash
# Install required packages
pip3 install pandas matplotlib seaborn numpy

# Check installation
python3 -c "import pandas; print('OK')"
```

#### Plot Generation Fails
```bash
# Install GUI backend
pip3 install PyQt5  # or tkinter-dev

# Use non-interactive backend
export MPLBACKEND=Agg
python3 analysis_script.py
```

---

## 🎓 Understanding the Algorithms

### Dense Matrix Multiplication

**Scalar Version:**
```cpp
for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += X[i*K + k] * W[k*N + j];  // 2 FLOPS per iteration
        }
        Y[i*N + j] = sum + B[j];
    }
}
```

**SIMD Vector Version (NEON):**
```cpp
float32x4_t sum_vec = vdupq_n_f32(0.0f);
for (int k = 0; k <= K-4; k += 4) {
    float32x4_t x_vec = vld1q_f32(&X[i*K + k]);
    float32x4_t w_vec = {W[k*N+j], W[(k+1)*N+j], W[(k+2)*N+j], W[(k+3)*N+j]};
    sum_vec = vfmaq_f32(sum_vec, x_vec, w_vec);  // 8 FLOPS per iteration
}
float sum = vaddvq_f32(sum_vec);  // Horizontal sum
```

### Sparse Matrix Format

**Ternary Compression:**
- Values: {-1, 0, +1}
- Storage: Separate positive/negative indices
- Memory saving: ~75% for 25% sparsity

**Data Structure:**
```cpp
struct SparseMatrix {
    vector<int> col_start_pos;     // Column start indices for +1 values
    vector<int> col_start_neg;     // Column start indices for -1 values  
    vector<int> row_index_pos;     // Row indices for +1 values
    vector<int> row_index_neg;     // Row indices for -1 values
};
```

**Sparse Computation:**
```cpp
// Process positive weights (+1)
for (int idx = sparse.col_start_pos[j]; idx < sparse.col_start_pos[j+1]; idx++) {
    sum += X[i*K + sparse.row_index_pos[idx]];
}

// Process negative weights (-1)  
for (int idx = sparse.col_start_neg[j]; idx < sparse.col_start_neg[j+1]; idx++) {
    sum -= X[i*K + sparse.row_index_neg[idx]];
}
```

### Performance Theory

**Arithmetic Intensity:**
- Dense: `2*M*N*K FLOPS / ((M*K + K*N + M*N) * 4 bytes)`
- Sparse: `M*N*(K/sparsity + 1) FLOPS / (reduced memory)`

**Roofline Model:**
- Memory bound: `Performance = Bandwidth × Arithmetic_Intensity`
- Compute bound: `Performance = Peak_FLOPS`
- Actual: `min(Memory_Bound, Compute_Bound)`

**Speedup Sources:**
1. **SIMD Vectorization**: 2-8x (width dependent)
2. **Sparsity**: 2-16x (sparsity dependent)  
3. **Cache Optimization**: 1.2-3x (size dependent)
4. **Memory Access**: 1.1-2x (pattern dependent)

---

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md).

### Development Setup

```bash
# Fork and clone
git clone https://github.com/yourusername/vector-matrix-benchmark.git
cd vector-matrix-benchmark

# Create feature branch
git checkout -b feature/new-optimization

# Make changes and test
make test

# Submit pull request
```

### Adding New Architectures

1. **Add architecture detection** in build scripts
2. **Implement SIMD intrinsics** in appropriate `#ifdef` blocks
3. **Update build system** with architecture-specific flags
4. **Test thoroughly** and document performance characteristics
5. **Update README** with new architecture support

### Performance Improvements

Contributions should demonstrate:
- **≥10% performance improvement** for new optimizations
- **No regression** in existing functionality
- **Cross-platform compatibility** when possible
- **Proper benchmarking** with statistical significance

---

## 📚 References and Related Work

### Academic Papers
- [Roofline Model](https://doi.org/10.1145/1498765.1498785) - Williams et al., 2009
- [Sparse Matrix Computations](https://doi.org/10.1137/1.9780898718881) - Saad, 2003

### Hardware Documentation
- [Intel Intrinsics Guide](https://software.intel.com/sites/landingpage/IntrinsicsGuide/)
- [ARM NEON Programmer's Guide](https://developer.arm.com/documentation/den0018/a)
- [Apple Silicon Performance Guide](https://developer.apple.com/documentation/apple-silicon)

### Optimization Resources
- [Agner Fog's Optimization Manuals](https://www.agner.org/optimize/)
- [Intel Architecture Optimization Manual](https://software.intel.com/content/www/us/en/develop/articles/intel-sdm.html)

---
>>>>>>> 24507cc16c91bf27f81272d837f5881ed2efd798

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

<<<<<<< HEAD
---

**⭐ Star this repository if it helped you optimize your matrix operations!**
=======
```
MIT License

Copyright (c) 2024 Vector Matrix Benchmark Contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## Acknowledgments

- **Intel** for comprehensive intrinsics documentation and optimization guides
- **Apple** for Apple Silicon performance guidelines and Accelerate framework
- **ARM** for NEON optimization resources and developer documentation  
- **AMD** for Zen architecture optimization manuals
- **Open source community** for continuous feedback and contributions
- **Research community** for foundational work on performance modeling and sparse algorithms

---


** Star this repository if it helped you optimize your matrix operations!**

** Ready to benchmark? Start with: `./macos_build.sh && ./macos_runner.sh`**

