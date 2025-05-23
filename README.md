# 🚀 Vector-Optimized Sparse Matrix Multiplication Benchmark

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![C++17](https://img.shields.io/badge/C%2B%2B-17-blue.svg)](https://isocpp.org/std/the-standard)
[![Platform](https://img.shields.io/badge/Platform-Linux%20%7C%20macOS%20%7C%20Windows-lightgrey)](https://github.com)

**High-performance matrix multiplication benchmarking suite with SIMD optimizations for modern CPU architectures.**

## ✨ Features

- 🎯 **Multi-Architecture SIMD**: AVX2 (8-wide), NEON (4-wide), AVX-512 support
- ⚡ **Sparse Optimization**: Ternary matrix format with up to 20x speedups
- 📊 **Comprehensive Analysis**: Roofline models, performance plots, bottleneck identification
- 🍎 **Apple Silicon Ready**: Native M1/M2/M3 optimizations with Accelerate framework
- 🔧 **Cross-Platform**: Linux, macOS (Intel + Apple Silicon), Windows WSL

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

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**⭐ Star this repository if it helped you optimize your matrix operations!**
