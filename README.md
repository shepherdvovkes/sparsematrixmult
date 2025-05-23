🚀 Vector-Optimized Sparse Matrix Multiplication Benchmark
High-performance matrix multiplication benchmarking suite with SIMD optimizations for x86_64 (AVX2) and ARM64 (NEON) architectures.
✨ Features

Multi-Architecture SIMD: AVX2 (8-wide) and NEON (4-wide) vectorization
Sparse Optimization: Ternary matrix format with up to 15x speedups
Comprehensive Analysis: Roofline models, performance plots, and bottleneck identification
Cross-Platform: Linux, macOS (Apple Silicon + Intel), Windows WSL
Automated Pipeline: One-command build, benchmark, and analysis

🎯 Quick Start
bash# macOS (Apple Silicon/Intel)
./macos_build.sh && ./macos_runner.sh

# Linux
make benchmark

# Manual
g++ -O3 -march=native -mavx2 -mfma vector_sparse_gemm.cpp -o benchmark
./benchmark > results.csv
📊 Performance Results
ImplementationSpeedupUse CaseDense Vector2-4xSIMD optimizationSparse 50%2-8xModerate sparsitySparse 12.5%8-15xHigh sparsity
🏗️ Architecture Support

x86_64: Intel/AMD with AVX2/FMA
ARM64: Apple Silicon, ARM Cortex-A
Accelerate: macOS framework integration
Generic: Fallback scalar implementation

📈 Analysis Tools

Roofline Models: Hardware utilization analysis
Performance Plots: Speedup and efficiency visualization
CSV Export: Raw data for custom analysis
System Profiling: Architecture-specific optimizations

Perfect for researchers, HPC developers, and performance engineers studying matrix multiplication optimizations across modern CPU architectures.
