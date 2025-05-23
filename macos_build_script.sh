#!/bin/bash

# macOS Build Script for Vector-Optimized Sparse Matrix Multiplication
# Supports both Apple Silicon (M1/M2/M3) and Intel Macs

set -e

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build_macos"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Functions for colored output
print_header() {
    echo -e "${BLUE}=== $1 ===${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_info() {
    echo -e "${CYAN}ℹ $1${NC}"
}

# Detect macOS architecture and setup
detect_macos_environment() {
    print_header "Detecting macOS Environment"
    
    # Get macOS version
    MACOS_VERSION=$(sw_vers -productVersion)
    MACOS_BUILD=$(sw_vers -buildVersion)
    
    # Detect architecture
    ARCH=$(uname -m)
    case $ARCH in
        arm64)
            ARCH_NAME="apple_silicon"
            SIMD_TYPE="NEON"
            SIMD_WIDTH=4
            print_success "Detected Apple Silicon (ARM64)"
            ;;
        x86_64)
            ARCH_NAME="intel_mac"
            SIMD_TYPE="AVX2"
            SIMD_WIDTH=8
            print_success "Detected Intel Mac (x86_64)"
            ;;
        *)
            print_error "Unsupported architecture: $ARCH"
            exit 1
            ;;
    esac
    
    print_info "macOS Version: $MACOS_VERSION ($MACOS_BUILD)"
    print_info "Architecture: $ARCH_NAME"
    print_info "SIMD Type: $SIMD_TYPE (Width: $SIMD_WIDTH)"
    
    # Detect CPU details
    if [[ $ARCH == "arm64" ]]; then
        # Apple Silicon detection
        CPU_BRAND=$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo "Apple Silicon")
        CPU_CORES=$(sysctl -n hw.perflevel0.physicalcpu 2>/dev/null || sysctl -n hw.physicalcpu)
        CPU_THREADS=$(sysctl -n hw.perflevel0.logicalcpu 2>/dev/null || sysctl -n hw.logicalcpu)
        
        # Check for specific Apple chips
        if sysctl hw.optional.arm.FEAT_DotProd 2>/dev/null | grep -q 1; then
            APPLE_FEATURES="DotProd"
        fi
        if sysctl hw.optional.arm.FEAT_FP16 2>/dev/null | grep -q 1; then
            APPLE_FEATURES="$APPLE_FEATURES FP16"
        fi
        
        print_info "CPU: $CPU_BRAND"
        print_info "Cores: $CPU_CORES physical, $CPU_THREADS logical"
        [[ -n "$APPLE_FEATURES" ]] && print_info "Features: $APPLE_FEATURES"
        
    else
        # Intel Mac detection
        CPU_BRAND=$(sysctl -n machdep.cpu.brand_string)
        CPU_CORES=$(sysctl -n hw.physicalcpu)
        CPU_THREADS=$(sysctl -n hw.logicalcpu)
        CPU_FEATURES=$(sysctl -n machdep.cpu.features machdep.cpu.leaf7_features 2>/dev/null | tr '\n' ' ')
        
        print_info "CPU: $CPU_BRAND"
        print_info "Cores: $CPU_CORES physical, $CPU_THREADS logical"
        print_info "Features: $CPU_FEATURES"
    fi
    
    # Memory info
    MEMORY_GB=$(($(sysctl -n hw.memsize) / 1024 / 1024 / 1024))
    print_info "Memory: ${MEMORY_GB} GB"
}

# Check and install prerequisites
check_prerequisites() {
    print_header "Checking Prerequisites"
    
    # Check Xcode Command Line Tools
    if ! xcode-select -p &> /dev/null; then
        print_warning "Xcode Command Line Tools not found"
        print_info "Installing Xcode Command Line Tools..."
        xcode-select --install
        print_info "Please complete the installation and re-run this script"
        exit 1
    else
        print_success "Xcode Command Line Tools: $(xcode-select -p)"
    fi
    
    # Check compiler
    if command -v clang++ &> /dev/null; then
        CLANG_VERSION=$(clang++ --version | head -1)
        print_success "Clang++: $CLANG_VERSION"
        COMPILER="clang++"
    elif command -v g++ &> /dev/null; then
        GCC_VERSION=$(g++ --version | head -1)
        print_success "GCC: $GCC_VERSION"
        COMPILER="g++"
    else
        print_error "No suitable C++ compiler found"
        exit 1
    fi
    
    # Check Homebrew (optional but recommended)
    if command -v brew &> /dev/null; then
        BREW_VERSION=$(brew --version | head -1)
        print_success "Homebrew: $BREW_VERSION"
        HOMEBREW_AVAILABLE=true
        
        # Check for performance tools
        if brew list --formula | grep -q "^google-benchmark$"; then
            print_success "Google Benchmark available"
        fi
        
    else
        print_warning "Homebrew not found (optional)"
        HOMEBREW_AVAILABLE=false
    fi
    
    # Check Python for analysis
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version)
        print_success "Python: $PYTHON_VERSION"
        
        # Check required packages
        python3 -c "import pandas, matplotlib, seaborn, numpy" 2>/dev/null && {
            print_success "Python analysis packages available"
        } || {
            print_warning "Some Python packages missing"
            print_info "Installing Python packages..."
            python3 -m pip install --user pandas matplotlib seaborn numpy
        }
    else
        print_warning "Python3 not found - analysis features will be limited"
    fi
}

# Setup build environment
setup_build_environment() {
    print_header "Setting Up Build Environment"
    
    # Create build directory
    mkdir -p "$BUILD_DIR"
    print_success "Created build directory: $BUILD_DIR"
    
    # Architecture-specific compiler flags
    COMMON_FLAGS="-std=c++17 -O3 -DNDEBUG -Wall -Wextra -Wno-unused-parameter"
    
    if [[ $ARCH == "arm64" ]]; then
        # Apple Silicon optimizations
        ARCH_FLAGS="-mcpu=apple-m1 -mtune=apple-m1"
        SIMD_FLAGS="-DARCH_AARCH64"
        
        # Check for specific Apple Silicon generation
        if sysctl hw.optional.arm.FEAT_DotProd 2>/dev/null | grep -q 1; then
            ARCH_FLAGS="$ARCH_FLAGS -march=armv8.2-a+fp16+dotprod"
        else
            ARCH_FLAGS="$ARCH_FLAGS -march=armv8-a+simd"
        fi
        
        # Apple Silicon specific optimizations
        OPTIMIZATION_FLAGS="-ffast-math -funroll-loops -fvectorize"
        
    else
        # Intel Mac optimizations
        ARCH_FLAGS="-march=native -mtune=native"
        SIMD_FLAGS="-mavx2 -mfma -DARCH_X86_64"
        OPTIMIZATION_FLAGS="-ffast-math -funroll-loops"
        
        # Check for specific Intel features
        if sysctl machdep.cpu.features | grep -q AVX512; then
            SIMD_FLAGS="$SIMD_FLAGS -mavx512f -mavx512dq"
            print_info "AVX-512 support detected"
        fi
    fi
    
    # Combine all flags
    CXXFLAGS="$COMMON_FLAGS $ARCH_FLAGS $SIMD_FLAGS $OPTIMIZATION_FLAGS"
    
    # macOS specific flags
    MACOS_FLAGS="-stdlib=libc++ -framework Accelerate"
    
    # Final compiler command
    COMPILE_CMD="$COMPILER $CXXFLAGS $MACOS_FLAGS"
    
    print_info "Compiler: $COMPILER"
    print_info "Architecture flags: $ARCH_FLAGS"
    print_info "SIMD flags: $SIMD_FLAGS"
    print_info "Optimization flags: $OPTIMIZATION_FLAGS"
    print_info "macOS flags: $MACOS_FLAGS"
}

# Build the benchmark with macOS-specific optimizations
build_matrix_benchmark() {
    print_header "Building Matrix Multiplication Benchmark"
    
    local source_file="vector_sparse_gemm.cpp"
    local target_file="${BUILD_DIR}/benchmark_${ARCH_NAME}"
    
    if [[ ! -f "$source_file" ]]; then
        print_error "Source file not found: $source_file"
        exit 1
    fi
    
    print_info "Compiling with optimization level: -O3"
    print_info "Target: $target_file"
    
    # Create macOS-optimized version of the source
    local macos_source="${BUILD_DIR}/vector_sparse_gemm_macos.cpp"
    
    # Add macOS-specific optimizations to the source
    cat > "$macos_source" << 'EOF'
// macOS-specific optimizations and includes
#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#include <mach/mach_time.h>
#include <sys/sysctl.h>

// Use mach_absolute_time for high-resolution timing on macOS
static mach_timebase_info_data_t timebase_info;
static bool timebase_initialized = false;

double get_time_ns() {
    if (!timebase_initialized) {
        mach_timebase_info(&timebase_info);
        timebase_initialized = true;
    }
    uint64_t time = mach_absolute_time();
    return (double)time * timebase_info.numer / timebase_info.denom;
}

// macOS-specific memory allocation
void* aligned_alloc_macos(size_t alignment, size_t size) {
    void* ptr = nullptr;
    if (posix_memalign(&ptr, alignment, size) == 0) {
        return ptr;
    }
    return nullptr;
}

#define aligned_alloc aligned_alloc_macos
#endif

EOF
    
    # Append the original source
    cat "$source_file" >> "$macos_source"
    
    # Compile
    print_info "Executing: $COMPILE_CMD"
    if $COMPILE_CMD -o "$target_file" "$macos_source" 2>&1 | tee "${BUILD_DIR}/compile.log"; then
        print_success "Compilation successful"
        
        # Verify the binary
        if [[ -x "$target_file" ]]; then
            print_success "Executable created: $target_file"
            
            # Get binary info
            file "$target_file"
            
            # Check architecture
            if [[ $ARCH == "arm64" ]]; then
                if lipo -info "$target_file" 2>/dev/null | grep -q "arm64"; then
                    print_success "ARM64 binary verified"
                fi
            else
                if lipo -info "$target_file" 2>/dev/null | grep -q "x86_64"; then
                    print_success "x86_64 binary verified"
                fi
            fi
            
        else
            print_error "Executable not created or not executable"
            exit 1
        fi
    else
        print_error "Compilation failed"
        print_info "Check compile log: ${BUILD_DIR}/compile.log"
        exit 1
    fi
}

# Build with Apple's Accelerate framework integration
build_accelerate_benchmark() {
    print_header "Building Accelerate Framework Benchmark"
    
    local source_file="vector_sparse_gemm.cpp"
    local target_file="${BUILD_DIR}/benchmark_accelerate_${ARCH_NAME}"
    
    # Create Accelerate-optimized version
    local accelerate_source="${BUILD_DIR}/vector_sparse_gemm_accelerate.cpp"
    
    cat > "$accelerate_source" << 'EOF'
// macOS Accelerate Framework Integration
#ifdef __APPLE__
#include <Accelerate/Accelerate.h>

// Use vDSP for optimized vector operations
void dense_gemm_accelerate(const float* X, const float* W, const float* B, float* Y,
                          int M, int N, int K) {
    // Initialize with bias using vDSP
    for (int i = 0; i < M; i++) {
        cblas_scopy(N, B, 1, &Y[i * N], 1);
    }
    
    // Use BLAS sgemm for dense matrix multiplication
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                M, N, K, 1.0f, X, K, W, N, 1.0f, Y, N);
}

// Accelerate-optimized vector operations
void vector_add_accelerate(const float* a, const float* b, float* result, int n) {
    vDSP_vadd(a, 1, b, 1, result, 1, n);
}

void vector_multiply_accelerate(const float* a, const float* b, float* result, int n) {
    vDSP_vmul(a, 1, b, 1, result, 1, n);
}

#endif

EOF
    
    # Append the original source
    cat "$source_file" >> "$accelerate_source"
    
    # Compile with Accelerate framework
    local accelerate_flags="$CXXFLAGS -framework Accelerate -DUSE_ACCELERATE"
    
    print_info "Building with Accelerate framework"
    if $COMPILER $accelerate_flags -o "$target_file" "$accelerate_source" 2>&1 | tee "${BUILD_DIR}/compile_accelerate.log"; then
        print_success "Accelerate benchmark built: $target_file"
    else
        print_warning "Accelerate benchmark build failed (this is optional)"
    fi
}

# Create performance comparison builds
build_performance_variants() {
    print_header "Building Performance Variants"
    
    local source_file="${BUILD_DIR}/vector_sparse_gemm_macos.cpp"
    
    # Build variants with different optimization levels
    local variants=("O1" "O2" "O3" "Ofast")
    
    for variant in "${variants[@]}"; do
        local opt_flag
        case $variant in
            "O1") opt_flag="-O1" ;;
            "O2") opt_flag="-O2" ;;
            "O3") opt_flag="-O3" ;;
            "Ofast") opt_flag="-Ofast -ffast-math" ;;
        esac
        
        local target_file="${BUILD_DIR}/benchmark_${variant}_${ARCH_NAME}"
        local variant_flags="$COMMON_FLAGS $ARCH_FLAGS $SIMD_FLAGS $opt_flag $MACOS_FLAGS"
        
        print_info "Building $variant variant..."
        if $COMPILER $variant_flags -o "$target_file" "$source_file" 2>/dev/null; then
            print_success "$variant variant built"
        else
            print_warning "$variant variant build failed"
        fi
    done
}

# Generate macOS-specific system report
generate_system_report() {
    print_header "Generating System Report"
    
    local report_file="${BUILD_DIR}/macos_system_report.txt"
    
    {
        echo "macOS System Report - $(date)"
        echo "=================================="
        echo ""
        
        echo "SYSTEM INFORMATION:"
        echo "  macOS Version: $(sw_vers -productVersion) ($(sw_vers -buildVersion))"
        echo "  Architecture: $ARCH ($ARCH_NAME)"
        echo "  Hostname: $(hostname)"
        echo "  Kernel: $(uname -r)"
        echo ""
        
        echo "HARDWARE INFORMATION:"
        if [[ $ARCH == "arm64" ]]; then
            echo "  Chip: $(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo 'Apple Silicon')"
            echo "  Performance Cores: $(sysctl -n hw.perflevel0.physicalcpu 2>/dev/null || echo 'N/A')"
            echo "  Efficiency Cores: $(sysctl -n hw.perflevel1.physicalcpu 2>/dev/null || echo 'N/A')"
            echo "  Total Logical CPUs: $(sysctl -n hw.logicalcpu)"
            
            # Apple Silicon specific features
            echo "  ARM Features:"
            sysctl hw.optional.arm 2>/dev/null | grep -E "(FEAT_|AdvSIMD)" | sed 's/^/    /'
            
        else
            echo "  CPU: $(sysctl -n machdep.cpu.brand_string)"
            echo "  Physical CPUs: $(sysctl -n hw.physicalcpu)"
            echo "  Logical CPUs: $(sysctl -n hw.logicalcpu)"
            echo "  CPU Frequency: $(sysctl -n hw.cpufrequency_max 2>/dev/null || echo 'N/A') Hz"
            echo "  CPU Features: $(sysctl -n machdep.cpu.features)"
            echo "  CPU Extended Features: $(sysctl -n machdep.cpu.leaf7_features 2>/dev/null || echo 'N/A')"
        fi
        
        echo "  Memory: $(($MEMORY_GB)) GB"
        echo "  Cache Line Size: $(sysctl -n hw.cachelinesize) bytes"
        echo "  L1 Cache: $(sysctl -n hw.l1icachesize) bytes (instruction), $(sysctl -n hw.l1dcachesize) bytes (data)"
        echo "  L2 Cache: $(sysctl -n hw.l2cachesize) bytes"
        echo "  L3 Cache: $(sysctl -n hw.l3cachesize 2>/dev/null || echo 'N/A') bytes"
        echo ""
        
        echo "COMPILER INFORMATION:"
        echo "  Compiler: $COMPILER"
        $COMPILER --version | head -3 | sed 's/^/  /'
        echo "  Xcode Tools: $(xcode-select -p)"
        if command -v xcodebuild &> /dev/null; then
            echo "  Xcode Version: $(xcodebuild -version | head -1)"
        fi
        echo ""
        
        echo "BUILD CONFIGURATION:"
        echo "  Architecture: $ARCH_NAME"
        echo "  SIMD Type: $SIMD_TYPE (Width: $SIMD_WIDTH)"
        echo "  Compile Flags: $CXXFLAGS"
        echo "  macOS Flags: $MACOS_FLAGS"
        echo ""
        
        if [[ $HOMEBREW_AVAILABLE == true ]]; then
            echo "HOMEBREW PACKAGES (relevant):"
            brew list --formula | grep -E "(gcc|llvm|python|numpy)" | sed 's/^/  /' || echo "  None found"
            echo ""
        fi
        
        echo "PERFORMANCE TOOLS:"
        command -v instruments &> /dev/null && echo "  Instruments: Available" || echo "  Instruments: Not available"
        command -v dtrace &> /dev/null && echo "  DTrace: Available" || echo "  DTrace: Not available"
        command -v sample &> /dev/null && echo "  Sample: Available" || echo "  Sample: Not available"
        
    } > "$report_file"
    
    print_success "System report generated: $report_file"
}

# Create run script
create_run_script() {
    print_header "Creating Run Script"
    
    local run_script="${BUILD_DIR}/run_benchmark.sh"
    
    cat > "$run_script" << EOF
#!/bin/bash
# macOS Benchmark Runner
# Generated on $(date)

set -e

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_header() {
    echo -e "\${BLUE}=== \$1 ===\${NC}"
}

print_success() {
    echo -e "\${GREEN}✓ \$1\${NC}"
}

print_info() {
    echo -e "\${BLUE}ℹ \$1\${NC}"
}

# Configuration
BUILD_DIR="\$(cd "\$(dirname "\${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="\${BUILD_DIR}/results_$(date +%Y%m%d_%H%M%S)"
BENCHMARK_EXEC="\${BUILD_DIR}/benchmark_${ARCH_NAME}"

# Create results directory
mkdir -p "\$RESULTS_DIR"

print_header "macOS Matrix Multiplication Benchmark"
print_info "Architecture: $ARCH_NAME"
print_info "SIMD: $SIMD_TYPE (Width: $SIMD_WIDTH)"
print_info "Results: \$RESULTS_DIR"

# Check if benchmark exists
if [[ ! -x "\$BENCHMARK_EXEC" ]]; then
    echo -e "\${RED}✗ Benchmark executable not found: \$BENCHMARK_EXEC\${NC}"
    echo "Please run the build script first."
    exit 1
fi

# Run benchmark
print_header "Running Benchmark"
print_info "This may take 10-30 minutes..."

# Run with performance monitoring on macOS
if command -v caffeinate &> /dev/null; then
    print_info "Using caffeinate to prevent sleep during benchmark"
    caffeinate -i "\$BENCHMARK_EXEC" > "\$RESULTS_DIR/benchmark_output.txt" 2>&1
else
    "\$BENCHMARK_EXEC" > "\$RESULTS_DIR/benchmark_output.txt" 2>&1
fi

# Extract CSV data
if grep -q "^[A-Za-z].*," "\$RESULTS_DIR/benchmark_output.txt"; then
    grep "^[A-Za-z].*," "\$RESULTS_DIR/benchmark_output.txt" > "\$RESULTS_DIR/benchmark_results.csv"
    print_success "Results saved to: \$RESULTS_DIR/benchmark_results.csv"
else
    echo -e "\${RED}✗ No benchmark results found\${NC}"
    exit 1
fi

# Quick analysis if Python is available
if command -v python3 &> /dev/null; then
    print_header "Quick Analysis"
    python3 << 'PYTHON_EOF'
import pandas as pd
import sys

try:
    df = pd.read_csv('$RESULTS_DIR/benchmark_results.csv')
    print("\\nBenchmark Results Summary:")
    print("=" * 40)
    
    summary = df.groupby('Implementation').agg({
        'GFLOPS': ['mean', 'max', 'std'],
        'Speedup': ['mean', 'max']
    }).round(2)
    
    for impl in summary.index:
        gflops_mean = summary.loc[impl, ('GFLOPS', 'mean')]
        gflops_max = summary.loc[impl, ('GFLOPS', 'max')]
        speedup_mean = summary.loc[impl, ('Speedup', 'mean')]
        speedup_max = summary.loc[impl, ('Speedup', 'max')]
        print(f"{impl:15s}: {gflops_mean:7.2f} GFLOPS avg ({gflops_max:7.2f} max), {speedup_mean:5.2f}x speedup avg")
    
    print(f"\\nBest Overall Performance:")
    best_row = df.loc[df['GFLOPS'].idxmax()]
    print(f"  {best_row['Implementation']} on {best_row['Test']}: {best_row['GFLOPS']:.2f} GFLOPS")
    
    print(f"\\nTotal test configurations: {len(df)}")

except Exception as e:
    print(f"Error in analysis: {e}")

PYTHON_EOF
fi

print_header "Benchmark Complete"
print_success "Results directory: \$RESULTS_DIR"
print_info "View results: cat \$RESULTS_DIR/benchmark_results.csv"
print_info "Full output: cat \$RESULTS_DIR/benchmark_output.txt"

# Optional: Open results in default CSV viewer
if command -v open &> /dev/null && [[ "\${1:-}" == "--open" ]]; then
    open "\$RESULTS_DIR"
fi
EOF
    
    chmod +x "$run_script"
    print_success "Run script created: $run_script"
}

# Main build function
main() {
    print_header "macOS Vector Matrix Multiplication Build System"
    
    detect_macos_environment
    check_prerequisites
    setup_build_environment
    build_matrix_benchmark
    
    # Optional builds
    if [[ "${1:-}" == "--all" ]]; then
        build_accelerate_benchmark
        build_performance_variants
    fi
    
    generate_system_report
    create_run_script
    
    print_header "Build Summary"
    print_success "Architecture: $ARCH_NAME ($SIMD_TYPE)"
    print_success "Build directory: $BUILD_DIR"
    print_success "Main benchmark: ${BUILD_DIR}/benchmark_${ARCH_NAME}"
    print_success "Run script: ${BUILD_DIR}/run_benchmark.sh"
    print_success "System report: ${BUILD_DIR}/macos_system_report.txt"
    
    echo ""
    print_info "To run benchmark:"
    print_info "  cd $BUILD_DIR && ./run_benchmark.sh"
    print_info ""
    print_info "To run with results viewer:"
    print_info "  cd $BUILD_DIR && ./run_benchmark.sh --open"
}

# Handle command line arguments
case "${1:-}" in
    --help|-h)
        echo "macOS Build Script for Vector Matrix Multiplication"
        echo ""
        echo "Usage: $0 [OPTIONS]"
        echo ""
        echo "Options:"
        echo "  --help, -h    Show this help"
        echo "  --all         Build all variants (including Accelerate framework)"
        echo "  --clean       Clean build directory"
        echo ""
        echo "The script will:"
        echo "  1. Detect macOS environment (Apple Silicon vs Intel)"
        echo "  2. Check prerequisites and setup environment"
        echo "  3. Build optimized benchmark with native flags"
        echo "  4. Generate system report"
        echo "  5. Create run script"
        echo ""
        exit 0
        ;;
    --clean)
        print_header "Cleaning Build Directory"
        rm -rf "$BUILD_DIR"
        print_success "Build directory cleaned"
        exit 0
        ;;
    *)
        main "$@"
        ;;
esac