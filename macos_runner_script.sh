#!/bin/bash

# macOS Performance Benchmark Runner
# Advanced runner with macOS-specific performance monitoring and optimization

set -e

# Color codes for macOS Terminal
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m'

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build_macos"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RESULTS_DIR="${SCRIPT_DIR}/results_macos_${TIMESTAMP}"

# Detect architecture
ARCH=$(uname -m)
case $ARCH in
    arm64) ARCH_NAME="apple_silicon" ;;
    x86_64) ARCH_NAME="intel_mac" ;;
    *) echo "Unsupported architecture"; exit 1 ;;
esac

# Output functions
print_header() {
    echo -e "${BLUE}╔══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║ $(printf "%-60s" "$1") ║${NC}"
    echo -e "${BLUE}╚══════════════════════════════════════════════════════════════╝${NC}"
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

print_step() {
    echo -e "${MAGENTA}▶ $1${NC}"
}

# Check if build directory exists
check_build_environment() {
    print_header "Checking Build Environment"
    
    if [[ ! -d "$BUILD_DIR" ]]; then
        print_error "Build directory not found: $BUILD_DIR"
        print_info "Please run the build script first: ./macos_build.sh"
        exit 1
    fi
    
    # Find benchmark executable
    BENCHMARK_EXEC=$(find "$BUILD_DIR" -name "benchmark_${ARCH_NAME}" -type f -executable | head -1)
    
    if [[ ! -x "$BENCHMARK_EXEC" ]]; then
        print_error "Benchmark executable not found in $BUILD_DIR"
        print_info "Please run the build script first: ./macos_build.sh"
        exit 1
    fi
    
    print_success "Build environment ready"
    print_info "Benchmark executable: $BENCHMARK_EXEC"
    print_info "Architecture: $ARCH_NAME"
}

# Setup performance monitoring environment
setup_performance_monitoring() {
    print_header "Setting Up Performance Monitoring"
    
    # Create results directory
    mkdir -p "$RESULTS_DIR"
    
    # Check for performance tools
    PERF_TOOLS=()
    
    if command -v instruments &> /dev/null; then
        PERF_TOOLS+=("instruments")
        print_success "Instruments available"
    fi
    
    if command -v sample &> /dev/null; then
        PERF_TOOLS+=("sample")
        print_success "Sample tool available"
    fi
    
    if command -v powermetrics &> /dev/null && [[ $(id -u) -eq 0 ]]; then
        PERF_TOOLS+=("powermetrics")
        print_success "PowerMetrics available (requires sudo)"
    fi
    
    if command -v dtrace &> /dev/null; then
        PERF_TOOLS+=("dtrace")
        print_success "DTrace available"
    fi
    
    # System optimization
    print_step "Applying macOS performance optimizations"
    
    # Disable Spotlight indexing on results directory
    mdutil -i off "$RESULTS_DIR" 2>/dev/null || true
    
    # Set high performance mode (if supported)
    if [[ $ARCH == "arm64" ]]; then
        print_info "Apple Silicon detected - checking performance modes"
        # Note: Performance mode control requires specific entitlements
    else
        print_info "Intel Mac detected - checking thermal state"
        pmset -g therm 2>/dev/null || true
    fi
    
    print_success "Performance monitoring setup complete"
}

# Run system analysis
run_system_analysis() {
    print_header "System Performance Analysis"
    
    local analysis_file="${RESULTS_DIR}/system_analysis.txt"
    
    {
        echo "macOS System Performance Analysis"
        echo "Generated: $(date)"
        echo "Architecture: $ARCH ($ARCH_NAME)"
        echo "================================="
        echo ""
        
        echo "SYSTEM LOAD:"
        uptime
        echo ""
        
        echo "MEMORY USAGE:"
        vm_stat | head -10
        echo ""
        
        echo "CPU USAGE:"
        top -l 1 -n 5 | grep "CPU usage"
        echo ""
        
        if [[ $ARCH == "arm64" ]]; then
            echo "APPLE SILICON THERMAL STATE:"
            pmset -g thermlog 2>/dev/null | tail -5 || echo "Thermal log not available"
            echo ""
            
            echo "POWER METRICS (if available):"
            timeout 5s powermetrics --sample-rate 1000 --sample-count 5 2>/dev/null | grep -E "(CPU|GPU|ANE)" || echo "PowerMetrics requires sudo"
            echo ""
        else
            echo "INTEL CPU THERMAL:"
            pmset -g therm 2>/dev/null || echo "Thermal state not available"
            echo ""
        fi
        
        echo "DISK I/O:"
        iostat -d 1 2 | tail -5
        echo ""
        
    } > "$analysis_file"
    
    print_success "System analysis saved: $analysis_file"
}

# Run benchmark with monitoring
run_benchmark_with_monitoring() {
    print_header "Running Benchmark with Performance Monitoring"
    
    local benchmark_output="${RESULTS_DIR}/benchmark_output.txt"
    local performance_log="${RESULTS_DIR}/performance_monitor.txt"
    
    print_step "Starting benchmark execution"
    print_info "This may take 15-45 minutes depending on your system"
    print_info "Results will be saved to: $RESULTS_DIR"
    
    # Background monitoring
    if [[ " ${PERF_TOOLS[@]} " =~ " sample " ]]; then
        print_info "Starting CPU sampling in background"
        sample "$BENCHMARK_EXEC" 30 -file "${RESULTS_DIR}/cpu_sample.txt" &
        SAMPLE_PID=$!
    fi
    
    # Run benchmark with caffeinate to prevent sleep
    print_step "Executing benchmark (preventing system sleep)"
    
    if command -v caffeinate &> /dev/null; then
        # Monitor system resources during benchmark
        {
            echo "Performance monitoring started: $(date)"
            echo "Benchmark PID will be monitored"
            echo ""
            
            while true; do
                echo "=== $(date) ==="
                echo "Memory:"
                vm_stat | grep -E "(free|active|inactive|wired)"
                echo "Load:"
                uptime
                echo "Top processes:"
                top -l 1 -n 3 -o cpu | head -15 | tail -10
                echo ""
                sleep 10
            done
        } > "$performance_log" &
        MONITOR_PID=$!
        
        # Run the actual benchmark
        caffeinate -i "$BENCHMARK_EXEC" > "$benchmark_output" 2>&1
        BENCHMARK_EXIT_CODE=$?
        
        # Stop monitoring
        kill $MONITOR_PID 2>/dev/null || true
        [[ -n "${SAMPLE_PID:-}" ]] && kill $SAMPLE_PID 2>/dev/null || true
        
    else
        "$BENCHMARK_EXEC" > "$benchmark_output" 2>&1
        BENCHMARK_EXIT_CODE=$?
    fi
    
    if [[ $BENCHMARK_EXIT_CODE -eq 0 ]]; then
        print_success "Benchmark completed successfully"
    else
        print_error "Benchmark failed with exit code: $BENCHMARK_EXIT_CODE"
        return 1
    fi
    
    # Extract CSV results
    if grep -q "^[A-Za-z].*," "$benchmark_output"; then
        grep "^[A-Za-z].*," "$benchmark_output" > "${RESULTS_DIR}/benchmark_results.csv"
        local result_count=$(wc -l < "${RESULTS_DIR}/benchmark_results.csv")
        print_success "Extracted $result_count benchmark results"
    else
        print_warning "No CSV data found in benchmark output"
        return 1
    fi
}

# Advanced analysis with macOS-specific insights
run_advanced_analysis() {
    print_header "Advanced Performance Analysis"
    
    local results_csv="${RESULTS_DIR}/benchmark_results.csv"
    local analysis_report="${RESULTS_DIR}/advanced_analysis.txt"
    
    if [[ ! -f "$results_csv" ]]; then
        print_error "Results CSV not found"
        return 1
    fi
    
    # Python analysis
    if command -v python3 &> /dev/null; then
        print_step "Running Python analysis"
        
        python3 << 'PYTHON_ANALYSIS' > "$analysis_report"
import pandas as pd
import numpy as np
import sys
from pathlib import Path

def analyze_macos_performance():
    results_file = Path('${RESULTS_DIR}/benchmark_results.csv')
    
    if not results_file.exists():
        print("Results file not found")
        return
    
    df = pd.read_csv(results_file)
    
    print("="*80)
    print("macOS MATRIX MULTIPLICATION PERFORMANCE ANALYSIS")
    print("="*80)
    print(f"Generated: $(date)")
    print(f"Architecture: ${ARCH_NAME}")
    print(f"Total configurations tested: {len(df)}")
    print()
    
    # Overall performance summary
    print("PERFORMANCE SUMMARY BY IMPLEMENTATION:")
    print("-" * 50)
    
    summary_stats = df.groupby('Implementation').agg({
        'GFLOPS': ['count', 'mean', 'std', 'min', 'max'],
        'Speedup': ['mean', 'std', 'max'],
        'Time_ns': ['mean', 'std']
    }).round(3)
    
    for impl in summary_stats.index:
        gflops_mean = summary_stats.loc[impl, ('GFLOPS', 'mean')]
        gflops_std = summary_stats.loc[impl, ('GFLOPS', 'std')]
        gflops_max = summary_stats.loc[impl, ('GFLOPS', 'max')]
        speedup_mean = summary_stats.loc[impl, ('Speedup', 'mean')]
        speedup_max = summary_stats.loc[impl, ('Speedup', 'max')]
        test_count = summary_stats.loc[impl, ('GFLOPS', 'count')]
        
        print(f"{impl:20s}: {gflops_mean:8.2f}±{gflops_std:6.2f} GFLOPS "
              f"(max: {gflops_max:8.2f}), {speedup_mean:6.2f}x speedup "
              f"(max: {speedup_max:6.2f}x), {test_count:2.0f} tests")
    
    print()
    
    # Architecture-specific analysis
    if '${ARCH}' == 'arm64':
        print("APPLE SILICON SPECIFIC ANALYSIS:")
        print("-" * 35)
        print("• NEON vectorization efficiency")
        print("• Memory bandwidth optimization for unified memory")
        print("• Power efficiency considerations")
        
        # Calculate NEON efficiency (4-wide vectors)
        if 'Dense_Vector' in df['Implementation'].values:
            vector_data = df[df['Implementation'] == 'Dense_Vector']
            scalar_data = df[df['Implementation'] == 'Dense_Scalar']
            if not vector_data.empty and not scalar_data.empty:
                avg_vector_speedup = (vector_data['GFLOPS'].mean() / 
                                    scalar_data['GFLOPS'].mean())
                neon_efficiency = (avg_vector_speedup / 4.0) * 100  # 4-wide NEON
                print(f"• NEON efficiency: {neon_efficiency:.1f}% of theoretical 4x speedup")
        
    else:
        print("INTEL MAC SPECIFIC ANALYSIS:")
        print("-" * 30)
        print("• AVX2 vectorization efficiency")
        print("• Cache hierarchy optimization")
        print("• Thermal throttling considerations")
        
        # Calculate AVX2 efficiency (8-wide vectors)
        if 'Dense_Vector' in df['Implementation'].values:
            vector_data = df[df['Implementation'] == 'Dense_Vector']
            scalar_data = df[df['Implementation'] == 'Dense_Scalar']
            if not vector_data.empty and not scalar_data.empty:
                avg_vector_speedup = (vector_data['GFLOPS'].mean() / 
                                    scalar_data['GFLOPS'].mean())
                avx2_efficiency = (avg_vector_speedup / 8.0) * 100  # 8-wide AVX2
                print(f"• AVX2 efficiency: {avx2_efficiency:.1f}% of theoretical 8x speedup")
    
    print()
    
    # Sparsity analysis
    print("SPARSITY IMPACT ANALYSIS:")
    print("-" * 25)
    
    for sparsity in sorted(df['Sparsity'].unique()):
        sparse_data = df[df['Sparsity'] == sparsity]
        if not sparse_data.empty:
            sparse_scalar_perf = sparse_data[
                sparse_data['Implementation'] == 'Sparse_Scalar']['Speedup'].mean()
            sparse_vector_perf = sparse_data[
                sparse_data['Implementation'] == 'Sparse_Vector']['Speedup'].mean()
            
            print(f"Sparsity {sparsity:>4s}: Scalar {sparse_scalar_perf:5.2f}x, "
                  f"Vector {sparse_vector_perf:5.2f}x average speedup")
    
    print()
    
    # Top performers
    print("TOP PERFORMING CONFIGURATIONS:")
    print("-" * 35)
    
    top_5 = df.nlargest(5, 'GFLOPS')[['Test', 'Implementation', 'GFLOPS', 'Speedup']]
    for i, (_, row) in enumerate(top_5.iterrows(), 1):
        print(f"{i}. {row['Implementation']:15s} on {row['Test']:12s}: "
              f"{row['GFLOPS']:8.2f} GFLOPS ({row['Speedup']:5.2f}x speedup)")
    
    print()
    
    # Performance recommendations
    print("macOS OPTIMIZATION RECOMMENDATIONS:")
    print("-" * 40)
    
    # Vectorization recommendations
    dense_vector_avg = df[df['Implementation'] == 'Dense_Vector']['Speedup'].mean()
    if dense_vector_avg > 3.0:
        print("✓ Excellent vectorization performance - consider more aggressive optimizations")
    elif dense_vector_avg > 2.0:
        print("✓ Good vectorization performance - well optimized for this architecture")
    else:
        print("⚠ Vectorization performance below expectations - check memory alignment")
    
    # Memory recommendations
    large_matrix_perf = df[df['Test'].str.contains('Large')]['GFLOPS'].mean()
    small_matrix_perf = df[df['Test'].str.contains('Small')]['GFLOPS'].mean()
    
    if large_matrix_perf > small_matrix_perf * 0.8:
        print("✓ Good scaling to large matrices - memory subsystem well utilized")
    else:
        print("⚠ Large matrix performance drops significantly - memory bandwidth limited")
    
    # Architecture-specific recommendations
    if '${ARCH}' == 'arm64':
        print("• Consider using Accelerate framework for production workloads")
        print("• Apple Silicon unified memory allows larger working sets")
        print("• Monitor power consumption with powermetrics for mobile use")
    else:
        print("• Consider Intel MKL for production workloads")
        print("• Monitor thermal throttling on intensive workloads")
        print("• Optimize for specific Intel microarchitecture if known")
    
    print()
    print("="*80)

try:
    analyze_macos_performance()
except Exception as e:
    print(f"Analysis error: {e}")
    sys.exit(1)
PYTHON_ANALYSIS
        
        if [[ $? -eq 0 ]]; then
            print_success "Advanced analysis completed"
        else
            print_warning "Advanced analysis had issues"
        fi
    else
        print_warning "Python not available - skipping advanced analysis"
    fi
}

# Generate macOS-specific performance report
generate_macos_report() {
    print_header "Generating macOS Performance Report"
    
    local report_file="${RESULTS_DIR}/macos_performance_report.html"
    
    cat > "$report_file" << 'HTML_REPORT'
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>macOS Matrix Multiplication Benchmark Report</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 40px; background: #f5f5f7; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 40px; border-radius: 12px; box-shadow: 0 4px 20px rgba(0,0,0,0.1); }
        h1 { color: #1d1d1f; border-bottom: 3px solid #007AFF; padding-bottom: 10px; }
        h2 { color: #424245; margin-top: 30px; }
        .metric-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin: 20px 0; }
        .metric-card { background: #f8f9fa; padding: 20px; border-radius: 8px; border-left: 4px solid #007AFF; }
        .metric-value { font-size: 2em; font-weight: bold; color: #007AFF; }
        .metric-label { color: #666; font-size: 0.9em; }
        .recommendations { background: #e8f5e8; padding: 20px; border-radius: 8px; margin: 20px 0; }
        .warning { background: #fff3cd; padding: 15px; border-radius: 8px; border-left: 4px solid #ffc107; margin: 10px 0; }
        .success { background: #d1edff; padding: 15px; border-radius: 8px; border-left: 4px solid #007AFF; margin: 10px 0; }
        table { width: 100%; border-collapse: collapse; margin: 20px 0; }
        th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background: #f8f9fa; font-weight: 600; }
        .arch-badge { display: inline-block; padding: 4px 12px; background: #007AFF; color: white; border-radius: 20px; font-size: 0.8em; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🚀 macOS Matrix Multiplication Benchmark Report</h1>
        
        <div class="success">
            <strong>System:</strong> ${ARCH_NAME} <span class="arch-badge">${ARCH}</span><br>
            <strong>Generated:</strong> $(date)<br>
            <strong>Results Directory:</strong> ${RESULTS_DIR}
        </div>
        
        <h2>📊 Performance Metrics</h2>
        <div class="metric-grid">
            <div class="metric-card">
                <div class="metric-value" id="max-gflops">--</div>
                <div class="metric-label">Peak GFLOPS</div>
            </div>
            <div class="metric-card">
                <div class="metric-value" id="max-speedup">--</div>
                <div class="metric-label">Maximum Speedup</div>
            </div>
            <div class="metric-card">
                <div class="metric-value" id="vector-efficiency">--</div>
                <div class="metric-label">Vector Efficiency</div>
            </div>
            <div class="metric-card">
                <div class="metric-value" id="total-tests">--</div>
                <div class="metric-label">Total Tests</div>
            </div>
        </div>
        
        <h2>🏆 Top Performers</h2>
        <table id="top-performers">
            <thead>
                <tr><th>Rank</th><th>Implementation</th><th>Test Configuration</th><th>GFLOPS</th><th>Speedup</th></tr>
            </thead>
            <tbody></tbody>
        </table>
        
        <h2>🔧 Architecture-Specific Insights</h2>
        <div id="arch-insights"></div>
        
        <h2>💡 Optimization Recommendations</h2>
        <div class="recommendations" id="recommendations"></div>
        
        <h2>📈 Detailed Results</h2>
        <p>For detailed CSV results and analysis, see:</p>
        <ul>
            <li><code>benchmark_results.csv</code> - Raw performance data</li>
            <li><code>advanced_analysis.txt</code> - Detailed statistical analysis</li>
            <li><code>system_analysis.txt</code> - System performance metrics</li>
        </ul>
        
        <script>
            // This would normally load and display the actual data
            // For now, it's a template that could be populated by the analysis script
            document.getElementById('total-tests').textContent = 'Generated';
            document.getElementById('max-gflops').textContent = 'See CSV';
            document.getElementById('max-speedup').textContent = 'See Analysis';
            
            const archInsights = document.getElementById('arch-insights');
            if ('${ARCH}' === 'arm64') {
                archInsights.innerHTML = '<div class="success"><strong>Apple Silicon Optimizations:</strong><ul><li>NEON 4-wide vectorization</li><li>Unified memory architecture</li><li>Power-efficient compute</li></ul></div>';
            } else {
                archInsights.innerHTML = '<div class="success"><strong>Intel Mac Optimizations:</strong><ul><li>AVX2 8-wide vectorization</li><li>Traditional memory hierarchy</li><li>High-performance compute focus</li></ul></div>';
            }
        </script>
    </div>
</body>
</html>
HTML_REPORT
    
    print_success "HTML report generated: $report_file"
    print_info "Open with: open $report_file"
}

# Create archive with macOS metadata
create_macos_archive() {
    print_header "Creating Results Archive"
    
    local archive_name="benchmark_results_${ARCH_NAME}_${TIMESTAMP}.tar.gz"
    
    # Add macOS extended attributes and metadata
    xattr -w com.apple.metadata:kMDItemComment "Matrix Multiplication Benchmark Results" "$RESULTS_DIR" 2>/dev/null || true
    
    # Create archive
    tar -czf "$archive_name" -C "$(dirname "$RESULTS_DIR")" "$(basename "$RESULTS_DIR")"
    
    print_success "Archive created: $archive_name"
    
    # Create quick access alias (macOS)
    if command -v osascript &> /dev/null; then
        osascript -e "tell application \"Finder\" to make alias file to POSIX file \"$(pwd)/$RESULTS_DIR\" at POSIX file \"$(pwd)\"" 2>/dev/null || true
    fi
}

# Main execution function
main() {
    print_header "macOS Matrix Multiplication Benchmark Runner"
    
    check_build_environment
    setup_performance_monitoring
    run_system_analysis
    run_benchmark_with_monitoring
    run_advanced_analysis
    generate_macos_report
    create_macos_archive
    
    print_header "Benchmark Complete! 🎉"
    
    echo -e "${GREEN}"
    echo "Results Summary:"
    echo "━━━━━━━━━━━━━━━"
    echo -e "${NC}"
    echo "📁 Results Directory: $RESULTS_DIR"
    echo "📊 CSV Data: benchmark_results.csv"
    echo "📋 Analysis Report: advanced_analysis.txt"
    echo "🌐 HTML Report: macos_performance_report.html"
    echo "🔍 System Analysis: system_analysis.txt"
    echo ""
    echo -e "${CYAN}To view results:${NC}"
    echo "• CSV: open $RESULTS_DIR/benchmark_results.csv"
    echo "• Report: open $RESULTS_DIR/macos_performance_report.html"
    echo "• Directory: open $RESULTS_DIR"
    echo ""
    echo -e "${YELLOW}Next steps:${NC}"
    echo "• Analyze results for your specific use case"
    echo "• Compare with other systems/architectures"
    echo "• Optimize code based on recommendations"
}

# Handle command line options
case "${1:-}" in
    --help|-h)
        echo "macOS Matrix Multiplication Benchmark Runner"
        echo ""
        echo "Usage: $0 [OPTIONS]"
        echo ""
        echo "Options:"
        echo "  --help, -h          Show this help"
        echo "  --quick             Run abbreviated benchmark"
        echo "  --analysis-only     Only run analysis on existing results"
        echo "  --system-info       Show system information and exit"
        echo ""
        echo "The runner will:"
        echo "  1. Check build environment"
        echo "  2. Setup performance monitoring"
        echo "  3. Run comprehensive benchmarks"
        echo "  4. Generate detailed analysis"
        echo "  5. Create HTML and text reports"
        echo "  6. Archive all results"
        echo ""
        echo "Supports both Apple Silicon and Intel Macs"
        exit 0
        ;;
    --system-info)
        print_header "macOS System Information"
        echo "Architecture: $ARCH ($ARCH_NAME)"
        echo "macOS Version: $(sw_vers -productVersion)"
        echo "Build: $(sw_vers -buildVersion)"
        if [[ $ARCH == "arm64" ]]; then
            echo "Chip: $(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo 'Apple Silicon')"
        else
            echo "CPU: $(sysctl -n machdep.cpu.brand_string)"
        fi
        echo "Memory: $(($(sysctl -n hw.memsize) / 1024 / 1024 / 1024)) GB"
        exit 0
        ;;
    --quick)
        print_warning "Quick mode not yet implemented - running full benchmark"
        main
        ;;
    --analysis-only)
        if [[ ! -d "${RESULTS_DIR}" ]]; then
            print_error "No existing results found"
            exit 1
        fi
        print_header "Analysis Only Mode"
        run_advanced_analysis
        generate_macos_report
        ;;
    *)
        main
        ;;
esac