import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def analyze_mpi_performance():
    """
    Analyze MPI performance for dot product and convolution algorithms.
    """
    # Read the CSV file
    try:
        df = pd.read_csv('performance_data.csv')
        print("Performance data loaded successfully!")
        print(df)
    except FileNotFoundError:
        print("Error: performance_data.csv not found!")
        return
    
    # Remove rows with missing data
    df_clean = df.dropna()
    
    if df_clean.empty:
        print("No complete data found. Please fill in the CSV with your measurements.")
        print("Run your MPI programs and record the execution times.")
        return
    
    # Create comprehensive performance analysis
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('MPI Performance Analysis - Assignment 8', fontsize=16, fontweight='bold')
    
    # Plot 1: Execution Time vs Number of Processes
    ax1.plot(df_clean['num_processes'], df_clean['dot_product_time'], 
             'o-', linewidth=2, markersize=8, label='Dot Product', color='blue')
    ax1.plot(df_clean['num_processes'], df_clean['convolution_time'], 
             's-', linewidth=2, markersize=8, label='Convolution', color='red')
    ax1.set_xlabel('Number of Processes')
    ax1.set_ylabel('Execution Time (seconds)')
    ax1.set_title('Execution Time vs Number of Processes')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')  # Log scale for better visualization
    
    # Plot 2: Speedup Analysis
    if len(df_clean) > 1:
        # Calculate speedup (T1/Tp)
        dot_speedup = df_clean['dot_product_time'].iloc[0] / df_clean['dot_product_time']
        conv_speedup = df_clean['convolution_time'].iloc[0] / df_clean['convolution_time']
        
        ax2.plot(df_clean['num_processes'], dot_speedup, 
                 'o-', linewidth=2, markersize=8, label='Dot Product Speedup', color='blue')
        ax2.plot(df_clean['num_processes'], conv_speedup, 
                 's-', linewidth=2, markersize=8, label='Convolution Speedup', color='red')
        
        # Ideal speedup line
        ideal_speedup = df_clean['num_processes']
        ax2.plot(df_clean['num_processes'], ideal_speedup, 
                 '--', linewidth=2, label='Ideal Speedup', color='green', alpha=0.7)
        
        ax2.set_xlabel('Number of Processes')
        ax2.set_ylabel('Speedup')
        ax2.set_title('Speedup vs Number of Processes')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # Plot 3: Parallel Efficiency
    if len(df_clean) > 1:
        # Calculate efficiency (Speedup/P)
        dot_efficiency = dot_speedup / df_clean['num_processes']
        conv_efficiency = conv_speedup / df_clean['num_processes']
        
        ax3.plot(df_clean['num_processes'], dot_efficiency * 100, 
                 'o-', linewidth=2, markersize=8, label='Dot Product Efficiency', color='blue')
        ax3.plot(df_clean['num_processes'], conv_efficiency * 100, 
                 's-', linewidth=2, markersize=8, label='Convolution Efficiency', color='red')
        
        ax3.set_xlabel('Number of Processes')
        ax3.set_ylabel('Efficiency (%)')
        ax3.set_title('Parallel Efficiency vs Number of Processes')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, 110)  # Efficiency should be between 0-100%
    
    # Plot 4: Performance Comparison
    x = np.arange(len(df_clean))
    width = 0.35
    
    ax4.bar(x - width/2, df_clean['dot_product_time'], width, 
            label='Dot Product', color='blue', alpha=0.7)
    ax4.bar(x + width/2, df_clean['convolution_time'], width, 
            label='Convolution', color='red', alpha=0.7)
    
    ax4.set_xlabel('Test Cases')
    ax4.set_ylabel('Execution Time (seconds)')
    ax4.set_title('Algorithm Performance Comparison')
    ax4.set_xticks(x)
    ax4.set_xticklabels([f'{p} proc' for p in df_clean['num_processes']])
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('assignment8_performance_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print detailed performance summary
    print("\n" + "="*60)
    print("ASSIGNMENT 8 - MPI PERFORMANCE ANALYSIS SUMMARY")
    print("="*60)
    
    if len(df_clean) > 1:
        print(f"\n🔵 DOT PRODUCT ANALYSIS:")
        print(f"   Best time: {df_clean['dot_product_time'].min():.6f}s with {df_clean.loc[df_clean['dot_product_time'].idxmin(), 'num_processes']} processes")
        print(f"   Maximum speedup: {dot_speedup.max():.2f}x")
        print(f"   Best efficiency: {(dot_efficiency.max() * 100):.1f}%")
        
        print(f"\n🔴 CONVOLUTION ANALYSIS:")
        print(f"   Best time: {df_clean['convolution_time'].min():.6f}s with {df_clean.loc[df_clean['convolution_time'].idxmin(), 'num_processes']} processes")
        print(f"   Maximum speedup: {conv_speedup.max():.2f}x")
        print(f"   Best efficiency: {(conv_efficiency.max() * 100):.1f}%")
        
        # Performance comparison
        if df_clean['dot_product_time'].min() < df_clean['convolution_time'].min():
            faster_algo = "Dot Product"
            time_ratio = df_clean['convolution_time'].min() / df_clean['dot_product_time'].min()
        else:
            faster_algo = "Convolution"
            time_ratio = df_clean['dot_product_time'].min() / df_clean['convolution_time'].min()
        
        print(f"\n📊 ALGORITHM COMPARISON:")
        print(f"   Faster algorithm: {faster_algo}")
        print(f"   Performance ratio: {time_ratio:.2f}x faster")
    
    print(f"\n📈 MEASUREMENT SUMMARY:")
    print(f"   Total test cases: {len(df_clean)}")
    print(f"   Process count range: {df_clean['num_processes'].min()} - {df_clean['num_processes'].max()}")
    
    print("\n✅ Analysis complete! Check 'assignment8_performance_analysis.png' for detailed visualizations.")

def display_instructions():
    """Display instructions for running the MPI programs and collecting data."""
    print("="*70)
    print("MPI PERFORMANCE ANALYSIS - ASSIGNMENT 8")
    print("="*70)
    print("\n📋 INSTRUCTIONS:")
    print("1. Compile your MPI programs:")
    print("   mpicc -o dot_product_mpi dot_product_mpi.c")
    print("   mpicc -o convolution_mpi convolution_mpi.c")
    print("\n2. Run with different process counts and record times:")
    print("   mpirun -np 1 ./dot_product_mpi")
    print("   mpirun -np 2 ./dot_product_mpi")
    print("   mpirun -np 4 ./dot_product_mpi")
    print("   ... (continue for 8, 16 processes)")
    print("\n   For convolution (example with grid size 1000, kernel size 3):")
    print("   mpirun -np 1 ./convolution_mpi 1000 3")
    print("   mpirun -np 2 ./convolution_mpi 1000 3")
    print("   ... (continue for different process counts)")
    print("\n3. Fill the execution times in performance_data.csv")
    print("4. Run this script again to see the analysis")
    print("\n💡 TIP: Both programs already print execution times automatically!")

if __name__ == "__main__":
    display_instructions()
    print("\n" + "="*70)
    analyze_mpi_performance()