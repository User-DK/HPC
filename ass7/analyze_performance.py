import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def analyze_performance():
    """
    Analyze MPI performance data from CSV file and create visualization plots.
    """
    # Read the CSV file
    try:
        df = pd.read_csv('performance_data.csv')
        print("Data loaded successfully!")
        print(df)
    except FileNotFoundError:
        print("Error: performance_data.csv not found!")
        return
    
    # Remove rows with missing data
    df_clean = df.dropna()
    
    if df_clean.empty:
        print("No complete data found. Please fill in the CSV with your measurements.")
        return
    
    # Create the performance analysis plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('MPI Performance Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Execution Time vs Number of Processes
    ax1.plot(df_clean['num_processes'], df_clean['matrix_vector_time'], 
             'o-', linewidth=2, markersize=8, label='Matrix-Vector Multiplication', color='blue')
    ax1.plot(df_clean['num_processes'], df_clean['matrix_matrix_time'], 
             's-', linewidth=2, markersize=8, label='Matrix-Matrix Multiplication', color='red')
    ax1.set_xlabel('Number of Processes')
    ax1.set_ylabel('Execution Time (seconds)')
    ax1.set_title('Execution Time vs Number of Processes')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')  # Log scale for better visualization
    
    # Plot 2: Speedup Analysis
    if len(df_clean) > 1:
        # Calculate speedup (T1/Tp)
        mv_speedup = df_clean['matrix_vector_time'].iloc[0] / df_clean['matrix_vector_time']
        mm_speedup = df_clean['matrix_matrix_time'].iloc[0] / df_clean['matrix_matrix_time']
        
        ax2.plot(df_clean['num_processes'], mv_speedup, 
                 'o-', linewidth=2, markersize=8, label='Matrix-Vector Speedup', color='blue')
        ax2.plot(df_clean['num_processes'], mm_speedup, 
                 's-', linewidth=2, markersize=8, label='Matrix-Matrix Speedup', color='red')
        
        # Ideal speedup line
        ideal_speedup = df_clean['num_processes']
        ax2.plot(df_clean['num_processes'], ideal_speedup, 
                 '--', linewidth=2, label='Ideal Speedup', color='green', alpha=0.7)
        
        ax2.set_xlabel('Number of Processes')
        ax2.set_ylabel('Speedup')
        ax2.set_title('Speedup vs Number of Processes')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # Plot 3: Efficiency Analysis
    if len(df_clean) > 1:
        # Calculate efficiency (Speedup/P)
        mv_efficiency = mv_speedup / df_clean['num_processes']
        mm_efficiency = mm_speedup / df_clean['num_processes']
        
        ax3.plot(df_clean['num_processes'], mv_efficiency * 100, 
                 'o-', linewidth=2, markersize=8, label='Matrix-Vector Efficiency', color='blue')
        ax3.plot(df_clean['num_processes'], mm_efficiency * 100, 
                 's-', linewidth=2, markersize=8, label='Matrix-Matrix Efficiency', color='red')
        
        ax3.set_xlabel('Number of Processes')
        ax3.set_ylabel('Efficiency (%)')
        ax3.set_title('Parallel Efficiency vs Number of Processes')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, 110)  # Efficiency should be between 0-100%
    
    # Plot 4: Performance Comparison Bar Chart
    x = np.arange(len(df_clean))
    width = 0.35
    
    ax4.bar(x - width/2, df_clean['matrix_vector_time'], width, 
            label='Matrix-Vector', color='blue', alpha=0.7)
    ax4.bar(x + width/2, df_clean['matrix_matrix_time'], width, 
            label='Matrix-Matrix', color='red', alpha=0.7)
    
    ax4.set_xlabel('Test Cases')
    ax4.set_ylabel('Execution Time (seconds)')
    ax4.set_title('Performance Comparison')
    ax4.set_xticks(x)
    ax4.set_xticklabels([f'{p} proc' for p in df_clean['num_processes']])
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('mpi_performance_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print performance summary
    print("\n" + "="*50)
    print("PERFORMANCE ANALYSIS SUMMARY")
    print("="*50)
    
    if len(df_clean) > 1:
        print(f"\nMatrix-Vector Multiplication:")
        print(f"  Best time: {df_clean['matrix_vector_time'].min():.4f}s with {df_clean.loc[df_clean['matrix_vector_time'].idxmin(), 'num_processes']} processes")
        print(f"  Max speedup: {mv_speedup.max():.2f}x")
        print(f"  Best efficiency: {(mv_efficiency.max() * 100):.1f}%")
        
        print(f"\nMatrix-Matrix Multiplication:")
        print(f"  Best time: {df_clean['matrix_matrix_time'].min():.4f}s with {df_clean.loc[df_clean['matrix_matrix_time'].idxmin(), 'num_processes']} processes")
        print(f"  Max speedup: {mm_speedup.max():.2f}x")
        print(f"  Best efficiency: {(mm_efficiency.max() * 100):.1f}%")
    
    print(f"\nTotal measurements: {len(df_clean)}")
    print("Analysis complete! Check 'mpi_performance_analysis.png' for detailed plots.")

if __name__ == "__main__":
    print("MPI Performance Analysis Tool")
    print("="*30)
    print("Make sure your performance_data.csv file is filled with measurements.")
    print("CSV format: num_processes,matrix_vector_time,matrix_matrix_time")
    print()
    
    analyze_performance()