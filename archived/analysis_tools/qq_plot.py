import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import os
import numpy as np

# 1. Load the data
df = pd.read_csv("galton_ball_positions.csv")

# 2. Robust Filtering:
# Convert the 'final_x' column to numeric, turning any 'nan' strings or invalid 
# entries into actual floating point NaN objects.
df['final_x'] = pd.to_numeric(df['final_x'], errors='coerce')

# Drop any rows where final_x is NaN (this removes your "failed_or_out_of_bounds" entries)
df_clean = df.dropna(subset=['final_x'])

# Check if we have enough data left
if len(df_clean) < 30:
    print(f"Error: Only {len(df_clean)} valid points remaining after filtering. Check simulation bounds.")
else:
    data = df_clean['final_x']

    # 3. Quantify Normality
    skewness = stats.skew(data)
    kurtosis = stats.kurtosis(data)

    # 4. Create the Q-Q Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    (osm, osr), (slope, intercept, r_value) = stats.probplot(data, dist="norm", plot=ax)

    r_squared = r_value ** 2

    # Visual settings
    ax.get_lines()[0].set_alpha(0.3)
    ax.get_lines()[0].set_markersize(3)

    ax.set_title(f"Q-Q Plot (N = {len(data)})")
    ax.set_xlabel("Theoretical Quantiles (Standard Normal)")
    ax.set_ylabel("Ordered Values (final_x)")
    ax.grid(True, linestyle='--', alpha=0.7)

    # 5. Add metrics
    textstr = '\n'.join((
        f'Valid Points: {len(data)}',
        f'R² (Linearity): {r_squared:.4f}',
        f'Skewness: {skewness:.3f}',
        f'Kurtosis: {kurtosis:.3f}'
    ))

    props = dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.9)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=props)

    plt.tight_layout()
    os.makedirs("figures", exist_ok=True)
    plt.savefig(os.path.join("figures", "galton_qq_plot.png"), dpi=300)
    plt.show()