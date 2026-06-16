import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors  # Fixed the missing 'as' here
from matplotlib.patches import Polygon

def plot_pyramid_heatmap(matrix, mod_positions, mod_types, type_colors, rna_length):
    """
    Plots an upward-pointing pyramid heatmap (LD/Hi-C style) with a horizontal transcript axis,
    including downward-pointing coordinate ticks.
    
    Parameters:
    - matrix: NxN symmetric numpy array of association scores (-1 to 1).
    - mod_positions: List/array of length N with the nucleotide positions (must be sorted).
    - mod_types: List/array of length N with the modification types.
    - type_colors: Dict mapping mod_types to specific colors.
    - rna_length: Total length of the RNA transcript.
    """
    N = len(mod_positions)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_aspect('equal')
    
    cmap = plt.cm.bwr # Blue (-1) to White (0) to Red (+1)
    
    # 1. Define vertical layout
    # Minimum offset of 1.0 unit ensures small N values don't overlap the axis
    offset = max(1.0, N * 0.15)

    # 2. Plot the Rotated Matrix Cells (The Pyramid)
    for i in range(N):
        for j in range(i, N):
            val = matrix[i, j]
            
            # Calculate the center of the diamond
            x_c = (i + j) / 2.0
            y_c = (j - i) / 2.0 + offset
            
            # Define the 4 corners of the diamond
            vertices = [
                (x_c, y_c + 0.5),       # Top
                (x_c + 0.5, y_c),       # Right
                (x_c, y_c - 0.5),       # Bottom
                (x_c - 0.5, y_c)        # Left
            ]
            
            # Normalize matrix value from [-1, 1] to [0, 1] for the colormap
            color_val = (val + 1) / 2.0
            color = cmap(color_val)
            
            # Draw the cell
            polygon = Polygon(vertices, facecolor=color, edgecolor='white', linewidth=1)
            ax.add_patch(polygon)

    # 3. Plot the horizontal RNA Transcript Axis at Y=0
    axis_start_x = -0.5
    axis_end_x = N - 0.5
    axis_width = axis_end_x - axis_start_x
    
    ax.plot([axis_start_x, axis_end_x], [0, 0], color='black', lw=3, zorder=3)
    
    # Standard fonts (no bold)
    ax.text(axis_start_x - 0.2, 0, "5'", va='center', ha='right', fontsize=12)
    ax.text(axis_end_x + 0.2, 0, "3'", va='center', ha='left', fontsize=12)

    # 4. Add physical coordinate ticks and labels pointing downwards
    # Generates 5 evenly spaced ticks along the transcript length
    tick_positions = np.linspace(0, rna_length, 5, dtype=int)
    for tick in tick_positions:
        # Calculate X coordinate for the tick mark
        tick_fraction = tick / rna_length
        tick_x = axis_start_x + (tick_fraction * axis_width)
        
        # Draw a thin tick line pointing downwards (from Y=0 to Y=-0.15)
        ax.plot([tick_x, tick_x], [0, -0.15], color='black', lw=1, zorder=3)
        
        # Add the position label text just below the tick line
        ax.text(tick_x, -0.35, f"{tick}", va='top', ha='center', fontsize=10)

    # 5. Map physical positions and draw connecting lines
    for i, (pos, m_type) in enumerate(zip(mod_positions, mod_types)):
        col = type_colors[m_type]
        
        # Calculate proportional X coordinate on the transcript axis
        fraction = pos / rna_length
        x_axis = axis_start_x + (fraction * axis_width)
        
        # Draw the modification dot on the transcript
        label = m_type if m_type not in ax.get_legend_handles_labels()[1] else ""
        ax.scatter(x_axis, 0, color=col, s=80, zorder=4, label=label)
        
        # Draw the line connecting the physical coordinate to the matrix column
        x_matrix = i
        y_matrix = offset - 0.5 
        
        # Solid and thin line
        ax.plot([x_axis, x_matrix], [0, y_matrix], color=col, linestyle='-', lw=0.8, zorder=2)

    # 6. Colorbar and Formatting
    norm = mcolors.Normalize(vmin=-1, vmax=1)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    
    cbar = fig.colorbar(sm, ax=ax, orientation='horizontal', pad=0.08, shrink=0.4, aspect=20)
    cbar.set_label('Association Score (Blue = Negative, Red = Positive)')
    
    # Set limits dynamically (Y-lower extended to -1.2 to give the downward labels breathing room)
    ax.set_xlim(axis_start_x - 1.5, axis_end_x + 1.5)
    ax.set_ylim(-1.2, ((N - 1) / 2.0) + offset + 1)
    ax.axis('off')
    
    # Place legend neatly
    plt.legend(title="Modification Type", loc='upper left', bbox_to_anchor=(0.85, 0.95))
    
    plt.tight_layout()
    plt.show()

# --- Example Usage ---
if __name__ == "__main__":
    np.random.seed(42) 
    
    rna_len = 3000
    
    # Example dataset with 6 sites
    positions = np.array([250, 600, 1350, 1400, 2200, 2850])
    types = ['m6A', 'Pseudouridine', 'm6A', '5mC', 'Pseudouridine', 'm6A']
    
    colors = {
        'm6A': '#d62728',          
        'Pseudouridine': '#1f77b4', 
        '5mC': '#2ca02c'            
    }
    
    N = len(positions)
    mock_matrix = np.zeros((N, N))
    for i in range(N):
        for j in range(i, N):
            if i == j:
                mock_matrix[i, j] = 1.0
            else:
                score = np.random.uniform(-1, 1)
                mock_matrix[i, j] = score
                mock_matrix[j, i] = score

    plot_pyramid_heatmap(mock_matrix, positions, types, colors, rna_len)
