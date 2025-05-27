#!/usr/bin/env python
"""
This script analyzes loop representations from the transformer decoder,
computes PCA to reduce dimensionality, and visualizes the latent space trajectories.
"""

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from pathlib import Path
import os
import json
from tqdm import tqdm
import seaborn as sns

from models.multimodal_to_smiles import MultiModalToSMILESModel
from models.smiles_tokenizer import SmilesTokenizer
from inference.inference import ModelInference, DecodingStrategy

# Import utility functions from test_inference.py
from test_inference import load_config, get_ir_tokenizer, detect_ir_as_prompt, SimpleSpectralSmilesDataset

def compute_pca_for_representations(representations, n_components=6):
    """
    Compute PCA for a list of representation tensors.
    
    Args:
        representations: List of tensors, each with shape [batch, seq_len, hidden_dim]
        n_components: Number of PCA components to compute
        
    Returns:
        A tuple of (pca_model, transformed_representations)
    """
    # Flatten the representations to 2D: [batch*seq_len*num_loops, hidden_dim]
    all_reps = []
    for rep in representations:
        # Convert to numpy for sklearn compatibility
        rep_np = rep.cpu().numpy()
        # Reshape to [batch*seq_len, hidden_dim]
        batch_size, seq_len, hidden_dim = rep_np.shape
        flattened = rep_np.reshape(-1, hidden_dim)
        all_reps.append(flattened)
    
    # Concatenate all representations
    all_reps_flat = np.vstack(all_reps)
    
    # Fit PCA
    pca = PCA(n_components=n_components)
    transformed = pca.fit_transform(all_reps_flat)
    
    # Reshape transformed back to list of [batch, seq_len, n_components]
    transformed_reps = []
    idx = 0
    for rep in representations:
        batch_size, seq_len, _ = rep.shape
        n_samples = batch_size * seq_len
        rep_transformed = transformed[idx:idx+n_samples]
        rep_transformed = rep_transformed.reshape(batch_size, seq_len, n_components)
        transformed_reps.append(rep_transformed)
        idx += n_samples
    
    return pca, transformed_reps

def plot_pca_trajectories(pca_results, tokens, output_dir, token_to_visualize=None, tokenizer=None,
                          early_exit_loop_index=None, plot_early_exit_point_flag=True):
    """
    Plot PCA trajectories of representations across loops.
    
    Args:
        pca_results: List of arrays with shape [batch, seq_len, n_components]
        tokens: List of token IDs in the sequence
        output_dir: Directory to save the plots
        token_to_visualize: Specific token ID to visualize, or None for all tokens
        tokenizer: Tokenizer to convert token IDs to strings for the legend
        early_exit_loop_index: Index of the loop where the early exit condition is met, or None (ignored in this function now)
        plot_early_exit_point_flag: Whether to highlight the early exit point (ignored in this function now)
    """
    num_loops = len(pca_results)
    n_components = pca_results[0].shape[2]
    seq_len = pca_results[0].shape[1]
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine which tokens to analyze
    token_positions = []
    token_labels = []
    
    if token_to_visualize is not None:
        # Find all positions where the specified token appears
        for i, token in enumerate(tokens):
            if token == token_to_visualize:
                token_positions.append(i)
                if tokenizer:
                    token_labels.append(f"{tokenizer.decode([token])} (pos {i})")
                else:
                    token_labels.append(f"Token {token} (pos {i})")
        if not token_positions:
            print(f"Token ID {token_to_visualize} not found in sequence.")
            return
    else:
        # Use ALL token positions - no limit
        token_positions = list(range(seq_len))
        
        # Create labels for each token
        for i in token_positions:
            if tokenizer and i < len(tokens):
                token_labels.append(f"{tokenizer.decode([tokens[i]])} (pos {i})")
            else:
                token_labels.append(f"Pos {i}")
    
    print(f"Plotting trajectories for {len(token_positions)} tokens")
    
    # Create a color map for tokens - use a cyclic colormap that works well with many tokens
    token_cmap = plt.cm.tab20
    # If we have more than 20 tokens, we'll cycle through the colors
    token_colors = [token_cmap(i % 20) for i in range(len(token_positions))]
    
    # For more than 20 tokens, adjust transparency to avoid overwhelming the plot
    token_alpha = min(0.9, max(0.3, 5.0 / max(1, len(token_positions) / 20)))
    print(f"Using transparency level: {token_alpha:.2f} for {len(token_positions)} tokens")
    
    # Loop color map for trajectory steps
    loop_cmap = plt.cm.viridis
    loop_colors = [loop_cmap(i/num_loops) for i in range(num_loops)]
    
    # Create plots for pairs of PCA components
    component_pairs = [(0, 1), (2, 3), (4, 5)] if n_components >= 6 else [(0, 1)]
    
    # Create a figure for each component pair
    for pair_idx, (pc1, pc2) in enumerate(component_pairs):
        fig, ax = plt.subplots(figsize=(14, 12))
        
        # Set title
        fig.suptitle(f"Latent Space Trajectories across {num_loops} Loops (PC{pc1+1} vs PC{pc2+1})", fontsize=16)
        
        # Add loop count indicator in the corner
        loop_info = f"Total loops: {num_loops}"
        ax.text(0.02, 0.98, loop_info, transform=ax.transAxes, fontsize=12,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        # Number of tokens being plotted
        num_tokens_info = f"Tokens plotted: {len(token_positions)}/{seq_len}"
        ax.text(0.02, 0.93, num_tokens_info, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        # Plot trajectories for each selected token position
        all_trajectories = []
        
        for idx, pos in enumerate(token_positions):
            token_color = token_colors[idx]
            label = token_labels[idx] if idx < len(token_labels) else f"Pos {pos}"
            
            # Extract trajectory for this token position
            trajectory = []
            for loop_idx in range(num_loops):
                # Get PCA coordinates for this token at this loop
                if pos < pca_results[loop_idx].shape[1]:
                    point = pca_results[loop_idx][0, pos, [pc1, pc2]]
                    trajectory.append(point)
                else:
                    # Skip if position is out of bounds
                    print(f"Warning: Position {pos} out of bounds for loop {loop_idx}")
                    continue
            
            if not trajectory:
                continue
                
            # Convert to numpy array for easier indexing
            trajectory = np.array(trajectory)
            all_trajectories.append((trajectory, token_color, label))
            
            # Plot the full trajectory line
            ax.plot(
                trajectory[:, 0],
                trajectory[:, 1],
                '-', 
                linewidth=1.0,
                alpha=max(0.2, token_alpha * 0.6),  # More transparent line
                color=token_color,
                label=label
            )
            
            # Plot ALL points in the trajectory
            for i, point in enumerate(trajectory):
                # Size varies with loop index to show progression
                size = 15 + i * 1.0  # Smaller points but still increasing with loop
                # Use a different marker for the first point
                if i == 0:
                    marker = 'o'  # Circle for first point
                    z_order = 4
                    point_alpha = token_alpha * 1.1  # Slightly more visible
                    edgecolor = 'black'
                    linewidth = 0.5
                else:
                    marker = '.'  # Dot for intermediate and last points
                    z_order = 3
                    point_alpha = token_alpha * 0.9
                    edgecolor = None
                    linewidth = 0
                
                ax.scatter(
                    point[0],
                    point[1], 
                    s=size, 
                    color=loop_colors[i],  # Color by loop index
                    marker=marker,
                    edgecolor=edgecolor,
                    linewidth=linewidth,
                    alpha=point_alpha,
                    zorder=z_order
                )
            
            # Annotate only the first point - only if not too many tokens
            if len(token_positions) <= 30:
                ax.annotate(
                    "0",  # Use step number instead of "start"
                    (trajectory[0, 0], trajectory[0, 1]),
                    fontsize=7,
                    xytext=(3, 3),
                    textcoords='offset points',
                    color=token_color
                )
        
        # Create a colorbar to show loop progression
        sm = plt.cm.ScalarMappable(cmap=loop_cmap, norm=plt.Normalize(0, num_loops-1))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label('Loop Number', fontsize=10)
        
        # Set axis labels and grid
        ax.set_xlabel(f"Principal Component {pc1+1}", fontsize=12)
        ax.set_ylabel(f"Principal Component {pc2+1}", fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)
        
        # Add a legend for token colors (use a subset if there are too many)
        if len(all_trajectories) <= 15:
            ax.legend(loc='upper right', fontsize=8)
        else:
            # With many tokens, create a very small subset for the legend
            max_legend_entries = 10
            step = max(1, len(all_trajectories) // max_legend_entries)
            handles, labels = ax.get_legend_handles_labels()
            
            num_actual_token_labels = len(labels) # No early exit label in this plot

            if step > 1:
                sample_indices = list(range(0, num_actual_token_labels, step))
                legend_handles = [handles[i] for i in sample_indices if i < len(handles)]
                legend_labels = [labels[i] for i in sample_indices if i < len(labels)]
                
                if len(legend_handles) > 0 and num_actual_token_labels > len(sample_indices):
                    more_count = num_actual_token_labels - len(legend_handles)
                    if more_count > 0:
                        legend_labels[-1] += f" (+ {more_count} more)"
                
                ax.legend(
                    legend_handles, 
                    legend_labels,
                    loc='upper right', 
                    fontsize=8, 
                    title=f"Showing subset of {num_actual_token_labels} tokens"
                )
            else:
                ax.legend(
                    handles, 
                    labels, 
                    loc='upper right', 
                    fontsize=8
                )
                
        # Also save a complete list of tokens to a text file for reference
        token_list_file = os.path.join(output_dir, f"token_list_{pc1+1}_{pc2+1}.txt")
        with open(token_list_file, 'w') as f:
            f.write(f"Token list for PC{pc1+1} vs PC{pc2+1} visualization:\n")
            f.write(f"Total tokens: {len(token_labels)}\n\n")
            for i, label in enumerate(token_labels):
                f.write(f"{i+1}. {label}\n")
        
        # Adjust layout and save
        plt.tight_layout()
        
        # Save figure
        token_str = f"token_{token_to_visualize}" if token_to_visualize is not None else "all_tokens"
        filename = f"pca_trajectories_{token_str}_pc{pc1+1}_pc{pc2+1}.png"
        plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()

def plot_all_token_trajectories(pca_results, tokens, tokenizer, output_dir, num_tokens_to_plot=5,
                                early_exit_loop_index=None, plot_early_exit_point_flag=True):
    """
    Generate separate plots for individual tokens.
    
    Args:
        pca_results: List of arrays with shape [batch, seq_len, n_components]
        tokens: List of token IDs in the sequence
        tokenizer: The tokenizer to convert token IDs to strings
        output_dir: Directory to save the plots
        num_tokens_to_plot: Number of tokens to plot individually
        early_exit_loop_index: Index of the loop where the early exit condition is met, or None
        plot_early_exit_point_flag: Whether to highlight the early exit point
    """
    # Create a directory for token-specific plots
    token_plots_dir = os.path.join(output_dir, "token_trajectories")
    os.makedirs(token_plots_dir, exist_ok=True)
    
    # Create a grid of PCA visualizations for different tokens
    seq_len = min(len(tokens), pca_results[0].shape[1])
    tokens_to_visualize = tokens[:seq_len]
    
    # Create a single summary plot showing all selected tokens
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    component_pairs = [(0, 1), (2, 3), (4, 5)]
    
    # Limit the number of tokens to plot
    if num_tokens_to_plot < len(tokens_to_visualize):
        # Choose tokens at equally spaced intervals
        indices = np.linspace(0, len(tokens_to_visualize) - 1, num_tokens_to_plot, dtype=int)
        tokens_to_visualize = [tokens_to_visualize[i] for i in indices]
    
    # Create a color map for different tokens
    token_cmap = plt.cm.tab10
    token_colors = [token_cmap(i % 10) for i in range(len(tokens_to_visualize))]
    
    # First create the summary plot with all tokens
    for pair_idx, (pc1, pc2) in enumerate(component_pairs):
        if pair_idx >= len(axes):
            break
            
        ax = axes[pair_idx]
        ax.set_title(f"PC{pc1+1} vs PC{pc2+1}")
        
        for i, token_id in enumerate(tokens_to_visualize):
            token_str = tokenizer.decode([token_id])
            token_pos = None
            
            # Find position of this token in the sequence
            for pos, t in enumerate(tokens):
                if t == token_id:
                    token_pos = pos
                    break
                    
            if token_pos is not None:
                # Extract trajectory for this token
                trajectory = []
                for loop_idx in range(len(pca_results)):
                    point = pca_results[loop_idx][0, token_pos, [pc1, pc2]]
                    trajectory.append(point)
                
                trajectory = np.array(trajectory)
                
                # Plot the trajectory
                ax.plot(
                    trajectory[:, 0],
                    trajectory[:, 1],
                    '-',
                    color=token_colors[i],
                    linewidth=1.2,
                    alpha=0.7,
                    label=f"{token_str}"
                )
                
                # Mark first and last points
                ax.scatter(
                    trajectory[0, 0],
                    trajectory[0, 1],
                    s=40,
                    marker='o',
                    color=token_colors[i],
                    edgecolor='black',
                    linewidth=0.5,
                    zorder=3
                )
                
                ax.scatter(
                    trajectory[-1, 0],
                    trajectory[-1, 1],
                    s=80,
                    marker='*',
                    color=token_colors[i],
                    edgecolor='black',
                    linewidth=0.5,
                    zorder=3
                )
        
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    
    # Add legend to the last subplot
    if len(axes) > 0:
        handles, labels = axes[-1].get_legend_handles_labels()
        fig.legend(handles, labels, loc='center right', fontsize=10)
    
    plt.tight_layout()
    plt.subplots_adjust(right=0.85)  # Make room for the legend
    
    # Save the summary plot
    summary_filename = os.path.join(output_dir, f"token_trajectories_summary.png")
    plt.savefig(summary_filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Now create individual detailed plots for each token
    print(f"Creating detailed plots and zoomed views for {len(tokens_to_visualize)} tokens...")
    for i, token_id in enumerate(tokens_to_visualize):
        token_str = tokenizer.decode([token_id])
        try:
            print(f"  Plotting full trajectory for token: '{token_str}' (ID: {token_id})")
            plot_pca_trajectories(pca_results, tokens, token_plots_dir, token_to_visualize=token_id, tokenizer=tokenizer,
                                  early_exit_loop_index=early_exit_loop_index,
                                  plot_early_exit_point_flag=plot_early_exit_point_flag)
            
            # Also create a zoomed plot for the last 15 steps
            print(f"  Plotting zoomed trajectory for token: '{token_str}' (ID: {token_id})")
            plot_zoomed_trajectory(pca_results, tokens, token_plots_dir, token_to_visualize=token_id, tokenizer=tokenizer, num_last_steps=20,
                                   early_exit_loop_index=early_exit_loop_index, plot_early_exit_point_flag=plot_early_exit_point_flag)
            
        except Exception as e:
            print(f"Error plotting token {token_id}: {e}")
    
    print(f"Individual token plots saved to {token_plots_dir}")

def analyze_representation_distances(representations):
    """
    Analyze distances between representation vectors across loops.
    
    Args:
        representations: List of tensors, each with shape [batch, seq_len, hidden_dim]
        
    Returns:
        A dictionary of distance metrics
    """
    num_loops = len(representations)
    metrics = {}
    
    # Compute cosine distances between consecutive loop representations
    cosine_distances = []
    euclidean_distances = []
    
    for i in range(num_loops - 1):
        rep1 = representations[i]
        rep2 = representations[i + 1]
        
        # Normalize representations for cosine distance
        rep1_norm = rep1 / rep1.norm(dim=2, keepdim=True)
        rep2_norm = rep2 / rep2.norm(dim=2, keepdim=True)
        
        # Compute cosine similarity
        cosine_sim = (rep1_norm * rep2_norm).sum(dim=2)
        # Convert to distance (1 - similarity)
        cosine_dist = 1 - cosine_sim
        
        # Compute Euclidean distance
        euclidean_dist = (rep1 - rep2).norm(dim=2)
        
        cosine_distances.append(cosine_dist.mean().item())
        euclidean_distances.append(euclidean_dist.mean().item())
    
    metrics['cosine_distances'] = cosine_distances
    metrics['euclidean_distances'] = euclidean_distances
    
    # Also compute the "convergence" - how much the representation changes between the first and last loop
    first_rep = representations[0]
    last_rep = representations[-1]
    
    # Normalize for cosine distance
    first_norm = first_rep / first_rep.norm(dim=2, keepdim=True)
    last_norm = last_rep / last_rep.norm(dim=2, keepdim=True)
    
    # Cosine similarity between first and last
    cosine_sim_first_last = (first_norm * last_norm).sum(dim=2)
    cosine_dist_first_last = 1 - cosine_sim_first_last
    
    # Euclidean distance between first and last
    euclidean_dist_first_last = (first_rep - last_rep).norm(dim=2)
    
    metrics['cosine_dist_first_last'] = cosine_dist_first_last.mean().item()
    metrics['euclidean_dist_first_last'] = euclidean_dist_first_last.mean().item()
    
    return metrics

def plot_distance_metrics(metrics, output_dir):
    """
    Plot distance metrics across loops.
    
    Args:
        metrics: Dictionary of distance metrics
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot cosine distances
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(metrics['cosine_distances']) + 1), 
             metrics['cosine_distances'], 
             marker='o', 
             linestyle='-',
             linewidth=2)
    plt.title('Cosine Distance Between Consecutive Loop Representations')
    plt.xlabel('Loop Transition')
    plt.ylabel('Cosine Distance')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'cosine_distances.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot Euclidean distances
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(metrics['euclidean_distances']) + 1), 
             metrics['euclidean_distances'], 
             marker='o', 
             linestyle='-',
             linewidth=2,
             color='orange')
    plt.title('Euclidean Distance Between Consecutive Loop Representations')
    plt.xlabel('Loop Transition')
    plt.ylabel('Euclidean Distance')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'euclidean_distances.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_zoomed_trajectory(pca_results, tokens, output_dir, token_to_visualize, tokenizer, num_last_steps=15,
                             early_exit_loop_index=None, plot_early_exit_point_flag=True):
    """
    Plot a zoomed-in view of the last N steps of a token's trajectory.

    Args:
        pca_results: List of arrays with shape [batch, seq_len, n_components]
        tokens: List of token IDs in the sequence
        output_dir: Directory to save the plots
        token_to_visualize: Specific token ID to visualize
        tokenizer: Tokenizer to convert token ID to string
        num_last_steps: Number of final steps to include in the zoom
        early_exit_loop_index: Index of the loop where early exit is met
        plot_early_exit_point_flag: Whether to plot the early exit point
    """
    num_loops = len(pca_results)
    if num_loops <= num_last_steps:
        print(f"Not enough loops ({num_loops}) to zoom on the last {num_last_steps} steps. Skipping zoom plot.")
        return
        
    n_components = pca_results[0].shape[2]
    seq_len = pca_results[0].shape[1]
    
    # Find the position of the token
    token_pos = -1
    for i, token in enumerate(tokens):
        if token == token_to_visualize:
            token_pos = i
            break
            
    if token_pos == -1:
        print(f"Token ID {token_to_visualize} not found for zoomed plot.")
        return
        
    token_str = tokenizer.decode([token_to_visualize])
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Loop color map for trajectory steps
    loop_cmap = plt.cm.viridis
    loop_colors = [loop_cmap(i/num_loops) for i in range(num_loops)]
    
    # Component pairs
    component_pairs = [(0, 1), (2, 3), (4, 5)] if n_components >= 6 else [(0, 1)]
    
    # Loop start index for the zoomed view
    zoom_start_loop = num_loops - num_last_steps
    
    # Create a plot for each component pair
    for pair_idx, (pc1, pc2) in enumerate(component_pairs):
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Extract the latter part of the trajectory
        zoomed_trajectory = []
        for loop_idx in range(zoom_start_loop, num_loops):
            if token_pos < pca_results[loop_idx].shape[1]:
                point = pca_results[loop_idx][0, token_pos, [pc1, pc2]]
                zoomed_trajectory.append(point)
            else:
                continue
        
        if not zoomed_trajectory:
            print(f"Could not extract zoomed trajectory for token {token_str}")
            plt.close(fig)
            continue
            
        zoomed_trajectory = np.array(zoomed_trajectory)
        
        # Determine plot limits based on the zoomed trajectory range
        x_min, x_max = zoomed_trajectory[:, 0].min(), zoomed_trajectory[:, 0].max()
        y_min, y_max = zoomed_trajectory[:, 1].min(), zoomed_trajectory[:, 1].max()
        x_margin = (x_max - x_min) * 0.15  # Add 15% margin
        y_margin = (y_max - y_min) * 0.15
        ax.set_xlim(x_min - x_margin, x_max + x_margin)
        ax.set_ylim(y_min - y_margin, y_max + y_margin)
        
        # Plot the zoomed trajectory segments and points
        for i in range(len(zoomed_trajectory) - 1):
            current_loop_idx = zoom_start_loop + i
            ax.plot(
                zoomed_trajectory[i:i+2, 0],
                zoomed_trajectory[i:i+2, 1],
                '-',
                color=loop_colors[current_loop_idx],
                linewidth=1.5,
                alpha=0.8
            )
        
        # Plot points with varying sizes and annotations
        for i, point in enumerate(zoomed_trajectory):
            current_loop_idx = zoom_start_loop + i
            size = 20 + i * 1.5
            ax.scatter(
                point[0],
                point[1],
                s=size,
                color=loop_colors[current_loop_idx],
                edgecolor='black',
                linewidth=0.5,
                alpha=0.9,
                zorder=3
            )
            # Annotate points with their loop index
            ax.annotate(
                f"{current_loop_idx}",
                (point[0], point[1]),
                fontsize=8,
                xytext=(4, 4),
                textcoords='offset points'
            )

        # Highlight the early exit point if it's in the zoomed window
        if plot_early_exit_point_flag and early_exit_loop_index is not None and zoom_start_loop <= early_exit_loop_index < num_loops:
            # Calculate the index relative to the zoomed_trajectory array
            relative_exit_idx = early_exit_loop_index - zoom_start_loop
            if 0 <= relative_exit_idx < len(zoomed_trajectory):
                early_exit_point_zoomed = zoomed_trajectory[relative_exit_idx]
                ax.scatter(
                    early_exit_point_zoomed[0],
                    early_exit_point_zoomed[1],
                    s=200,  # Larger size for emphasis in zoom
                    marker='X',
                    color='red',
                    edgecolor='black',
                    linewidth=1.5,
                    alpha=1.0,
                    zorder=6, # Ensure it's on top
                    label=f'Early Exit @ Loop {early_exit_loop_index}'
                )
                ax.annotate(
                    f"Exit@{early_exit_loop_index}",
                    (early_exit_point_zoomed[0], early_exit_point_zoomed[1]),
                    fontsize=9,
                    xytext=(5, -5),
                    textcoords='offset points',
                    color='red',
                    bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.8)
                )
                ax.legend(loc='best', fontsize=8) # Add legend if exit point shown

            
        # Add title and labels
        title = f"Zoomed Trajectory (Loops {zoom_start_loop}-{num_loops-1}) for Token '{token_str}' (pos {token_pos})\nPC{pc1+1} vs PC{pc2+1}"
        ax.set_title(title, fontsize=14)
        ax.set_xlabel(f"Principal Component {pc1+1}", fontsize=12)
        ax.set_ylabel(f"Principal Component {pc2+1}", fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Create a colorbar for loop number reference
        sm = plt.cm.ScalarMappable(cmap=loop_cmap, norm=plt.Normalize(zoom_start_loop, num_loops-1))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label('Loop Number', fontsize=10)
        
        plt.tight_layout()
        
        # Save figure
        filename = f"zoomed_trajectory_token_{token_to_visualize}_pc{pc1+1}_pc{pc2+1}.png"
        plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description="Analyze Loop Representations and Generate PCA Visualizations")
    parser.add_argument('--checkpoint', type=str,default = 'checkpoints/100k.pt', help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='configs/test_config.yaml', help='Path to configuration YAML file')
    parser.add_argument('--max_loops', type=int, default=30, help='Maximum number of loops for representation analysis')
    parser.add_argument('--sample_idx', type=int, default=0, help='Index of the sample to analyze in the dataset')
    parser.add_argument('--split', type=str, default='test', help='Dataset split to use')
    parser.add_argument('--output_dir', type=str, default='representation_analysis', help='Directory to save results')
    
    # Arguments for early exit point plotting
    parser.add_argument('--plot_early_exit_point', type=lambda x: (str(x).lower() == 'true'), default=True, help='Whether to highlight the early exit point (default: True)')
    parser.add_argument('--early_exit_threshold', type=float, default=0.001, help='Convergence threshold for early exit (default: 0.01)')
    parser.add_argument('--early_exit_metric', type=str, default='euclidean', choices=['cosine', 'euclidean'], help='Distance metric for early exit (cosine or euclidean, default: cosine)')

    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize SMILES tokenizer
    current_dir = os.path.dirname(os.path.realpath(__file__))
    vocab_path = os.path.join(current_dir, 'training/vocab.txt')
    tokenizer = SmilesTokenizer(vocab_file=vocab_path)
    
    # Load NMR tokenizer
    nmr_vocab_path = Path(config['data']['tokenized_dir']).parent / 'vocab.json'
    if not nmr_vocab_path.exists():
        raise FileNotFoundError(f"NMR vocabulary not found at {nmr_vocab_path}")
    with open(nmr_vocab_path) as f:
        nmr_tokenizer = json.load(f)
    
    # Load checkpoint and detect IR settings
    checkpoint = torch.load(args.checkpoint, map_location=device)
    auto_ir_as_prompt, extra_params = detect_ir_as_prompt(checkpoint, config)
    ir_as_prompt = auto_ir_as_prompt
    
    # Get IR tokenizer if needed
    ir_tokenizer = None
    if ir_as_prompt:
        ir_tokenizer = get_ir_tokenizer(config)
        print(f"IR tokenizer has {len(ir_tokenizer)} tokens")
    
    # Create the model with appropriate parameters
    smiles_vocab_size = len(tokenizer)
    token_ids = list(nmr_tokenizer.values())
    nmr_vocab_size = max(token_ids) + 1
    model_kwargs = {
        'smiles_vocab_size': smiles_vocab_size,
        'nmr_vocab_size': nmr_vocab_size,
        'max_seq_length': config['model']['max_seq_length'],
        'max_nmr_length': config['model']['max_nmr_length'],
        'max_memory_length': config['model']['max_memory_length'],
        'embed_dim': config['model']['embed_dim'],
        'num_heads': config['model']['num_heads'],
        'num_layers': config['model']['num_layers'],
        'dropout': config['model']['dropout'],
        'verbose': False,
        'use_stablemax': config['model'].get('use_stablemax', False),
        'ir_as_prompt': ir_as_prompt,
        'ir_encoder_type': config['model'].get('ir_encoder_type', 'regular'),
        'max_loops': args.max_loops,
        'loops_representation': True,  # Enable representation tracking
        'use_rmsnorm': True
    }
    if ir_as_prompt:
        if 'ir_vocab_size' in extra_params:
            model_kwargs['ir_vocab_size'] = extra_params['ir_vocab_size']
        else:
            model_kwargs['ir_vocab_size'] = len(ir_tokenizer)
    
    model = MultiModalToSMILESModel(**model_kwargs).to(device)
    try:
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Successfully loaded model state from checkpoint")
    except Exception as e:
        print(f"Error loading model state: {e}")
        return
    model.eval()
    
    # Create inference wrapper
    inference = ModelInference(model, tokenizer, device, ir_as_prompt=ir_as_prompt)
    
    # Create dataset
    dataset = SimpleSpectralSmilesDataset(
        data_dir=config['data']['tokenized_dir'],
        split=args.split,
        smiles_tokenizer=tokenizer,
        spectral_tokenizer=nmr_tokenizer,
        max_smiles_len=config['model']['max_seq_length'],
        max_nmr_len=config['model']['max_nmr_length'],
        ir_as_prompt=ir_as_prompt,
        ir_tokenizer=ir_tokenizer
    )
    
    # Ensure sample index is valid
    if args.sample_idx >= len(dataset):
        print(f"Sample index {args.sample_idx} exceeds dataset size {len(dataset)}. Using sample 0.")
        args.sample_idx = 0
    
    # Get the sample to analyze
    target_tokens, (ir_tensor, _), nmr_tokens, _ = dataset[args.sample_idx]
    target_smiles = dataset.targets[args.sample_idx]
    print(f"Analyzing sample {args.sample_idx}: {target_smiles}")
    
    if ir_tensor is not None:
        ir_data = ir_tensor.to(device)
    else:
        ir_data = None
    nmr_tokens = nmr_tokens.to(device)
    
    # Create output directory
    output_dir = Path(args.output_dir) / f"sample_{args.sample_idx}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect representations for different loop counts
    all_representations = []
    all_metrics = {}
    
    # Run inference only once with the maximum number of loops
    print(f"Running inference with {args.max_loops} loops...")
    decoded, representations = inference.decode(
        nmr_tokens=nmr_tokens,
        ir_data=ir_data,
        strategy=DecodingStrategy.GREEDY_LOOP,
        max_len=config['model']['max_seq_length'],
        num_loops=args.max_loops,
        loops_representation=True
    )
    
    print(f"Generated SMILES: {decoded[0]}")
    
    # The returned representations already contain all intermediate loop states
    all_representations = representations
    
    # Analyze distances between consecutive representations
    metrics = analyze_representation_distances(representations)
    all_metrics[args.max_loops] = metrics
    
    # Determine early exit loop index
    early_exit_loop_index = None
    if args.plot_early_exit_point and metrics:
        distance_key = f'{args.early_exit_metric}_distances'
        if distance_key in metrics:
            distances = metrics[distance_key]
            for i, dist in enumerate(distances):
                if dist < args.early_exit_threshold:
                    early_exit_loop_index = i + 1 # Exit happens after this loop (i.e., at loop i+1)
                    print(f"Early exit condition met at loop {early_exit_loop_index} (distance {dist:.4f} < threshold {args.early_exit_threshold}) using {args.early_exit_metric} metric.")
                    break
            if early_exit_loop_index is None:
                print(f"Early exit condition not met within {args.max_loops} loops with threshold {args.early_exit_threshold} using {args.early_exit_metric} metric.")
        else:
            print(f"Warning: Distance metric '{distance_key}' not found in metrics. Cannot determine early exit point.")

    # Compute PCA for all representations
    pca, transformed_reps = compute_pca_for_representations(all_representations, n_components=6)
    
    # Plot PCA trajectories for all loops with all tokens in the same plot
    print("Generating main PCA visualization with all tokens...")
    plot_pca_trajectories(transformed_reps, target_tokens, output_dir, tokenizer=tokenizer,
                          early_exit_loop_index=early_exit_loop_index,
                          plot_early_exit_point_flag=args.plot_early_exit_point)
    
    # Plot individual token trajectories
    plot_all_token_trajectories(transformed_reps, target_tokens, tokenizer, output_dir, num_tokens_to_plot=5,
                                early_exit_loop_index=early_exit_loop_index,
                                plot_early_exit_point_flag=args.plot_early_exit_point)
    
    # Plot distance metrics
    if all_metrics:
        plot_distance_metrics(all_metrics[args.max_loops], output_dir)
    
    print(f"Analysis complete. Results saved to {output_dir}")

if __name__ == '__main__':
    main() 