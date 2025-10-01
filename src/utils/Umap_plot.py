"""
Robust version of the consistent UMAP analysis with proper error handling
and data format conversion.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import umap
import traceback

# Your existing color palette
cp = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

def convert_data_format(data, method_name=""):
    """
    Convert data to the appropriate format for different embedding methods.
    
    Parameters:
    -----------
    data : pandas.DataFrame or numpy.array
        Input data
    method_name : str
        Name of the method (for debugging)
        
    Returns:
    --------
    Converted data in appropriate format
    """
    if hasattr(data, 'values'):
        # It's a DataFrame, convert to numpy
        return data.values
    elif isinstance(data, list):
        # It's already a list
        return data
    else:
        # It's already a numpy array
        return data

def safe_embedding_call(embedding_func, *args, method_name="Unknown", **kwargs):
    """
    Safely call an embedding function with error handling.
    
    Parameters:
    -----------
    embedding_func : callable
        The embedding function to call
    *args : arguments for the embedding function
    method_name : str
        Name of the method for error reporting
    **kwargs : keyword arguments for the embedding function
        
    Returns:
    --------
    tuple or None : Result of embedding function or None if failed
    """
    try:
        print(f"  Processing {method_name}...")
        result = embedding_func(*args, **kwargs)
        print(f"  ✓ {method_name} completed successfully")
        return result
    except Exception as e:
        print(f"  ✗ {method_name} failed: {str(e)}")
        print(f"    Error details: {traceback.format_exc()}")
        return None

def generate_all_embeddings_robust(train_sct, val_sct, test_sct, ny_train, ny_val, ny_test):
    """
    Robustly generate all embedding methods with error handling.
    """
    import time_series_embeddings as embd  # Adjust import as needed
    
    embeddings_results = {}
    
    print("Generating embeddings with robust error handling...")
    print("-" * 60)
    
    # Convert data to numpy arrays for consistent processing
    train_data = convert_data_format(train_sct, "train")
    val_data = convert_data_format(val_sct, "val") 
    test_data = convert_data_format(test_sct, "test")
    
    print(f"Data shapes: Train={train_data.shape}, Val={val_data.shape}, Test={test_data.shape}")
    
    # PCA Embedding
    print("\n1. PCA Embedding:")
    result = safe_embedding_call(
        embd.pca_embedding_with_timing,
        train_data, val_data, test_data, 96,
        method_name="PCA"
    )
    if result:
        embeddings_results['PCA'] = result
    
    # Autoencoder Embedding  
    print("\n2. Autoencoder Embedding:")
    result = safe_embedding_call(
        embd.AE_embedding_with_timing,
        train_data, val_data, test_data,
        method_name="Autoencoder"
    )
    if result:
        embeddings_results['Autoencoder'] = result
    
    # FFT Embedding
    print("\n3. FFT Embedding:")
    result = safe_embedding_call(
        embd.fft_embedding_with_timing,
        train_data, val_data, test_data,
        method_name="FFT"
    )
    if result:
        embeddings_results['FFT'] = result
    
    # Wavelet Embedding
    print("\n4. Wavelet Embedding:")
    result = safe_embedding_call(
        embd.wavelet_embedding_with_timing,
        train_data, val_data, test_data,
        method_name="Wavelet"
    )
    if result:
        embeddings_results['Wavelet'] = result
    
    # LLE Embedding
    print("\n5. LLE Embedding:")
    lle_params = {'n_neighbors': 10, 'n_components': 50, 'random_state': 42}
    result = safe_embedding_call(
        embd.lle_embedding_with_timing,
        train_data, val_data, test_data, lle_params,
        method_name="LLE"
    )
    if result:
        embeddings_results['LLE'] = result
    
    # UMAP Embedding
    print("\n6. UMAP Embedding:")
    result = safe_embedding_call(
        embd.umap_embedding_with_timing,
        train_data, val_data, test_data,
        method_name="UMAP"
    )
    if result:
        embeddings_results['UMAP'] = result
    
    # Graph Embedding
    print("\n7. Graph Embedding:")
    # Convert to list format for graph embedding
    # try:
    #     train_list = [train_data[i] for i in range(len(train_data))]
    #     val_list = [val_data[i] for i in range(len(val_data))]
    #     test_list = [test_data[i] for i in range(len(test_data))]
        
    #     result = safe_embedding_call(
    #         embd.graph_embedding_with_timing,
    #         train_list, val_list, test_list,
    #         method_name="Graph"
    #     )
    #     if result:
    #         embeddings_results['Graph'] = result
    # except Exception as e:
    #     print(f"  ✗ Graph embedding failed during data conversion: {e}")
    
    # TDA Embedding
    print("\n8. TDA Embedding:")
    # TDA needs special list format
    try:
        # Convert to proper list format for TDA
        train_list = [train_data[i] for i in range(len(train_data))]
        val_list = [val_data[i] for i in range(len(val_data))]
        test_list = [test_data[i] for i in range(len(test_data))]
        
        print(f"  TDA input format check - first element type: {type(train_list[0])}, shape: {train_list[0].shape if hasattr(train_list[0], 'shape') else 'no shape'}")
        
        result = safe_embedding_call(
            embd.tda_embedding_with_timing,
            train_list, val_list, test_list,
            method_name="TDA"
        )
        if result:
            embeddings_results['TDA'] = result
    except Exception as e:
        print(f"  ✗ TDA embedding failed during data conversion: {e}")
    
    print("\n" + "="*60)
    print("DEEP LEARNING METHODS (NNCLR-based)")
    print("="*60)
    
    # NNCLR CNN Embedding
    print("\n9. NNCLR CNN Embedding:")
    # try:
    #     # NNCLR methods need DataFrame input and additional parameters
    #     # Convert numpy arrays back to DataFrames
    #     train_df = pd.DataFrame(train_data)
    #     val_df = pd.DataFrame(val_data) 
    #     test_df = pd.DataFrame(test_data)
        
    #     # Need to get number of classes from labels
    #     unique_classes = len(np.unique(ny_train.values if hasattr(ny_train, 'values') else ny_train))
    #     width = 128  # Default width for NNCLR embeddings
        
        # Import NNCLR modules
    #     try:
    #         import nnclr_embdcnn2 as nnclr_cnn
    #         result = safe_embedding_call(
    #             nnclr_cnn.nnclr_cnn_embedding_with_timing,
    #             train_df, val_df, test_df, ny_train, ny_val, ny_test, width, unique_classes,
    #             method_name="NNCLR CNN"
    #         )
    #         if result:
    #             embeddings_results['NNCLR CNN'] = result
    #     except ImportError as e:
    #         print(f"  ✗ Could not import nnclr_embdcnn2: {e}")
    # except Exception as e:
    #     print(f"  ✗ NNCLR CNN setup failed: {e}")
    
    # NNCLR LSTM Embedding
    print("\n10. NNCLR LSTM Embedding:")
    # try:
    #     train_df = pd.DataFrame(train_data)
    #     val_df = pd.DataFrame(val_data)
    #     test_df = pd.DataFrame(test_data)
        
    #     unique_classes = len(np.unique(ny_train.values if hasattr(ny_train, 'values') else ny_train))
    #     width = 128
        
    #     try:
    #         import nnclr_embd_rnn as nnclr_lstm
    #         result = safe_embedding_call(
    #             nnclr_lstm.nnclr_lstm_embedding_with_timing,
    #             train_df, val_df, test_df, ny_train, ny_val, ny_test, width, unique_classes,
    #             method_name="NNCLR LSTM"
    #         )
    #         if result:
    #             embeddings_results['NNCLR LSTM'] = result
    #     except ImportError as e:
    #         print(f"  ✗ Could not import nnclr_embd_rnn: {e}")
    # except Exception as e:
    #     print(f"  ✗ NNCLR LSTM setup failed: {e}")
    
    # NNCLR Transformer Embedding
    print("\n11. NNCLR Transformer Embedding:")
    # try:
    #     train_df = pd.DataFrame(train_data)
    #     val_df = pd.DataFrame(val_data)
    #     test_df = pd.DataFrame(test_data)
        
    #     unique_classes = len(np.unique(ny_train.values if hasattr(ny_train, 'values') else ny_train))
    #     width = 128
        
    #     try:
    #         import nnclr_embd_transformer as nnclr_transformer
    #         result = safe_embedding_call(
    #             nnclr_transformer.nnclr_cnn_embedding_with_timing,  # Note: function name in transformer file
    #             train_df, val_df, test_df, ny_train, ny_val, ny_test, width, unique_classes,
    #             method_name="NNCLR Transformer"
    #         )
    #         if result:
    #             embeddings_results['NNCLR Transformer'] = result
    #     except ImportError as e:
    #         print(f"  ✗ Could not import nnclr_embd_transformer: {e}")
    # except Exception as e:
    #     print(f"  ✗ NNCLR Transformer setup failed: {e}")
    
    print("\n" + "="*60)
    print(f"Successfully generated {len(embeddings_results)} out of 11 embedding methods:")
    for method in embeddings_results.keys():
        print(f"  ✓ {method}")
    
    all_methods = ['PCA', 'Autoencoder', 'FFT', 'Wavelet', 'LLE', 'UMAP', 'Graph', 'TDA', 
                   'NNCLR CNN', 'NNCLR LSTM', 'NNCLR Transformer']
    failed_methods = set(all_methods) - set(embeddings_results.keys())
    if failed_methods:
        print(f"Failed methods: {', '.join(failed_methods)}")
    print("="*60)
    
    return embeddings_results

def create_consistent_umap_layouts_robust(embeddings_results, labels_test, reference_method='PCA'):
    """
    Create consistent UMAP layouts with robust error handling.
    """
    if not embeddings_results:
        print("No embeddings available for UMAP visualization!")
        return {}
    
    if reference_method not in embeddings_results:
        print(f"Reference method '{reference_method}' not available. Using first available method.")
        reference_method = list(embeddings_results.keys())[0]
    
    print(f"Creating consistent UMAP layouts using {reference_method} as reference...")
    
    # Extract only test embeddings for visualization
    test_embeddings = {}
    for method_name, (train_emb, val_emb, test_emb, train_time, inf_time) in embeddings_results.items():
        # Convert to numpy array if it's a DataFrame
        if hasattr(test_emb, 'values'):
            test_embeddings[method_name] = test_emb.values
        else:
            test_embeddings[method_name] = np.array(test_emb)
        
        print(f"  {method_name} test embedding shape: {test_embeddings[method_name].shape}")
    
    # Step 1: Generate the reference layout
    print(f"\nGenerating reference layout from {reference_method}...")
    try:
        reference_reducer = umap.UMAP(
            n_neighbors=30,
            min_dist=0.2,
            n_components=2,
            random_state=42
        )
        reference_layout = reference_reducer.fit_transform(test_embeddings[reference_method])
        print(f"  Reference layout shape: {reference_layout.shape}")
    except Exception as e:
        print(f"Failed to create reference layout: {e}")
        return {}
    
    # Step 2: Generate all layouts using reference initialization
    layouts_2d = {}
    
    for method_name, test_embedding in test_embeddings.items():
        print(f"Generating layout for {method_name}...")
        try:
            # Create UMAP reducer with reference initialization
            reducer_with_init = umap.UMAP(
                n_neighbors=30,
                min_dist=0.2,
                n_components=2,
                init=reference_layout,
                random_state=42
            )
            
            layout = reducer_with_init.fit_transform(test_embedding)
            layouts_2d[method_name] = layout
            print(f"  ✓ {method_name} layout created: {layout.shape}")
            
        except Exception as e:
            print(f"  ✗ Failed to create layout for {method_name}: {e}")
            continue
    
    print(f"\nSuccessfully created {len(layouts_2d)} consistent UMAP layouts!")
    return layouts_2d

def plot_consistent_umap_grid_robust(layouts_2d, labels_test, figsize=(24, 18), save_path=None):
    """
    Plot all UMAP layouts in a grid with robust error handling.
    Enhanced for handling more methods (up to 11 methods).
    """
    if not layouts_2d:
        print("No layouts available for plotting!")
        return
    
    n_methods = len(layouts_2d)
    # Use exactly 3 columns per row
    n_cols = 3
    n_rows = (n_methods + n_cols - 1) // n_cols
    
    print(f"Creating grid plot: {n_rows}x{n_cols} for {n_methods} methods")
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_methods == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    # Ensure labels are in the right format
    if hasattr(labels_test, 'values'):
        label_values = labels_test.values.flatten()
    else:
        label_values = np.array(labels_test).flatten()
    
    unique_classes = sorted(np.unique(label_values))
    print(f"Found {len(unique_classes)} unique classes: {unique_classes}")
    
    # Color mapping for consistency
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_classes)))
    
    for i, (method_name, layout) in enumerate(layouts_2d.items()):
        ax = axes[i]
        
        try:
            # Create scatter plot with better styling
            for class_idx, class_label in enumerate(unique_classes):
                mask = label_values == class_label
                ax.scatter(
                    layout[mask, 0], 
                    layout[mask, 1], 
                    c=[colors[class_idx]],
                    label=f'Class {class_label}',
                    s=25,
                    alpha=0.7,
                    edgecolors='black',
                    linewidth=0.3
                )
            
            # Clean up method names and use consistent styling
            clean_method_name = method_name.replace('NNCLR ', '')
            ax.set_title(f'{clean_method_name}', fontsize=13, fontweight='bold')
            
            # Remove axis labels
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=9)
            
        except Exception as e:
            print(f"Failed to plot {method_name}: {e}")
            ax.text(0.5, 0.5, f'Error plotting\n{method_name}', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{method_name} (Error)', fontsize=13, color='red')
    
    # Remove unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    
    # Add legend
    try:
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc='upper center',
            bbox_to_anchor=(0.5, 0.02),
            ncol=min(len(unique_classes), 8),
            fontsize=11,
            frameon=True,
            fancybox=True,
            shadow=True
        )
    except Exception as e:
        print(f"Warning: Could not create legend: {e}")
    
    # Enhanced title
    plt.suptitle('Consistent UMAP Projections - Complete Embedding Methods Comparison\n' +
                 f'Traditional Methods + Deep Learning (NNCLR) Methods', 
                 fontsize=16, fontweight='bold', y=0.95)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12, top=0.88)
    
    if save_path:
        try:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
        except Exception as e:
            print(f"Failed to save plot: {e}")
    
    plt.show()

def main_robust_umap_analysis(train_sct, val_sct, test_sct, ny_train, ny_val, ny_test):
    """
    Main function for robust UMAP analysis with comprehensive error handling.
    Now includes both traditional and deep learning (NNCLR) methods.
    """
    print("="*70)
    print("COMPREHENSIVE ROBUST CONSISTENT UMAP ANALYSIS")
    print("Traditional Methods + Deep Learning (NNCLR) Methods")
    print("="*70)
    
    # Step 1: Generate all embeddings
    print("\nSTEP 1: Generating embeddings...")
    print("-"*50)
    embeddings_results = generate_all_embeddings_robust(
        train_sct, val_sct, test_sct, ny_train, ny_val, ny_test
    )
    
    if not embeddings_results:
        print("❌ No embeddings were generated successfully. Please check your data and methods.")
        return None, None
    
    # Step 2: Create consistent UMAP layouts
    print("\nSTEP 2: Creating consistent UMAP layouts...")
    print("-"*50)
    layouts_2d = create_consistent_umap_layouts_robust(
        embeddings_results, ny_test, reference_method='PCA'
    )
    
    if not layouts_2d:
        print("❌ No UMAP layouts were created successfully.")
        return embeddings_results, None
    
    # Step 3: Create comprehensive visualization
    print("\nSTEP 3: Creating comprehensive visualization...")
    print("-"*50)
    plot_consistent_umap_grid_robust(
        layouts_2d, ny_test, 
        save_path='comprehensive_consistent_umap_comparison.png'
    )
    
    # Step 4: Create separate deep learning methods comparison
    print("\nSTEP 4: Creating deep learning methods comparison...")
    print("-"*50)
    dl_methods = {k: v for k, v in layouts_2d.items() if 'NNCLR' in k}
    if dl_methods:
        plot_deep_learning_comparison(dl_methods, ny_test)
    else:
        print("No deep learning methods available for separate comparison.")
    
    print("\n" + "="*70)
    print("🎉 COMPREHENSIVE ANALYSIS COMPLETE!")
    print("="*70)
    print(f"✅ Successfully processed {len(embeddings_results)} embedding methods")
    print(f"✅ Successfully created {len(layouts_2d)} UMAP visualizations")
    print("✅ Comprehensive comparison plot created")
    if dl_methods:
        print(f"✅ Deep learning methods comparison created ({len(dl_methods)} methods)")
    print("✅ All plots saved and displayed")
    print("\nNow you can compare:")
    print("• Traditional embedding methods vs Deep Learning methods")
    print("• Different NNCLR architectures (CNN, LSTM, Transformer)")
    print("• Overall clustering quality across all approaches")
    
    return embeddings_results, layouts_2d

def plot_deep_learning_comparison(dl_layouts, labels_test, figsize=(15, 5), save_path=None):
    """
    Create a focused comparison plot for just the deep learning (NNCLR) methods.
    
    Parameters:
    -----------
    dl_layouts : dict
        Dictionary with NNCLR method names and their 2D layouts
    labels_test : array-like
        Test labels for color-coding
    figsize : tuple, default=(15, 5)
        Figure size
    save_path : str, optional
        Path to save the figure
    """
    if not dl_layouts:
        print("No deep learning methods available for comparison!")
        return
    
    n_methods = len(dl_layouts)
    fig, axes = plt.subplots(1, n_methods, figsize=figsize)
    
    if n_methods == 1:
        axes = [axes]
    
    # Ensure labels are in the right format
    if hasattr(labels_test, 'values'):
        label_values = labels_test.values.flatten()
    else:
        label_values = np.array(labels_test).flatten()
    
    unique_classes = sorted(np.unique(label_values))
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_classes)))
    
    for i, (method_name, layout) in enumerate(dl_layouts.items()):
        ax = axes[i]
        
        # Create scatter plot
        for class_idx, class_label in enumerate(unique_classes):
            mask = label_values == class_label
            ax.scatter(
                layout[mask, 0], 
                layout[mask, 1], 
                c=[colors[class_idx]],
                label=f'Class {class_label}',
                s=40,
                alpha=0.8,
                edgecolors='white',
                linewidth=0.5
            )
        
        # Clean method name for title and use consistent styling
        clean_name = method_name.replace('NNCLR ', '')
        ax.set_title(f'{clean_name}', fontsize=14, fontweight='bold')
        # Remove axis labels for cleaner look
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.grid(True, alpha=0.3)
    
    # Add legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc='upper center',
        bbox_to_anchor=(0.5, -0.05),
        ncol=len(unique_classes),
        fontsize=11
    )
    
    plt.suptitle('Deep Learning Methods Comparison (NNCLR-based)', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    
    if save_path is None:
        save_path = 'nnclr_methods_comparison.png'
        
    try:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Deep learning comparison saved to: {save_path}")
    except Exception as e:
        print(f"Failed to save deep learning comparison: {e}")
    
    plt.show()

def plot_individual_method(layouts_2d, method_name, labels_test, figsize=(10, 8), save_individual=True, custom_label=None):
    """
    Plot a single embedding method independently with customizable label.
    
    Parameters:
    -----------
    layouts_2d : dict
        Dictionary with all method layouts (from main analysis)
    method_name : str
        Name of the specific method to plot
    labels_test : array-like
        Test labels for color-coding
    figsize : tuple, default=(10, 8)
        Figure size
    save_individual : bool, default=True
        Whether to save the individual plot
    custom_label : str, optional
        Custom label to display on top of the plot. If None, uses method_name
    """
    if method_name not in layouts_2d:
        print(f"Method '{method_name}' not found in layouts.")
        print(f"Available methods: {list(layouts_2d.keys())}")
        return
    
    layout = layouts_2d[method_name]
    
    # Ensure labels are in the right format
    if hasattr(labels_test, 'values'):
        label_values = labels_test.values.flatten()
    else:
        label_values = np.array(labels_test).flatten()
    
    unique_classes = sorted(np.unique(label_values))
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_classes)))
    
    plt.figure(figsize=figsize)
    
    # Create scatter plot with clear class separation
    for class_idx, class_label in enumerate(unique_classes):
        mask = label_values == class_label
        plt.scatter(
            layout[mask, 0], 
            layout[mask, 1], 
            c=[colors[class_idx]],
            label=f'Class {class_label}',
            s=50,
            alpha=0.8,
            edgecolors='white',
            linewidth=0.5
        )
    
    # Determine the label to display
    if custom_label:
        display_label = custom_label
        clean_name = custom_label  # Use custom label for filename too
    else:
        clean_name = method_name.replace('NNCLR ', '')
        display_label = clean_name
    
    # Create title with the label prominently displayed
    plt.title(f'{display_label}', 
              fontsize=18, fontweight='bold', pad=25)
    
    # Add a subtitle if you want to show it's a UMAP projection
    # plt.suptitle('UMAP Projection', fontsize=12, y=0.02)
    
    # Remove axis labels for consistency
    plt.xlabel('')
    plt.ylabel('')
    
    plt.grid(True, alpha=0.3)
    
    # Customize legend
    plt.legend(
        title='Classes',
        title_fontsize=12,
        fontsize=11,
        frameon=True,
        fancybox=True,
        shadow=True,
        bbox_to_anchor=(1.05, 1),
        loc='upper left'
    )
    
    plt.tight_layout()
    
    if save_individual:
        filename = f'individual_{clean_name.lower().replace(" ", "_")}_umap.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Individual plot saved: {filename}")
    
    plt.show()

def plot_multiple_individual_methods(layouts_2d, method_names, labels_test, figsize=(10, 8), custom_labels=None):
    """
    Plot multiple specific methods individually.
    
    Parameters:
    -----------
    layouts_2d : dict
        Dictionary with all method layouts
    method_names : list
        List of method names to plot individually
    labels_test : array-like
        Test labels for color-coding
    figsize : tuple, default=(10, 8)
        Figure size for each plot
    custom_labels : dict, optional
        Dictionary mapping method names to custom labels
        Example: {'PCA': 'Principal Component Analysis', 'CNN': 'Convolutional Neural Network'}
    """
    for method_name in method_names:
        print(f"\nPlotting {method_name}...")
        custom_label = custom_labels.get(method_name) if custom_labels else None
        plot_individual_method(layouts_2d, method_name, labels_test, figsize, custom_label=custom_label)

def plot_all_methods_individually(layouts_2d, labels_test, figsize=(10, 8)):
    """
    Plot every method individually with high-quality settings.
    
    Parameters:
    -----------
    layouts_2d : dict
        Dictionary with all method layouts
    labels_test : array-like
        Test labels for color-coding  
    figsize : tuple, default=(10, 8)
        Figure size for each plot
    """
    print(f"Creating individual plots for all {len(layouts_2d)} methods...")
    print("="*60)
    
    for i, method_name in enumerate(layouts_2d.keys(), 1):
        print(f"[{i}/{len(layouts_2d)}] Plotting {method_name}...")
        plot_individual_method(layouts_2d, method_name, labels_test, figsize)
    
    print("="*60)
    print("✅ All individual plots completed!")

def save_method_data(layouts_2d, embeddings_results, method_name, save_path=None):
    """
    Save the embedding data and UMAP layout for a specific method.
    
    Parameters:
    -----------
    layouts_2d : dict
        Dictionary with UMAP layouts
    embeddings_results : dict  
        Dictionary with original embedding results
    method_name : str
        Name of the method to save
    save_path : str, optional
        Base path for saving files
    """
    if method_name not in layouts_2d:
        print(f"Method '{method_name}' not found.")
        return
    
    if save_path is None:
        save_path = method_name.lower().replace(' ', '_').replace('nnclr ', '')
    
    # Save UMAP layout
    umap_layout = layouts_2d[method_name]
    np.save(f"{save_path}_umap_layout.npy", umap_layout)
    
    # Save original embeddings if available
    if method_name in embeddings_results:
        train_emb, val_emb, test_emb, train_time, inf_time = embeddings_results[method_name]
        
        # Convert to numpy if needed
        if hasattr(test_emb, 'values'):
            test_emb = test_emb.values
            
        np.save(f"{save_path}_original_embedding.npy", test_emb)
        
        # Save timing information
        with open(f"{save_path}_timing.txt", 'w') as f:
            f.write(f"Method: {method_name}\n")
            f.write(f"Training time: {train_time:.4f} seconds\n")
            f.write(f"Inference time: {inf_time:.4f} seconds\n")
        
        print(f"✅ Saved data for {method_name}:")
        print(f"  - UMAP layout: {save_path}_umap_layout.npy")
        print(f"  - Original embedding: {save_path}_original_embedding.npy") 
        print(f"  - Timing info: {save_path}_timing.txt")

# Usage example functions
def example_usage_individual_plots():
    """
    Example showing different ways to create individual plots with custom labels.
    """
    print("""
    # USAGE EXAMPLES - Run after main_robust_umap_analysis():
    
    # 1. Plot with default label (method name)
    plot_individual_method(layouts_2d, 'PCA', ny_test)
    
    # 2. Plot with custom label
    plot_individual_method(layouts_2d, 'PCA', ny_test, custom_label='Principal Component Analysis')
    plot_individual_method(layouts_2d, 'CNN', ny_test, custom_label='Convolutional Neural Network')
    plot_individual_method(layouts_2d, 'LSTM', ny_test, custom_label='Long Short-Term Memory')
    
    # 3. Plot with shorter/cleaner labels
    plot_individual_method(layouts_2d, 'FFT', ny_test, custom_label='Fast Fourier Transform')
    plot_individual_method(layouts_2d, 'Transformer', ny_test, custom_label='Vision Transformer')
    
    # 4. Plot multiple methods with custom labels
    methods_to_plot = ['PCA', 'FFT', 'CNN', 'LSTM']
    custom_labels = {
        'PCA': 'Principal Component Analysis',
        'FFT': 'Fast Fourier Transform', 
        'CNN': 'Convolutional Neural Network',
        'LSTM': 'Long Short-Term Memory'
    }
    plot_multiple_individual_methods(layouts_2d, methods_to_plot, ny_test, custom_labels=custom_labels)
    
    # 5. Plot with paper-ready labels
    plot_individual_method(layouts_2d, 'PCA', ny_test, custom_label='(a) PCA')
    plot_individual_method(layouts_2d, 'CNN', ny_test, custom_label='(b) CNN-NNCLR')
    plot_individual_method(layouts_2d, 'LSTM', ny_test, custom_label='(c) LSTM-NNCLR')
    
    # 6. Plot with dataset-specific labels
    plot_individual_method(layouts_2d, 'PCA', ny_test, custom_label='PCA Embedding - UniMiB Dataset')
    
    # 7. No custom label (uses method name)
    plot_individual_method(layouts_2d, 'Wavelet', ny_test)  # Will show "Wavelet"
    """)

# Additional helper function for common label formats
def plot_with_paper_labels(layouts_2d, labels_test, method_order=None):
    """
    Plot methods with paper-ready labels like (a), (b), (c), etc.
    
    Parameters:
    -----------
    layouts_2d : dict
        Dictionary with all method layouts
    labels_test : array-like
        Test labels for color-coding
    method_order : list, optional
        Specific order for methods. If None, uses alphabetical order.
    """
    if method_order is None:
        method_order = sorted(layouts_2d.keys())
    
    letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o']
    
    print("Creating paper-ready plots with (a), (b), (c) labels...")
    
    for i, method_name in enumerate(method_order[:len(letters)]):
        if method_name in layouts_2d:
            clean_name = method_name.replace('NNCLR ', '')
            paper_label = f'({letters[i]}) {clean_name}'
            print(f"Plotting {paper_label}...")
            plot_individual_method(layouts_2d, method_name, labels_test, 
                                 custom_label=paper_label, figsize=(10, 8))

def plot_with_descriptive_labels(layouts_2d, labels_test):
    """
    Plot methods with full descriptive names for presentations.
    """
    descriptive_labels = {
        'PCA': 'Principal Component Analysis',
        'FFT': 'Fast Fourier Transform',
        'Wavelet': 'Wavelet Transform',
        'LLE': 'Locally Linear Embedding',
        'UMAP': 'Uniform Manifold Approximation',
        'Graph': 'Visibility Graph Analysis',
        'TDA': 'Topological Data Analysis', 
        'Autoencoder': 'Autoencoder Neural Network',
        'CNN': 'Convolutional Neural Network',
        'LSTM': 'Long Short-Term Memory',
        'Transformer': 'Vision Transformer'
    }
    
    print("Creating plots with descriptive labels...")
    for method_name in layouts_2d.keys():
        label = descriptive_labels.get(method_name, method_name)
        print(f"Plotting {label}...")
        plot_individual_method(layouts_2d, method_name, labels_test, custom_label=label)

# Usage example:
# After running: embeddings_results, layouts_2d = main_robust_umap_analysis(...)
# You can then use any of these functions to create individual plots