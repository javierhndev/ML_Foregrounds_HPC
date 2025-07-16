#!/usr/bin/env python3
"""
Jupyter-Compatible Animation: Q_pred and U_pred with FWHM Target Slider

This version uses ipywidgets instead of matplotlib widgets for better
interactivity in Jupyter notebooks.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import scipy.ndimage
import ipywidgets as widgets
from IPython.display import display, clear_output
import warnings
warnings.filterwarnings('ignore')

def create_lr_images(fwhm_target, fwhm_init=15, Q_HR=None, U_HR=None):
    """Create low-resolution images based on fwhm_target."""
    if Q_HR is None or U_HR is None:
        # Create dummy data if not provided
        size = 80
        np.random.seed(42)
        Q_HR = np.random.randn(size, size) * 0.1
        U_HR = np.random.randn(size, size) * 0.1
    
    fwhm = np.sqrt(fwhm_target**2 - fwhm_init**2)
    fwhm = fwhm / (20*fwhm_init/80)
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    
    Q_LR = scipy.ndimage.gaussian_filter(Q_HR, sigma=sigma)
    U_LR = scipy.ndimage.gaussian_filter(U_HR, sigma=sigma)
    
    return Q_LR, U_LR

def update_val_img(fwhm_target, fwhm_init=15, val_img=None, Q_HR=None, U_HR=None):
    """Update validation image with new low-resolution images."""
    Q_LR, U_LR = create_lr_images(fwhm_target, fwhm_init, Q_HR, U_HR)
    val_img[0, 3, 0] = torch.from_numpy(Q_LR)
    val_img[0, 4, 0] = torch.from_numpy(U_LR)
    return val_img

def get_predictions(val_img, val_res, model=None):
    """Get predictions from model or create dummy predictions."""
    if model is not None:
        with torch.no_grad():
            pred_imgs = model(val_img, val_res).detach().cpu().numpy()
    else:
        # Create dummy predictions for demonstration
        size = val_img.shape[-1]
        pred_imgs = np.random.randn(1, 2, 1, size, size) * 0.05
        pred_imgs[0, 0, 0] = val_img[0, 3, 0].numpy() + np.random.randn(size, size) * 0.01
        pred_imgs[0, 1, 0] = val_img[0, 4, 0].numpy() + np.random.randn(size, size) * 0.01
    
    return pred_imgs

def create_simple_jupyter_animation(val_img, val_res, model=None):
    """Create a simple animation with just the slider and plot."""
    
    # Extract Q_HR and U_HR from val_img
    Q_HR = val_img[0, 5, 0]
    U_HR = val_img[0, 6, 0]
    
    # Create output widget for the plot
    output = widgets.Output()
    
    # Create slider
    slider = widgets.IntSlider(
        value=40,
        min=val_res.item(),
        max=5*val_res.item(),
        step=5,
        description='FWHM Target:',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='400px')
    )
    
    def update_plot(change):
        """Update function for the slider."""
        with output:
            clear_output(wait=True)
            
            fwhm_target = change['new']
            
            # Update validation image
            val_img_updated = update_val_img(fwhm_target, val_res.item(), val_img=val_img, Q_HR=Q_HR, U_HR=U_HR)
            
            # Get new predictions
            pred_imgs = get_predictions(val_img_updated, val_res, model)
            
            # Create the plot
            fig, axes = plt.subplots(2, 3, figsize=(15, 10), constrained_layout=True)
            
            # Input images
            idx_order = [3, 5, 4, 6]  # q_lr, q_hr, u_lr, u_hr
            title_arr = [r"$Q_{LR}$", r"$Q_{HR}$", r"$U_{LR}$", r"$U_{HR}$", r'$Q_{pred}$', r'$U_{pred}$']
            
            for i in range(4):
                row, col = i // 2, i % 2
                im = axes[row, col].imshow(val_img_updated[0, idx_order[i], 0], cmap='viridis')
                axes[row, col].set_title(title_arr[i], fontsize=16)
                axes[row, col].axis('off')
                plt.colorbar(im, ax=axes[row, col], shrink=0.8)
            
            # Prediction plots
            im_pred_q = axes[0, 2].imshow(pred_imgs[0, 0, 0], cmap='viridis')
            im_pred_u = axes[1, 2].imshow(pred_imgs[0, 1, 0], cmap='viridis')
            axes[0, 2].set_title(title_arr[4], fontsize=16)
            axes[1, 2].set_title(title_arr[5], fontsize=16)
            axes[0, 2].axis('off')
            axes[1, 2].axis('off')
            plt.colorbar(im_pred_q, ax=axes[0, 2], shrink=0.8)
            plt.colorbar(im_pred_u, ax=axes[1, 2], shrink=0.8)
            
            plt.suptitle(f'FWHM Target: {fwhm_target}', fontsize=16, y=1.0)
            plt.show()
    
    slider.observe(update_plot, names='value')
    
    # Trigger initial plot
    update_plot({'new': slider.value})
    
    return slider, output

def create_jupyter_animation(val_img, val_res, model=None, fwhm_target_init=40):
    """Create interactive animation using ipywidgets for Jupyter notebooks."""
    
    # Extract Q_HR and U_HR from val_img
    Q_HR = val_img[0, 5, 0]
    U_HR = val_img[0, 6, 0]
    
    # Initialize with default fwhm_target
    val_img_updated = update_val_img(fwhm_target_init, val_res.item(), val_img=val_img, Q_HR=Q_HR, U_HR=U_HR)
    pred_imgs = get_predictions(val_img_updated, val_res, model)
    
    # Create the figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10), constrained_layout=True)
    
    # Initial plots
    idx_order = [3, 5, 4, 6]  # q_lr, q_hr, u_lr, u_hr
    title_arr = [r"$Q_{LR}$", r"$Q_{HR}$", r"$U_{LR}$", r"$U_{HR}$", r'$Q_{pred}$', r'$U_{pred}$']
    
    images = []
    for i in range(4):
        row, col = i // 2, i % 2
        im = axes[row, col].imshow(val_img_updated[0, idx_order[i], 0], cmap='RdBu_r')
        axes[row, col].set_title(title_arr[i], fontsize=14)
        axes[row, col].axis('off')
        plt.colorbar(im, ax=axes[row, col], shrink=0.8)
        images.append(im)
    
    # Prediction plots
    im_pred_q = axes[0, 2].imshow(pred_imgs[0, 0, 0], cmap='RdBu_r')
    im_pred_u = axes[1, 2].imshow(pred_imgs[0, 1, 0], cmap='RdBu_r')
    axes[0, 2].set_title(title_arr[4], fontsize=14)
    axes[1, 2].set_title(title_arr[5], fontsize=14)
    axes[0, 2].axis('off')
    axes[1, 2].axis('off')
    plt.colorbar(im_pred_q, ax=axes[0, 2], shrink=0.8)
    plt.colorbar(im_pred_u, ax=axes[1, 2], shrink=0.8)
    images.extend([im_pred_q, im_pred_u])
    
    plt.suptitle(f'FWHM Target: {fwhm_target_init}', fontsize=16, y=0.95)
    
    # Create ipywidgets slider
    slider = widgets.IntSlider(
        value=fwhm_target_init,
        min=20,
        max=80,
        step=5,
        description='FWHM Target:',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='500px')
    )
    
    def update_plot(change):
        """Update function for the slider."""
        fwhm_target = change['new']
        
        # Update validation image
        val_img_updated = update_val_img(fwhm_target, 15, val_img=val_img, Q_HR=Q_HR, U_HR=U_HR)
        
        # Get new predictions
        pred_imgs_new = get_predictions(val_img_updated, val_res, model)
        
        # Update input images
        for i, idx in enumerate(idx_order):
            images[i].set_array(val_img_updated[0, idx, 0])
        
        # Update prediction images
        images[4].set_array(pred_imgs_new[0, 0, 0])
        images[5].set_array(pred_imgs_new[0, 1, 0])
        
        # Update title
        plt.suptitle(f'FWHM Target: {fwhm_target}', fontsize=16, y=0.95)
        
        # Force redraw
        fig.canvas.draw()
    
    slider.observe(update_plot, names='value')
    
    # Create reset button
    reset_button = widgets.Button(
        description='Reset',
        button_style='info',
        layout=widgets.Layout(width='100px')
    )
    
    def reset_plot(b):
        slider.value = fwhm_target_init
    
    reset_button.on_click(reset_plot)
    
    # Create layout
    controls = widgets.HBox([slider, reset_button])
    
    return fig, controls, slider, reset_button

def create_jupyter_advanced_animation(val_img, val_res, model=None, fwhm_target_init=40):
    """Create advanced animation with multiple controls using ipywidgets."""
    
    # Extract Q_HR and U_HR from val_img
    Q_HR = val_img[0, 5, 0]
    U_HR = val_img[0, 6, 0]
    
    # Initialize with default values
    val_img_updated = update_val_img(fwhm_target_init, val_res.item(), val_img=val_img, Q_HR=Q_HR, U_HR=U_HR)
    pred_imgs = get_predictions(val_img_updated, val_res, model)
    
    # Create the figure and subplots
    fig, axes = plt.subplots(2, 4, figsize=(20, 10), constrained_layout=True)
    
    # Input images
    titles = [r"$Q_{LR}$", r"$Q_{HR}$", r"$U_{LR}$", r"$U_{HR}$"]
    idx_order = [3, 5, 4, 6]
    
    images = []
    for i in range(4):
        row, col = i // 2, i % 2
        im = axes[row, col].imshow(val_img_updated[0, idx_order[i], 0], cmap='RdBu_r')
        axes[row, col].set_title(titles[i], fontsize=14)
        axes[row, col].axis('off')
        plt.colorbar(im, ax=axes[row, col], shrink=0.8)
        images.append(im)
    
    # Predictions
    pred_titles = [r'$Q_{pred}$', r'$U_{pred}$']
    for i in range(2):
        im = axes[i, 2].imshow(pred_imgs[0, i, 0], cmap='RdBu_r')
        axes[i, 2].set_title(pred_titles[i], fontsize=14)
        axes[i, 2].axis('off')
        plt.colorbar(im, ax=axes[i, 2], shrink=0.8)
        images.append(im)
    
    # Difference plots
    diff_titles = [r'$Q_{pred} - Q_{HR}$', r'$U_{pred} - U_{HR}$']
    for i in range(2):
        diff = pred_imgs[0, i, 0] - val_img_updated[0, 5+i, 0].numpy()
        im = axes[i, 3].imshow(diff, cmap='RdBu_r')
        axes[i, 3].set_title(diff_titles[i], fontsize=14)
        axes[i, 3].axis('off')
        plt.colorbar(im, ax=axes[i, 3], shrink=0.8)
        images.append(im)
    
    plt.suptitle(f'FWHM Target: {fwhm_target_init}, FWHM Init: {val_res.item()}', fontsize=16, y=0.95)
    
    # Create ipywidgets sliders
    slider_fwhm = widgets.IntSlider(
        value=fwhm_target_init,
        min=20,
        max=80,
        step=5,
        description='FWHM Target:',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='300px')
    )
    
    slider_init = widgets.IntSlider(
        value=val_res.item(),
        min=5,
        max=30,
        step=1,
        description='FWHM Init:',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='300px')
    )
    
    def update_plot(change):
        """Update function for the sliders."""
        fwhm_target = slider_fwhm.value
        fwhm_init = slider_init.value
        
        # Update validation image
        val_img_updated = update_val_img(fwhm_target, fwhm_init, val_img=val_img, Q_HR=Q_HR, U_HR=U_HR)
        
        # Get new predictions
        pred_imgs_new = get_predictions(val_img_updated, val_res, model)
        
        # Update input images
        for i, idx in enumerate(idx_order):
            images[i].set_array(val_img_updated[0, idx, 0])
        
        # Update prediction images
        images[4].set_array(pred_imgs_new[0, 0, 0])
        images[5].set_array(pred_imgs_new[0, 1, 0])
        
        # Update difference images
        for i in range(2):
            diff = pred_imgs_new[0, i, 0] - val_img_updated[0, 5+i, 0].numpy()
            images[6+i].set_array(diff)
        
        # Update title
        plt.suptitle(f'FWHM Target: {fwhm_target}, FWHM Init: {fwhm_init}', fontsize=16, y=0.95)
        
        # Force redraw
        fig.canvas.draw()
    
    slider_fwhm.observe(update_plot, names='value')
    slider_init.observe(update_plot, names='value')
    
    # Create reset button
    reset_button = widgets.Button(
        description='Reset',
        button_style='info',
        layout=widgets.Layout(width='100px')
    )
    
    def reset_plot(b):
        slider_fwhm.value = fwhm_target_init
        slider_init.value = val_res.item()
    
    reset_button.on_click(reset_plot)
    
    # Create layout
    controls = widgets.VBox([
        widgets.HBox([slider_fwhm, slider_init]),
        widgets.HBox([reset_button])
    ])
    
    return fig, controls, slider_fwhm, slider_init, reset_button

if __name__ == "__main__":
    print("Jupyter-compatible animation functions created!")
    print("Import these functions in your Jupyter notebook for interactive sliders.") 