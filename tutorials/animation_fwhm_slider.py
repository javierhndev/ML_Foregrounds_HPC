#!/usr/bin/env python3
"""
Interactive Animation: Q_pred and U_pred with FWHM Target Slider

This script creates an interactive animation showing how Q_pred and U_pred 
change as a function of fwhm_target using matplotlib's interactive features.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import scipy.ndimage
from matplotlib.widgets import Slider, Button
import warnings
warnings.filterwarnings('ignore')

# Set up matplotlib for interactive plotting
plt.ion()

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

def create_animation(val_img, val_res, model=None, fwhm_target_init=40):
    """Create interactive animation with slider."""
    
    # Create dummy data if not provided
    Q_HR = val_img[0, 5, 0]
    U_HR = val_img[0, 6, 0]

    # Initialize with default fwhm_target
    val_img = update_val_img(fwhm_target_init, val_res.item(), val_img=val_img, Q_HR=Q_HR, U_HR=U_HR)
    pred_imgs = get_predictions(val_img, val_res, model)
    
    # Create the figure and subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10), constrained_layout=True)
    
    # Initial plots
    idx_order = [3, 5, 4, 6]  # q_lr, q_hr, u_lr, u_hr
    title_arr = [r"$Q_{LR}$", r"$Q_{HR}$", r"$U_{LR}$", r"$U_{HR}$", r'$Q_{pred}$', r'$U_{pred}$']
    
    images = []
    for i in range(4):
        row, col = i // 2, i % 2
        im = axes[row, col].imshow(val_img[0, idx_order[i], 0], cmap='RdBu_r')
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
    
    # Add slider
    ax_slider = plt.axes([0.2, 0.02, 0.6, 0.03])
    slider = Slider(
        ax_slider, 'FWHM Target', 20, 80, valinit=fwhm_target_init, valstep=5
    )
    
    def update(val):
        """Update function for the slider."""
        fwhm_target = slider.val
        
        # Update validation image
        val_img_updated = update_val_img(fwhm_target, val_img=val_img, Q_HR=Q_HR, U_HR=U_HR)
        
        # Get new predictions
        pred_imgs_new = get_predictions(val_img_updated, model)
        
        # Update input images
        for i, idx in enumerate(idx_order):
            images[i].set_array(val_img_updated[0, idx, 0])
        
        # Update prediction images
        images[4].set_array(pred_imgs_new[0, 0, 0])
        images[5].set_array(pred_imgs_new[0, 1, 0])
        
        # Update title
        plt.suptitle(f'FWHM Target: {fwhm_target}', fontsize=16, y=0.95)
        
        fig.canvas.draw_idle()
    
    slider.on_changed(update)
    
    # Add reset button
    ax_reset = plt.axes([0.8, 0.02, 0.1, 0.03])
    button_reset = Button(ax_reset, 'Reset')
    
    def reset(event):
        slider.reset()
    
    button_reset.on_clicked(reset)
    
    plt.show()
    
    return fig, slider, button_reset

def create_advanced_animation(val_img, val_res, model=None, fwhm_target_init=40):
    """Create advanced animation with additional controls."""
    
    # Create dummy data if not provided
    Q_HR = val_img[0, 5, 0]
    U_HR = val_img[0, 6, 0]
    
    # Initialize with default values
    fwhm_target_init = 40
    fwhm_init_init = 15
    val_img = update_val_img(fwhm_target_init, val_res.item(), val_img=val_img, Q_HR=Q_HR, U_HR=U_HR)
    pred_imgs = get_predictions(val_img, val_res, model)
    
    # Create the figure and subplots
    fig, axes = plt.subplots(2, 4, figsize=(20, 10), constrained_layout=True)
    
    # Input images
    titles = [r"$Q_{LR}$", r"$Q_{HR}$", r"$U_{LR}$", r"$U_{HR}$"]
    idx_order = [3, 5, 4, 6]
    
    images = []
    for i in range(4):
        row, col = i // 2, i % 2
        im = axes[row, col].imshow(val_img[0, idx_order[i], 0], cmap='RdBu_r')
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
        diff = pred_imgs[0, i, 0] - val_img[0, 5+i, 0].numpy()
        im = axes[i, 3].imshow(diff, cmap='RdBu_r')
        axes[i, 3].set_title(diff_titles[i], fontsize=14)
        axes[i, 3].axis('off')
        plt.colorbar(im, ax=axes[i, 3], shrink=0.8)
        images.append(im)
    
    plt.suptitle(f'FWHM Target: {fwhm_target_init}, FWHM Init: {fwhm_init_init}', fontsize=16, y=0.95)
    
    # Add sliders
    ax_slider_fwhm = plt.axes([0.2, 0.02, 0.3, 0.03])
    slider_fwhm = Slider(
        ax_slider_fwhm, 'FWHM Target', 20, 80, valinit=fwhm_target_init, valstep=5
    )
    
    ax_slider_init = plt.axes([0.6, 0.02, 0.3, 0.03])
    slider_init = Slider(
        ax_slider_init, 'FWHM Init', 5, 30, valinit=fwhm_init_init, valstep=1
    )
    
    def update(val):
        """Update function for the sliders."""
        fwhm_target = slider_fwhm.val
        fwhm_init = slider_init.val
        
        # Update validation image
        val_img_updated = update_val_img(fwhm_target, fwhm_init, val_img=val_img, Q_HR=Q_HR, U_HR=U_HR)
        
        # Get new predictions
        pred_imgs_new = get_predictions(val_img_updated, model)
        
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
        
        fig.canvas.draw_idle()
    
    slider_fwhm.on_changed(update)
    slider_init.on_changed(update)
    
    # Add reset button
    ax_reset = plt.axes([0.9, 0.02, 0.08, 0.03])
    button_reset = Button(ax_reset, 'Reset')
    
    def reset(event):
        slider_fwhm.reset()
        slider_init.reset()
    
    button_reset.on_clicked(reset)
    
    plt.show()
    
    return fig, slider_fwhm, slider_init, button_reset

if __name__ == "__main__":
    print("Creating interactive animation...")
    print("Use the slider to change FWHM Target and see how predictions change.")
    print("Note: This uses dummy data. Replace with your actual model and data.")
    
    # Create basic animation
    fig, slider, button = create_animation()
    
    # Uncomment to create advanced animation with additional controls
    # fig_adv, slider_fwhm, slider_init, button_reset = create_advanced_animation()
    
    print("Animation created! Use the slider to interact with the plot.")
    print("Close the plot window to exit.")
    
    # Keep the plot open
    plt.show(block=True) 