import cv2
import numpy as np
from PIL import Image
from scipy.signal import convolve2d
from skimage.filters import gabor_kernel
import matplotlib.pyplot as plt

class ContextAwareEnhancement:
    """
    Enhances and binarizes a latent fingerprint for maximum clarity.
    1. Uses the orientation flow to guide Gabor filters.
    2. Uses a fixed, reliable frequency to avoid errors from estimation.
    3. Applies an adaptive threshold to create a clean, binary (B&W) image.
    """
    def __call__(self, latent_img, orientation_flow_img):
        # --- 1. Ensure images have the same dimensions and create mask ---
        if latent_img.size != orientation_flow_img.size:
            orientation_flow_img = orientation_flow_img.resize(latent_img.size, Image.Resampling.NEAREST)

        latent_np = np.array(latent_img.convert('L'), dtype=np.uint8)
        mask = (latent_np > 0)

        # --- 2. Decode the Orientation Flow Image ---
        orientation_rgb_np = np.array(orientation_flow_img)
        orientation_hsv = cv2.cvtColor(orientation_rgb_np, cv2.COLOR_RGB2HSV)
        hue = orientation_hsv[:, :, 0]
        orientation_rad = (hue * np.pi / 180) - (np.pi / 2)

        # --- 3. Apply Context-Aware Gabor Filtering ---
        latent_float_np = latent_np.astype(np.float32)
        enhanced_np = np.zeros_like(latent_float_np)

        block_size = 16
        # Use a reliable, fixed frequency instead of estimating it.
        ridge_freq = 0.1

        # Pre-create a bank of Gabor filters
        gabor_kernels = [gabor_kernel(frequency=ridge_freq, theta=theta)
                         for theta in np.arange(0, np.pi, np.pi / 16)]

        for i in range(0, latent_np.shape[0], block_size):
            for j in range(0, latent_np.shape[1], block_size):
                mask_block = mask[i:i+block_size, j:j+block_size]
                if not np.any(mask_block):
                    continue

                img_block = latent_float_np[i:i+block_size, j:j+block_size]
                angle_block = orientation_rad[i:i+block_size, j:j+block_size]

                dominant_angle = np.median(angle_block[mask_block])

                kernel_idx = np.argmin(np.abs(dominant_angle - np.arange(0, np.pi, np.pi / 16)))
                chosen_kernel = gabor_kernels[kernel_idx].real

                enhanced_block = convolve2d(img_block, chosen_kernel, mode='same', boundary='symm')
                enhanced_np[i:i+block_size, j:j+block_size][mask_block] = enhanced_block[mask_block]

        # Normalize the enhanced grayscale image
        if (enhanced_np.max() - enhanced_np.min()) > 0:
            enhanced_np = (255 * (enhanced_np - enhanced_np.min()) / (enhanced_np.max() - enhanced_np.min())).astype(np.uint8)
        else:
            enhanced_np = enhanced_np.astype(np.uint8)

        # --- 4. Binarize the Enhanced Image for Maximum Clarity ---
        # Apply an adaptive threshold to the masked region.
        # This will force every pixel to be either black or white.
        binary_img = cv2.adaptiveThreshold(
            enhanced_np,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11, # Block size for thresholding
            2   # Constant subtracted from the mean
        )

        # Ensure the background remains black
        binary_img[mask == 0] = 0

        # Convert back to a 3-channel RGB PIL image for the model
        return Image.fromarray(binary_img).convert('RGB')


if __name__ == "__main__":

    try:
        latent_img_path = '/content/drive/MyDrive/latent_dataset/IITD_latent_ROI/full/10_LI_Card_1.jpg'
        orientation_flow_img_path = '/content/drive/MyDrive/latent_dataset/orientation_flow/10_LI_Card_1_OF.jpg'

        # 1. Load the sample input images
        latent_img = Image.open(latent_img_path)
        orientation_flow_img = Image.open(orientation_flow_img_path)

        # 2. Create an instance of the enhancer and process the images
        print("Enhancing fingerprint... please wait.")
        #enhancer = FixedOrientationEnhancement()
        enhancer = ContextAwareEnhancement()
        enhanced_image = enhancer(latent_img, orientation_flow_img)
        print("Enhancement complete.")

        # 3. Save the output image (optional)
        enhanced_image.save("enhanced_output.jpg")
        print("Enhanced image saved as 'enhanced_output.jpg'")

        # 4. Visualize the comparison
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        axes[0].imshow(latent_img, cmap='gray')
        axes[0].set_title('Original Latent Print')
        axes[0].axis('off')

        axes[1].imshow(enhanced_image)
        axes[1].set_title('Enhanced Output (Fixed Orientation)')
        axes[1].axis('off')

        fig.suptitle('Fingerprint Enhancement Comparison', fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig("enhancement_comparison.png")
        print("Comparison image saved as 'enhancement_comparison.png'")
        plt.show()

    except FileNotFoundError:
        print("\nError: One or both of the specified image files were not found.")
        print("Please ensure the paths in the script point to your actual image files.")
        print(f"Attempted to open: '{latent_img_path}' and '{orientation_flow_img_path}'")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        print("Please ensure you have all necessary libraries installed (Pillow, numpy, matplotlib, scikit-image, opencv-python).")
        print("You can install them using: pip install Pillow numpy matplotlib scikit-image opencv-python")