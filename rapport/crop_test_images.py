from PIL import Image

def crop_test_images(image_path, output_path, downscale_factor=1, extract_truth=False):
    with Image.open(image_path) as img:
        width, height = img.size
        result_width = width // 3
        
        img_cropped = img.crop((result_width if extract_truth else 0, 0, result_width * 2 if extract_truth else result_width, height))
        if downscale_factor > 1:
            img_cropped = img_cropped.resize(
                (result_width // downscale_factor, height // downscale_factor),
                Image.LANCZOS
            )
        img_cropped.save(output_path, format='PNG')

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Crop test images to 1/3 width and downscale if needed.")
    parser.add_argument("image_path", type=str, help="Path to the input image.")
    parser.add_argument("output_path", type=str, help="Path to save the cropped image.")
    parser.add_argument("--downscale_factor", type=int, default=1, help="Factor by which to downscale the image.")
    parser.add_argument("--truth", type=bool, default=False, help="Whether to extract the ground truth image.")

    args = parser.parse_args()
    
    crop_test_images(args.image_path, args.output_path, args.downscale_factor, args.truth)
