import os
from PIL import Image
import argparse
import tqdm

def downscale_image(image_path, output_path, scale_factor, algorithm):
    with Image.open(image_path) as img:
        new_size = (img.width // scale_factor, img.height // scale_factor)
        resized_img = img.resize(new_size, algorithm)
        resized_img.save(output_path)

def process_directory(input_dir, output_dir, scale_factor, algorithm):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff')

    for filename in tqdm.tqdm(os.listdir(input_dir)):
        if filename.lower().endswith(supported_formats):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            downscale_image(input_path, output_path, scale_factor, algorithm)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Downscale all images in the images and semantic directory.")
    parser.add_argument("input_dir", help="Path to the dataset directory.")

    args = parser.parse_args()
    input_dir = args.input_dir
    images_dir = input_dir + "/images"
    semantic_dir = input_dir + "/semantic"
    for i in [2, 4, 8]:
        process_directory(images_dir, "{}/images_{}".format(input_dir, i), i, Image.LANCZOS)
        process_directory(semantic_dir, "{}/semantic_{}".format(input_dir, i), i, Image.NEAREST)