from PIL import Image, ImageDraw
import argparse
import os
import math
from itertools import product

def draw_shape(draw, shape, center, shape_radius, shape_color):
    # Draw the specified shape
    if shape == "circle":
        draw.ellipse([(center[0] - shape_radius, center[1] - shape_radius),
                    (center[0] + shape_radius, center[1] + shape_radius)],
                    fill=shape_color)
    elif shape == "rectangle":
        draw.rectangle([(center[0] - shape_radius, center[1] - shape_radius),
                        (center[0] + shape_radius, center[1] + shape_radius)],
                       fill=shape_color)

# Modify the create_image function to take RGB values for colors
def create_image(shape_list, rgb_values_list, image_size, output_folder):
    # Define the position and size of the shapes
    center = (image_size // 2, image_size // 2)
    
    # Set shape_radius to be 50% of the image size
    shape_radius = image_size // 4

    # Generate all combinations of shapes and RGB values
    shape_rgb_combinations = product(shape_list, rgb_values_list)

    # Iterate over each shape-RGB pair and create an image
    for shape, rgb_values in shape_rgb_combinations:
        # Create a sub-folder for each shape-RGB pair
        subfolder = os.path.join(output_folder, f"{shape}_{rgb_values}")
        os.makedirs(subfolder, exist_ok=True)

        # Convert RGB values to tuple
        color = tuple(map(int, rgb_values.split(',')))

        # Create a blank image with a white background for each shape-RGB pair
        image = Image.new("RGB", (image_size, image_size), "white")
        draw = ImageDraw.Draw(image)

        # Draw the specified shape for the current pair
        draw_shape(draw, shape, center, shape_radius, color)

        # Save the drawn image inside the sub-folder with a unique filename
        image.save(os.path.join(subfolder, f"{shape}_{rgb_values}.png"))

def main():
    parser = argparse.ArgumentParser(description="Draw 2D shapes and save the image.")
    parser.add_argument("image_size", type=int, help="Size of the image")
    parser.add_argument("--shapes", nargs='+', type=str, help="List of shape names")
    parser.add_argument("--rgb_values", nargs='+', type=str, help="List of RGB values as 'R,G,B'")
    parser.add_argument("--out", default="output_images", help="Output folder for saving images")

    args = parser.parse_args()

    create_image(args.shapes, args.rgb_values, args.image_size, args.out)

if __name__ == "__main__":
    main()
