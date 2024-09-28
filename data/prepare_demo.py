import cv2
import os
import random

def create_video_from_images(image_folder, output_video, fps=10/7):
    # Get all image files from the folder
    images = [img for img in os.listdir(image_folder) if img.endswith(".png") or img.endswith(".jpg")]
    images = random.sample(images, k=10)
    images.sort()  # Make sure they are sorted correctly

    # Read the first image to get the dimensions
    first_image_path = os.path.join(image_folder, images[0])
    first_image = cv2.imread(first_image_path)
    height, width, layers = first_image.shape

    # Define the video codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for mp4
    video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    # Write each image to the video
    for image_file in images:
        image_path = os.path.join(image_folder, image_file)
        image = cv2.imread(image_path)
        video.write(image)

    # Release the video writer
    video.release()
    print(f"Video saved as {output_video}")

# Example usage
image_folder = 'data/raw/zip/SCface_database/surveillance_cameras_all'
output_video = 'video.mp4'
create_video_from_images(image_folder, output_video)
