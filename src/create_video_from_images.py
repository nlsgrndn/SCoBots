import cv2
import os

def images_to_video(image_folder, video_name, fps):
    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    images.sort()
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    cv2.destroyAllWindows()
    video.release()

# Replace 'image_folder' with the path to your folder containing images
# Replace 'output_video.mp4' with the desired output video name
# Set 'fps' to the desired frames per second for the video
images_to_video(f'tmp/', f'boxing_agent_video.mp4', fps=30)
