import cv2
import os

def convert_video_to_images(video_path, output_folder, num_images):
    video_capture = cv2.VideoCapture(video_path)
    success, image = video_capture.read()
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    step = total_frames // num_images
    count = 0
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    while success:
        if count % step == 0:
            # Generate unique filename based on current frame count
            unique_filename = f"{count}_{os.path.splitext(os.path.basename(video_path))[0]}.jpg"
            cv2.imwrite(os.path.join(output_folder, unique_filename), image)
        success, image = video_capture.read()
        count += 1
    video_capture.release()
    cv2.destroyAllWindows()

def batch_convert_videos(video_folder, output_folder, num_images):
    video_files = [f for f in os.listdir(video_folder) if f.endswith(".mp4")]
    for video_file in video_files:
        video_path = os.path.join(video_folder, video_file)
        output_subfolder = os.path.join(output_folder, video_file.split(".")[0])
        if not os.path.exists(output_subfolder):
            os.makedirs(output_subfolder)
        convert_video_to_images(video_path, output_subfolder, num_images)

video_folder = "videos"
output_folder = "dataset"
num_images = 49

batch_convert_videos(video_folder, output_folder, num_images)
