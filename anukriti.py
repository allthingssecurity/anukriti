import cv2
import numpy as np
import insightface
import os
import torch
from insightface.app import FaceAnalysis
from moviepy.editor import *

def create_directories():
    dir_names = ['frames', 'output']
    for dir_name in dir_names:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
            print(f"Directory {dir_name} created.")
        else:
            print(f"Directory {dir_name} already exists.")

def initialize_insightface(model_path):
    app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))
    swapper = insightface.model_zoo.get_model(model_path, download=False, download_zip=False)
    return app, swapper

from concurrent.futures import ThreadPoolExecutor

def process_frame(frame, frame_count, output_folder, app, swapper, source_face, frame_photo_face_features):
    faces = app.get(frame)
    for face in faces:
        face_features = face.normed_embedding
        similarity = np.dot(face_features, frame_photo_face_features.T)
        if similarity > 0.5:
            frame = swapper.get(frame, face, source_face[0], paste_back=True)

    frame_filename = os.path.join(output_folder, f"frame_{frame_count}.png")
    cv2.imwrite(frame_filename, frame)

def read_and_process_video(video_path, output_folder, app, swapper, source_face_dict):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Video has ended.")
            break

        faces = app.get(frame)
        for face in faces:
            face_features = face.normed_embedding
            for frame_photo_name, (source_face, frame_photo_face_features) in source_face_dict.items():
                similarity = np.dot(face_features, frame_photo_face_features.T)
                if similarity > 0.5:
                    frame = swapper.get(frame, face, source_face[0], paste_back=True)

        frame_filename = os.path.join(output_folder, f"frame_{frame_count}.png")
        cv2.imwrite(frame_filename, frame)
        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()
def reassemble_video(output_folder, original_video_path, final_output_folder):
    # Get frame rate of original video
    cap = cv2.VideoCapture(original_video_path)
    if not cap.isOpened():
        print("Error: Could not open original video.")
        return
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
    cap.release()

    # Get dimensions of the first frame
    first_frame_path = os.path.join(output_folder, "frame_0.png")
    img = cv2.imread(first_frame_path)
    if img is None:
        print("Error: Could not read the first frame.")
        return
    height, width, _ = img.shape

    # Initialize VideoWriter with the path in the final_output_folder
    output_file_path = os.path.join(final_output_folder, 'output_video.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('temp_video.mp4', fourcc, frame_rate, (width, height))

    frame_count = len([name for name in os.listdir(output_folder) if os.path.isfile(os.path.join(output_folder, name))])

    # Write frames into the video
    for i in range(frame_count):
        frame_path = os.path.join(output_folder, f"frame_{i}.png")
        frame = cv2.imread(frame_path)
        if frame is None:
            print(f"Error: Could not read frame_{i}. Skipping this frame.")
            continue
        out.write(frame)

    out.release()

    # Extract audio from the original video
    video = VideoFileClip(original_video_path)
    audio = video.audio
    audio.write_audiofile("temp_audio.mp3")

    # Merge audio and reassembled video
    final_video = VideoFileClip("temp_video.mp4")
    final_video = final_video.set_audio(AudioFileClip("temp_audio.mp3"))
    final_video.write_videofile(output_file_path)  # Changed to output_file_path

    # Remove temporary files
    os.remove("temp_video.mp4")
    os.remove("temp_audio.mp3")

    print("Video reassembled successfully with audio.")

def clear_gpu_memory(app, swapper):
    del app
    del swapper
    torch.cuda.empty_cache()

def main():
    create_directories()

    app, swapper = initialize_insightface('/content/drive/MyDrive/inswapper_128 (1).onnx')

    # Dictionary to hold multiple source and frame photos
    source_face_dict = {}

    # Example with two source-frame photo pairs
    source_img1 = cv2.imread('/content/drive/MyDrive/khujli/ji_1.jpg')
    source_face1 = app.get(source_img1)
    frame_photo1 = cv2.imread('/content/drive/MyDrive/khujli/amol.jpg')
    frame_photo_face1 = app.get(frame_photo1)
    frame_photo_face_features1 = frame_photo_face1[0].normed_embedding
    source_face_dict["frame1"] = (source_face1, frame_photo_face_features1)

    source_img2 = cv2.imread('/content/drive/MyDrive/khujli/kat.jpg')
    source_face2 = app.get(source_img2)
    frame_photo2 = cv2.imread('/content/drive/MyDrive/khujli/actress.jpg')
    frame_photo_face2 = app.get(frame_photo2)
    frame_photo_face_features2 = frame_photo_face2[0].normed_embedding
    source_face_dict["frame2"] = (source_face2, frame_photo_face_features2)

    output_folder = '/content/frames'
    video_path = '/content/drive/MyDrive/khujli/amol1.mp4'

    read_and_process_video(video_path, output_folder, app, swapper, source_face_dict)

    original_video_path = '/content/drive/MyDrive/khujli/amol1.mp4'
    final_output_folder = '/content/output'
    reassemble_video(output_folder, original_video_path, final_output_folder)

    # Remove model from GPU
    clear_gpu_memory(app, swapper)

if __name__ == "__main__":
    main()
