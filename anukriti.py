import cv2
import numpy as np
import insightface
import os
import torch
from numpy.linalg import norm
from insightface.app import FaceAnalysis
from moviepy.editor import *

def create_directories(output_folder, final_output_folder):
    dir_names = [output_folder, final_output_folder]
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

        # Calculate cosine similarity
        similarity = np.dot(face_features, frame_photo_face_features.T) / (norm(face_features) * norm(frame_photo_face_features))

        # Adjust the threshold for cosine similarity (you may need to experiment with this value)
        if similarity > 0.3:  
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

        all_faces_similar = True  # Initialize flag to True for each frame

        faces = app.get(frame)
        for face in faces:
            face_features = face.normed_embedding
            face_similar = False  # Initialize a flag for individual face similarity

            for frame_photo_name, (source_face, frame_photo_face_features) in source_face_dict.items():
                #similarity = np.dot(face_features, frame_photo_face_features.T)
                similarity = np.dot(face_features, frame_photo_face_features.T) / (norm(face_features) * norm(frame_photo_face_features))
                if similarity > 0.3:
                    frame = swapper.get(frame, face, source_face[0], paste_back=True)
                    face_similar = True  # Set individual face similarity flag to True
                    break  # No need to check further for this face

            if not face_similar:  # If this face is not similar to any source face
                all_faces_similar = False  # Set the overall frame flag to False
                break  # No need to check further for this frame

        if all_faces_similar:  # If all faces in the frame are similar to some source face
            frame_filename = os.path.join(output_folder, f"frame_{frame_count}.png")
            cv2.imwrite(frame_filename, frame)
            frame_count += 1

    cap.release()
    cv2.destroyAllWindows()


def process_single_image(image_path, output_folder, app, swapper, source_face_dict):
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Could not read image.")
        return

    faces = app.get(img)
    for face in faces:
        face_features = face.normed_embedding
        for frame_photo_name, (source_face, frame_photo_face_features) in source_face_dict.items():
            similarity = np.dot(face_features, frame_photo_face_features.T)
            if similarity > 0.5:
                img = swapper.get(img, face, source_face[0], paste_back=True)

    output_filename = os.path.join(output_folder, "output_image.png")
    cv2.imwrite(output_filename, img)

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

import argparse

def main(args):
    

    app, swapper = initialize_insightface(args.model_path)

    source_face_dict = {}
    for source_img_path, frame_img_path in zip(args.source_images, args.frame_images):
        source_img = cv2.imread(source_img_path)
        source_face = app.get(source_img)
        frame_photo = cv2.imread(frame_img_path)
        frame_photo_face = app.get(frame_photo)
        frame_photo_face_features = frame_photo_face[0].normed_embedding
        source_face_dict[frame_img_path] = (source_face, frame_photo_face_features)

    output_folder = args.output_folder
    final_output_folder = args.final_output_folder
    create_directories(output_folder,final_output_folder)

    if args.process_type == "video":
        read_and_process_video(args.video_path, output_folder, app, swapper, source_face_dict)
        reassemble_video(output_folder, args.video_path, args.final_output_folder)
    elif args.process_type == "image":
        process_single_image(args.image_path, output_folder, app, swapper, source_face_dict)

    clear_gpu_memory(app, swapper)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Face Swapping Script')
    parser.add_argument('--model_path', type=str, help='Path to the model')
    parser.add_argument('--source_images', type=str, nargs='+', help='Paths to source images')
    parser.add_argument('--frame_images', type=str, nargs='+', help='Paths to frame images')
    parser.add_argument('--output_folder', type=str, help='Path to output frames folder')
    parser.add_argument('--final_output_folder', type=str, help='Path to final output folder')
    parser.add_argument('--video_path', type=str, help='Path to the video file')
    parser.add_argument('--image_path', type=str, help='Path to the image file')
    parser.add_argument('--process_type', type=str, choices=['video', 'image'], help='Type of processing: "video" or "image"')

    args = parser.parse_args()
    main(args)
