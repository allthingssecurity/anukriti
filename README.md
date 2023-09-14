# anukriti
**Deepfake  made easy **
This is a single file I created for creating deep fakes.

Download the file inswapper_128 (1).onnx from https://drive.google.com/file/d/1krOLgjW2tAPaqV-Bw4YALz0xT5zlb5HF/view?usp=sharing 


This is just for academic purposed and I dont take any responsibility of any usage apart from academic. This project should not be used for maligning anyone or political purposes.
I bear no responsibility for any malicious use of this

**Dependencies**

pip install insightface

pip install onnxruntime-gpu

**How to run**

!python anukriti.py --model_path "/content/drive/MyDrive/inswapper_128 (1).onnx" \
                      --source_images "/content/drive/MyDrive/source1.jpg" "/content/drive/MyDrive/source2.jpg"  \
                      --frame_images "/content/drive/MyDrive/frameimg1.jpg" "/content/drive/MyDrive/frameimg2.jpg" \
                      --output_folder "/content/frames" \
                      --final_output_folder "/content/output" \
                      --video_path "/content/drive/MyDrive/source_video.mp4" \
                      --process_type "video"
Source images are images which u want to be used like say u want a ranveer singh to replace someone. 

Frame images are faces in video frames . We have to also upload those face image. If say there are multiple faces in video maintain order accordingly with source images. First source replaces first face provided
in frame images

output folder is where your frames get accunmulated

final_output_folder is where final video is stored

video_path is the original video which is to be morphed

process_type depends on if you want to morph video or image

model_path is where u store the downloaded model from here ( https://drive.google.com/file/d/1krOLgjW2tAPaqV-Bw4YALz0xT5zlb5HF/view?usp=sharing )

Examples

Image of Keanu Reeves

![keanu](https://github.com/allthingssecurity/anukriti/assets/49463903/e7569587-e682-4eda-9a8c-c11ac79992d9)

Image of Obama

![obama](https://github.com/allthingssecurity/anukriti/assets/49463903/b2667317-7f34-4675-999b-6375a9c88d18)

Image of Kejriwal

![kejri(6)](https://github.com/allthingssecurity/anukriti/assets/49463903/6129fa9b-d1c4-43e3-9806-a88eccfcd763)

Image of Interviewer

![interviewer](https://github.com/allthingssecurity/anukriti/assets/49463903/e6f253f0-6aa6-4f60-ab8f-ffb4e06b487f)

Original Video



https://github.com/allthingssecurity/anukriti/assets/49463903/37a1d0f1-f984-4a89-a69a-74e805353279


Now i replace Keanu with Obama and Interviewer with Kejriwal

Here is the video

https://github.com/allthingssecurity/anukriti/assets/49463903/49fb6f4d-14e9-4909-bf2d-faaa27c72748


Anothe example

Amitabh photo

![amitabh](https://github.com/allthingssecurity/anukriti/assets/49463903/17b06c9f-af7c-4613-949c-8f3c35e35aee)


Ranveer Singh Photo

![ranveer](https://github.com/allthingssecurity/anukriti/assets/49463903/6ada8734-164c-4da6-a371-9fcc182ef43b)

Original Video

https://github.com/allthingssecurity/anukriti/assets/49463903/92178d6b-89e3-4d75-a81c-f86077e0ca73

Morphed Video


https://github.com/allthingssecurity/anukriti/assets/49463903/0310454c-2bea-4b1b-bba6-d35582d01ae5









