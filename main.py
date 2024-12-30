import streamlit as st
import pandas as pd
import numpy as np
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from copy import deepcopy
import re
import pandas as pd
from moviepy.editor import VideoFileClip
import os
from IPython.display import Audio
from pydub import AudioSegment
import numpy as np
import wave

from transformers import BartTokenizer, BartForConditionalGeneration



def find_speaker(frame,overlay=True):
    frame = cv2.resize(frame, (1000, 750))

    # Convert to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_green = np.array([20, 40, 60])
    upper_green = np.array([80, 255, 255])

    # Create a mask for green color
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    max_rect = None
    max_rect_area = 0
    if contours:
        for contour in contours:
            # Approximate the contour to a polygon
            approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)

            # If the polygon has four sides, it could be a rectangle
            if len(approx) == 4:
                # Get the bounding box coordinates
                x, y, w, h = cv2.boundingRect(approx)
                if w * h > max_rect_area:
                    max_rect = {"x": x, "y": y, "w": w, "h": h}
                    max_rect_area = max_rect["w"] * max_rect["h"]
    cutout=None
    if max_rect:
        cutout=frame[max_rect["y"]:max_rect["y"]+max_rect["h"], max_rect["x"]:max_rect["x"]+max_rect["w"]]
        if overlay==True:
            cv2.rectangle(frame, (max_rect["x"], max_rect["y"]),
                          (max_rect["x"] + max_rect["w"], max_rect["y"] + max_rect["h"]), (0, 0, 255), 3)
    return frame,max_rect,cutout

from PIL import Image , ImageOps
import pytesseract
def get_speaker_name(cutout):
#     plt.imshow(cutout)
#     plt.show()
    pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'  # Change this to your Tesseract path
    hsv = cv2.cvtColor(cutout, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 55, 255])
    mask = cv2.inRange(hsv, lower_white, upper_white)
    result = cv2.bitwise_and(cutout, cutout, mask=mask)
    text = pytesseract.image_to_string(result)
    print("text:",text)
    speaker_name=text.replace('\n','')
    speaker_name = re.sub(r'[^A-Za-z0-9 ]+', '', speaker_name)
    if speaker_name=="":
        speaker_name="No Speaker"
    print(speaker_name)
    return text 


import Levenshtein as lev

def find_similar(name, name_list):
    similarity_threshold = 0.5
    for existing_name in name_list:
        similarity = lev.ratio(existing_name, name)
        if similarity >= similarity_threshold:
            return True, existing_name
    return False, None


def extract_timestamps_of_speakers(filename):
    cv2.destroyAllWindows()
    cap = cv2.VideoCapture(f'dataset/{filename}.mp4')
    fps = cap.get(cv2.CAP_PROP_FPS)

    output_video_path=f"dataset/pre_proc/{filename}_overlay.mp4"
    if os.path.exists(output_video_path):
        os.remove(output_video_path)
    # Define the codec and create VideoWriter object to save the output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID' for .avi format
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (1000, 750))

    frame_count = 0

    # Calculate the number of frames to skip (process every 1 second)
    skip_frames = int(fps)

    panel_center_x=-100
    panel_center_y=-100
    cur_speaker_name="No Speaker"
    speaker_names=[]
    df = pd.DataFrame(columns=["time","speaker"])
    time_sec=0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process every 5th frame
        if frame_count % skip_frames == 0:
            frame,rect,cutout=find_speaker(frame)
            
            if not (cur_speaker_name!="No Speaker" and (panel_center_x-50<rect["x"]<panel_center_x+50 and \
                        panel_center_y-50<rect["y"]<panel_center_y+50)):
                cur_speaker_name=get_speaker_name(cutout)
            
            print("cur_speaker_name",cur_speaker_name)  
            print("speaker_names",speaker_names)
            is_similar,similar_speaker_name= find_similar(cur_speaker_name, speaker_names)
            if is_similar==False:
                speaker_names.append(cur_speaker_name)
            else:
                cur_speaker_name=similar_speaker_name
                
            text=cur_speaker_name    
    #         print(text)
            # Write the processed frame to the output video

            position = (50, 50)  # (x, y) coordinates

            # Define the font and text properties
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1
            font_color = (255, 255, 255)  # White color in BGR
            font_thickness = 2

            # Overlay the text on the image
            cv2.putText(frame, text, position, font, font_scale, font_color, font_thickness)
            time_sec=time_sec+1
            df.loc[len(df)] = {"time":time_sec,"speaker":cur_speaker_name}

        out.write(frame)
        frame_count += 1
    df.to_csv(f"dataset/pre_proc/{filename}_speakers.csv")
    # Release everything when the job is finished
    cap.release()
    out.release()
    cv2.destroyAllWindows()



def split_audio(df, audio_path_wav, output_dir):
    segments=[]
    start_time=0
    duration=1
    cur_speaker=df["speaker"][0]
    speaker_map={}
    speaker_id=1
    for i in range(1,len(df)):
        if df["speaker"][i] not in speaker_map.keys():
            speaker_map[str(df["speaker"][i])]=str(speaker_id)
            speaker_id=speaker_id+1
            
        if cur_speaker!=df["speaker"][i] or i==len(df)-1:
            segments.append({"start":start_time,"end":start_time+duration,"speaker_id":speaker_map[str(df["speaker"][i-1])]})
            cur_speaker=df["speaker"][i]
            start_time=df["time"][i]
        else:
            duration=duration+1
    count=0
    for segment in segments:
        count=count+1
        output = f"{output_dir}\\a{count}_{str(segment['speaker_id'])}.wav"

        sound = AudioSegment.from_wav(audio_path_wav) # for mp3: AudioSegment.from_mp3()

        StrtTime = float(segment["start"]) * 1000
        EndTime  = float(segment["end"]) * 1000
        extract = sound[StrtTime:EndTime]

        # save
        extract.export(output, format="wav")

    return speaker_map

def setup_prereq_for_audio(filename):
    input_video_path = os.path.join("dataset", f"{filename}.mp4")
    output_audio_path = os.path.join("dataset", "pre_proc", f"{filename}.mp3")
    audio = VideoFileClip(input_video_path).audio
    audio.write_audiofile(output_audio_path)
    audio_path_wav = f"{output_audio_path[:-4]}.wav"
    audio = AudioSegment.from_mp3(output_audio_path)
    audio = audio.set_channels(1)  # Convert to mono
    audio = audio.set_frame_rate(16000)  # Set frame rate to 16kHz
    audio.export(audio_path_wav, format="wav")

    return audio_path_wav





def audio_to_transcript(speaker_map):
    import whisper
    import glob
    model_whisper_t = whisper.load_model("tiny")
    reversed_speaker_map = {v: k for k, v in speaker_map.items()}
    audio_splits=glob.glob(f"{output_dir}/*")
    transcripts=[]
    
    for audio in audio_splits: 
        key = f'{audio.rsplit("_", 1)[-1][:-4]}'  # Extract the key from the audio file name
        if key in reversed_speaker_map:
            speaker = reversed_speaker_map[key]
            predicted_text = model_whisper_t.transcribe(audio)["text"]
            print(f"Speaker {speaker}:")
            print(predicted_text)
            transcripts.append({"speaker":speaker[:-2],"text":predicted_text})
        else:
            # Handle the case where the key is not found in the dictionary
            pass
    
    return transcripts


sample_text = """
Natural language processing (NLP) is a fascinating field.
In NLP, computers can understand and generate human-like text.
The ultimate goal is to make computers interpret natural language.
This involves challenges such as language ambiguity.
The challenges also include understanding context.
"""

st.title("Speaker Diarization")

st.markdown("---")

st.markdown("## Input")

st.markdown("<p style='font-size: 18px;'>Sample Input Video:</p>",
            unsafe_allow_html=True)

st.video("meet2_fin.mp4")

st.markdown(
    "<p style='font-size: 18px;'>Please upload a video to perform speaker diariazation:</p>", unsafe_allow_html=True)

uploaded_audio_file = st.file_uploader(
    "", type=["mp4"])

if uploaded_audio_file is not None:

    st.write("File uploaded successfully!")

    st.warning("Please be patient, processing might take upto 5 minutes!")

    st.markdown("---")
    # filename=uploaded_audio_file.name[:-4]
    filename="meet2_fin"
    # extract_timestamps_of_speakers(filename)

    # st.markdown("## Processed Video Overlay")
    # st.video(f"dataset/pre_proc/meet2_fin_overlay_v2.mp4")

    # video_path = 'dataset/pre_proc/meet2_fin_overlay_v2.mp4'  # Replace with the actual path to your video file

    # Create a button for downloading the video
    # download_button = st.button("Download Processed Video Overlay")

    # Check if the download button is clicked
    # if download_button:
    #     if os.path.isfile(video_path):  # Check if the file exists
    #         with open(video_path, 'rb') as f:
    #             video_bytes = f.read()
    #         st.download_button(
    #             label="Click to Download Video",
    #             data=video_bytes,
    #             key='download_video'
    #         )
    #     else:
    #         st.warning("The video file does not exist.")

    audio_path_wav=setup_prereq_for_audio(filename)

    import shutil
    output_dir = f"dataset/pre_proc/{filename}_audio_splits"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    os.makedirs(output_dir, exist_ok=True)

    df=pd.read_csv(f"dataset/pre_proc/{filename}_speakers.csv")
    speaker_map=split_audio(df, audio_path_wav, output_dir)

    transcripts=audio_to_transcript(speaker_map)

    st.markdown("## Generated Audio Transcript")

    summary_en = ""

    for i in range(len(transcripts)):

        text_input = st.text_area(
            "", transcripts[i]["speaker"]+":"+transcripts[i]["text"], height=50)
        
        model_name = "facebook/bart-base"
        tokenizer = BartTokenizer.from_pretrained(model_name)
        model = BartForConditionalGeneration.from_pretrained(model_name)

        inputs = tokenizer.encode("summarize: " + text_input,
                                return_tensors="pt", max_length=1024, truncation=True)

        summary_ids = model.generate(inputs, max_length=50, min_length=10,
                                    length_penalty=2.0, num_beams=4, early_stopping=True)

        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

        temp = summary.split(":",1)[-1]

        summary_en += temp+"\n\n"

    st.markdown("---")

    st.markdown("## Generated Summary")

    st.write(summary_en)

    st.download_button(
        label="Click here to download the results!",
        data=summary_en,
        file_name="results.txt",
        key=f"download_button"
    )
