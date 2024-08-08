from moviepy.editor import VideoFileClip
import whisper
from pydub import AudioSegment

#import all the neccessary libraries
import warnings
warnings.filterwarnings("ignore")
import torch
from transformers import T5ForConditionalGeneration,T5Tokenizer
from sense2vec import Sense2Vec
from sentence_transformers import SentenceTransformer
from textwrap3 import wrap
import random
import numpy as np
# import nltk
# nltk.download('punkt')
# nltk.download('brown')
# nltk.download('wordnet')
# from nltk.corpus import wordnet as wn
# from nltk.tokenize import sent_tokenize
# nltk.download('stopwords')
# from nltk.corpus import stopwords
import string
import pke
import traceback
from flashtext import KeywordProcessor
from collections import OrderedDict
from sklearn.metrics.pairwise import cosine_similarity
# nltk.download('omw-1.4')
# from similarity.normalized_levenshtein import NormalizedLevenshtein
import pickle
import time
import os 
# from utilities import utilities
from script import ask_main
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


filename = 'Statistics For Data Science - Data Science Tutorial - Simplilearn'

video_file = "./video/"+filename+".mp4"  # Replace with the path to your video file
audio_file = "./audio/"+filename+".mp3"  # Replace with the desired path for the audio file

text_file_path = "files/data.txt"

# mcq_file_path =  "text/"+filename+"MCQ.txt"



def video_to_audio(video_file, audio_file):
    # sound = AudioSegment.from_file(video_file,format="mp4", shell=False)
    # sound.export(audio_file, format="wav")
    
    # Load the video file
    video_clip = VideoFileClip(video_file)
    
    # Extract audio from the video file
    audio_clip = video_clip.audio
    
    # Write the audio to a file
    audio_clip.write_audiofile(audio_file)
    
    # Close the clips
    video_clip.close()
    audio_clip.close()


def audio_to_text(audio_file, text_file_path):
    print("++++++++++++++++++",audio_file)
    # audio_file = "C://Users//sagar.gyanchandani//OneDrive - I2e Consulting//Desktop//question generator//audio\How To Crack Interviews-20231220_121534-Meeting Recording.mp3"
    model = whisper.load_model("base")
    transcription = model.transcribe(audio_file)
    # print(transcription["text"])
    with open(text_file_path, 'w') as file:
        file.write(transcription["text"])
    

# NLTK
# def get_questions(text_file_path, mcq_file_path):

#     with open(text_file_path, 'r') as file:
#         file_content = file.read()

#     final_questions = utilities.get_mca_questions(file_content)
#     for q in final_questions:
#         print(q)
#         with open(mcq_file_path, 'a') as file:
#             file.write(q+"\n")

#LLM
def get_questions(filename):
    response = ask_main("Generate 10 MCQ question", filename)
    print("Response:", response)
    print("Response saved to chat_response.txt")

    
import time
start_time = time.time()

video_to_audio(video_file, audio_file)
print(f"Audio extracted and saved to {audio_file}")
print(audio_file)

audio_to_text(audio_file, text_file_path) 
print(f"Text extracted and saved to {text_file_path}")

get_questions(filename)

print("Time taken:",  (time.time() - start_time)/60, "minutes")