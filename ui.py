import streamlit as st
from moviepy.editor import VideoFileClip
import whisper
import os
import shutil
import time

PERSIST_DIR = "./chroma_db"

if "messages" not in st.session_state:  # Initialize the chat message history
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask me a question from the video!"}
    ]

if "new_file_name" not in st.session_state:
    st.session_state.new_file_name = ""

if "transcribed_text" not in st.session_state:
    st.session_state.transcribed_text = ""

if "text_file_path" not in st.session_state:
    st.session_state.text_file_path = ""


def remove_folder(folder_path):
    try:
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)
            print(f"Folder '{folder_path}' has been removed successfully.")
        else:
            print(f"Folder '{folder_path}' does not exist.")
    except Exception as e:
        print(f"An error occurred while removing the folder: {e}")
        
def create_dir_if_not_exists(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created.")
    else:
        print(f"Directory '{directory_path}' already exists.")

# Function to convert video to audio
def video_to_audio(video_file, audio_file):
    video_clip = VideoFileClip(video_file)
    audio_clip = video_clip.audio
    audio_clip.write_audiofile(audio_file)
    video_clip.close()
    audio_clip.close()

# Function to transcribe audio to text
def audio_to_text(audio_file, text_file_path):
    model = whisper.load_model("base")
    transcription = model.transcribe(audio_file)
    with open(text_file_path, 'w') as file:
        file.write(transcription["text"])
    return transcription["text"]

# Function to generate MCQ questions from the text
def get_questions(question):
    response = ask_main(question)
    return response

def video_processing():
    
    # Clear session state to ensure new video is processed
    st.session_state.pop("transcribed_text", None)
    st.session_state.pop("messages", None)

    filename = uploaded_file.name.split(".")[0]
    video_file_path = f"./video/{filename}.mp4"
    audio_file_path = f"./audio/{filename}.mp3"
    st.session_state.text_file_path = f"./files/{filename}_transcription.txt"
    
    remove_folder("./video")
    remove_folder('./audio')
    remove_folder('./files')
    remove_folder(PERSIST_DIR)
    time.sleep(1)
    
    create_dir_if_not_exists("./video")
    create_dir_if_not_exists("./audio")
    create_dir_if_not_exists("./files")

    # Save the uploaded video file
    with open(video_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.write("Video uploaded successfully!")

    # Convert video to audio
    with st.spinner('Extracting audio from video...'):
        video_to_audio(video_file_path, audio_file_path)
    st.write(f"Audio extracted and saved to {audio_file_path}")

    # Convert audio to text
    with st.spinner('Transcribing audio to text...'):
        transcribed_text = audio_to_text(audio_file_path, st.session_state.text_file_path)
    
    return transcribed_text


# Streamlit App
st.title("Video to MCQ Generator")

# File uploader for MP4 video
uploaded_file = st.file_uploader("Upload a video file (MP4)", type=["mp4"])

if uploaded_file is not None:
    
    filename = uploaded_file.name.split(".")[0]
    print(st.session_state.new_file_name, "================================", filename)
    
    if filename != st.session_state.new_file_name:
        st.session_state.new_file_name = filename 
        transcribed_text = video_processing()

    else:
        with open(st.session_state.text_file_path, 'r') as file:
            transcribed_text = file.read()

    st.session_state.transcribed_text = transcribed_text
    st.write(f"Text transcribed and saved to {st.session_state.text_file_path}")

    if "transcribed_text" in st.session_state:
        from script import ask_main, load_fxn  # Import after uploading
        load_fxn()
        # Provide options to generate MCQs or start a conversation
        option = st.selectbox("Choose an option:", ["Generate MCQ Questions", "Start a Conversation"])

        if option == "Generate MCQ Questions":
            with st.spinner('Generating MCQ questions...'):
                mcq_questions = get_questions("Generate 10 MCQ questions")
                st.write("MCQ Questions Generated:")
                st.write(mcq_questions)

        elif option == "Start a Conversation":
            st.write("Starting a conversation from the text...")

            user_question = st.chat_input("Ask a question about the transcribed text:")

            if user_question:
                with st.spinner('Processing your question...'):
                    # Initialize messages if not already done
                    if "messages" not in st.session_state:
                        st.session_state.messages = [
                            {"role": "assistant", "content": "Ask me a question from the video!"}
                        ]

                    # Append the user's question to the session state
                    st.session_state.messages.append({"role": "user", "content": user_question})

                    # Generate a response based on the user's question
                    response = get_questions(user_question)
                    st.session_state.messages.append({"role": "assistant", "content": response})

                    # Display the conversation history
                    for message in st.session_state.messages:
                        with st.chat_message(message["role"]):
                            st.write(message["content"])

        st.success("Operation completed successfully!")
