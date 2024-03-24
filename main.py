import streamlit as st
import openai
import requests

# OpenAI API key
openai.api_key = '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'

from dataclasses import dataclass
@dataclass
class Message:
    actor: str
    payload: str

import cv2
import pytesseract

# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
# Preprocess the image and extract text using OCR
def extract_text_from_image(image_path):
    # Read the image
    image = cv2.imread(image_path)

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply thresholding to binarize the image
    _, threshold_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # Use Tesseract OCR to extract text
    extracted_text = pytesseract.image_to_string(threshold_image)

    return extracted_text.strip()


def get_youtube_links(query):
    # Make a request to YouTube Data API to search for videos related to the query
    # Replace 'YOUR_API_KEY' with your actual YouTube Data API key
    youtube_api_key = 'AIzaSyD-7zL08bz8e42n4lrND3fEvGBOrGpQ58g'
    search_url = f'https://www.googleapis.com/youtube/v3/search?key={youtube_api_key}&part=snippet&type=video&q={query}'
    response = requests.get(search_url)
    data = response.json()

    # Extract video IDs from the API response and construct YouTube video links
    video_links = []
    for item in data['items']:
        video_id = item['id']['videoId']
        video_link = f'https://www.youtube.com/watch?v={video_id}'
        video_links.append(video_link+'\n')

    return video_links


def generate_response_with_gpt(wholecontext, text):
    prompt_with_input = f"Give answer to the given question while considering context as well since this context is having past asked questions. Context: {wholecontext}\nQuestion: {text}\nAnswer:"

    # Generate a response using GPT model
    response = openai.Completion.create(
        engine="gpt-3.5-turbo-instruct",
        # prompt=text,
        prompt=prompt_with_input,
        max_tokens=50
    )
    return response.choices[0].text.strip()

def get_and_append_to_list(message):
    if 'past_messages' not in st.session_state:
        st.session_state['past_messages'] = []
    st.session_state['past_messages'].append(message)
    return st.session_state['past_messages']

#--------------------
USER = "user"
ASSISTANT = "ai"
MESSAGES = "messages"
def initialize_session_state():
    if MESSAGES not in st.session_state:
        st.session_state[MESSAGES] = [Message(actor=ASSISTANT, payload="Hi! How can I help you?")]

import os
def save_uploaded_file(uploaded_file, filename):
    """Saves a file uploaded through Streamlit to the specified path."""
    with open(os.path.join("uploads", filename), "wb") as f:
        f.write(uploaded_file.getbuffer())
    return "done"
  # return st.success(f"File '{filename}' has been uploaded.")


def main():
    st.title("Chatbot OpenAI")
    initialize_session_state()

    msg: Message
    for msg in st.session_state[MESSAGES]:
        st.chat_message(msg.actor).write(msg.payload)

    message: str = st.chat_input("Enter a prompt here")

    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
    extracted_text = ""
    if uploaded_file is not None:
        filename = uploaded_file.name
        if save_uploaded_file(uploaded_file, filename):
            uploaded_image_path = os.path.join("uploads", filename)
            # print(uploaded_image_path)
            extracted_text = extract_text_from_image(uploaded_image_path)
            os.remove(uploaded_image_path)
            # print(extracted_text)

    if message:
        message += " " + extracted_text  # Concatenate text from image with user message

        st.session_state['messages'].append(Message(actor="user", payload=message))
        st.chat_message("user").write(message)

        with st.spinner("Please wait.."):
            my_list = get_and_append_to_list(message)
            print(my_list)
            response = generate_response_with_gpt(my_list, message)
            youtube_links = get_youtube_links(message)
            response += "\n\nHere are some YouTube videos related to your query:\n\n" + '\n'.join(youtube_links)
            st.session_state['messages'].append(Message(actor="ai", payload=response))
            st.chat_message("ai").write(response)


if __name__ == "__main__":
    main()

