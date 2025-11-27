import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os

st.set_page_config(layout="wide", page_title="Brain Tumor MRI Analysis", initial_sidebar_state="collapsed")

# --- Helper Functions ---
def is_mri_image(image):
    """Robustly checks if an image is an MRI scan by analyzing color saturation."""
    # Convert to RGB if it's not, to ensure consistency
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Convert to HSV color space
    hsv_image = image.convert('HSV')
    hsv_array = np.array(hsv_image)
    
    # Extract the Saturation channel
    saturation = hsv_array[:, :, 1]
    
    # A true MRI (grayscale) will have very low saturation.
    # We check the average saturation and its standard deviation.
    avg_saturation = np.mean(saturation)
    std_saturation = np.std(saturation)
    
    # These thresholds are chosen to be very strict.
    # Natural photos will have much higher values.
    if avg_saturation < 25 and std_saturation < 20:
        return True
    else:
        return False

# --- Page Configuration ---
col1, col2, col3 = st.columns([1,2,1])
with col2:
    st.image("neu.jpg")
st.title("🧠 Brain Tumor MRI Analysis with AI Assistant")
st.markdown("Upload a brain MRI scan to get a preliminary analysis and chat with an AI assistant about the results.")

# --- Model Loading ---
@st.cache(allow_output_mutation=True)
def load_keras_model():
    model_path = 'Model2.h5'
    if not os.path.exists(model_path):
        st.error(f"Model file not found at {model_path}. Please ensure it's in the same directory.")
        return None
    try:
        return load_model(model_path)
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        return None

model = load_keras_model()
if model is None:
    st.stop()

class_names = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']

# --- Main Application Layout ---
col1, col2 = st.columns([0.4, 0.6])

with col1:
    st.header("1. Upload Image")
    uploaded_file = st.file_uploader("Choose an MRI scan...", type=["jpg", "jpeg", "png"]) 

    if uploaded_file is not None:
        # Create a unique identifier from file name and size for compatibility
        file_identifier = f"{uploaded_file.name}-{uploaded_file.size}"

        # When a new file is uploaded, save the previous session and start a new one
        if 'uploaded_file_id' not in st.session_state or st.session_state.uploaded_file_id != file_identifier:
            # If there's a completed session, save it to the history
            if 'current_session' in st.session_state and st.session_state.current_session["title"]:
                st.session_state.all_sessions.append(st.session_state.current_session)
            
            # Reset for the new session
            st.session_state.current_session = {"title": None, "history": []}
            st.session_state.uploaded_file_id = file_identifier

        image = Image.open(uploaded_file)
        if not is_mri_image(image):
            st.error("Error: The uploaded file does not appear to be a valid MRI scan. Please upload a grayscale or pseudo-grayscale medical image.")
            st.session_state.clear()
        else:
            st.image(image, caption='Uploaded MRI Scan', use_column_width=True)
            st.session_state.image_processed = True
            st.session_state.image = image

with col2:
    st.header("2. Analysis & AI Chat")
    if 'image_processed' in st.session_state and st.session_state.image_processed:
        with st.spinner('Performing analysis...'):
            img = st.session_state.image.resize((299, 299)).convert('RGB')
            img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
            img_array = tf.expand_dims(img_array, 0)
            
            predictions = model.predict(img_array)
            predicted_class = class_names[np.argmax(predictions[0])]
            confidence = float(np.max(predictions[0]))

        st.markdown(f"<div style='text-align: center;'><p style='font-size: 24px; font-weight: bold;'>Prediction: {predicted_class}</p></div>", unsafe_allow_html=True)
        st.markdown(f"<div style='text-align: center;'><p style='font-size: 20px;'>Confidence: {confidence:.2%}</p></div>", unsafe_allow_html=True)

        # --- Gemini Chatbot Integration ---
        with st.expander("👨‍⚕️ Chat with AI Assistant", expanded=True):
            try:
                gemini_api_key = st.secrets["GEMINI_API_KEY"]
            except (KeyError, FileNotFoundError):
                st.error("Gemini API key not found. Please add it to your `.streamlit/secrets.toml` file.")
                st.code("GEMINI_API_KEY = \"YOUR_API_KEY_HERE\"", language="toml")
                st.stop()

            try:
                import google.generativeai as genai
                genai.configure(api_key=gemini_api_key)
                gemini_model = genai.GenerativeModel('models/gemini-pro-latest')

                # Initialize a more complex state to manage multiple chat sessions
                if 'all_sessions' not in st.session_state:
                    st.session_state.all_sessions = []
                if 'current_session' not in st.session_state:
                    st.session_state.current_session = {"title": None, "history": []}

                initial_prompt = f"The user has uploaded a brain MRI. Your analysis concluded it is a '{predicted_class}' with {confidence:.2%} confidence. Act as a helpful medical assistant for a doctor. Explain the result in technical but clear terms, suggest potential next steps for diagnosis or treatment, and answer the doctor's questions. Always state that you are an AI and not a substitute for a qualified pathologist or radiologist."

                # If the current session is new, add the initial AI response
                if not st.session_state.current_session["history"]:
                    initial_response = f"Analysis complete. The model predicts **{predicted_class}** with {confidence:.2%} confidence. How can I assist you further with these results, Doctor?"
                    st.session_state.current_session["history"].append(f"**AI Assistant:** {initial_response}")

                # Display saved chat sessions in the sidebar using expanders
                with st.sidebar:
                    st.header("Consultation History")
                    if not st.session_state.all_sessions:
                        st.info("Saved chats will appear here.")
                    else:
                        for i, session in enumerate(st.session_state.all_sessions):
                            with st.expander(session["title"]):
                                for msg in session["history"]:
                                    st.markdown(msg, unsafe_allow_html=True)

                # Display the current chat in the main area
                for message in st.session_state.current_session["history"]:
                    st.markdown(message)


                # User input form
                with st.form(key='chat_form', clear_on_submit=True):
                    user_input = st.text_input("Your question:", key="user_input", value="")
                    submit_button = st.form_submit_button(label='Send')

                if submit_button and user_input:
                    # Set the title of the session with the first question
                    if st.session_state.current_session["title"] is None:
                        st.session_state.current_session["title"] = user_input

                    st.session_state.current_session["history"].append(f"**You:** {user_input}")
                    
                    with st.spinner("Thinking..."):
                        chat = gemini_model.start_chat(history=[])
                        full_prompt = initial_prompt + "\n\nUser question: " + user_input
                        response = chat.send_message(full_prompt)
                        st.session_state.current_session["history"].append(f"**AI Assistant:** {response.text}")
                        st.experimental_rerun()

            except ImportError:
                st.error("The 'google-generativeai' package is not installed. Please run `pip install google-generativeai`.")
            except Exception as e:
                st.error(f"An error occurred with the Gemini API: {e}")
    else:
        st.info("Upload a valid MRI scan to begin the analysis.")
