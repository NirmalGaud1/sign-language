#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import google.generativeai as genai
import time
import textwrap

# Configuration - CHANGED TO TFLITE
MODEL_PATH = "sign_language_model.tflite"
GEMINI_API_KEY = "AIzaSyA-9-lTQTWdNM43YdOXMQwGKDy0SrMwo6c"
CATEGORIES = ['1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G',
              'H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']

# Initialize Gemini
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel('gemini-pro')

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=2,
                       min_detection_confidence=0.5)

# Load TFLite model - CHANGED SECTION
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Streamlit page configuration
st.set_page_config(page_title="Indian Sign Language Translator", layout="wide")

# Session state initialization
if 'buffer' not in st.session_state:
    st.session_state.buffer = []
if 'gemini_text' not in st.session_state:
    st.session_state.gemini_text = "Make gestures to begin..."
if 'last_update' not in st.session_state:
    st.session_state.last_update = 0
if 'running' not in st.session_state:
    st.session_state.running = False

def get_gemini_response(text):
    """Get explanation from Gemini"""
    try:
        response = gemini_model.generate_content(
            f"Interpret this sequence of Indian Sign Language gestures as meaningful text: {text}. "
            "Provide both the literal translation and possible meanings in 2 short paragraphs. "
            "If it appears to be random letters, suggest possible word formations."
        )
        return response.text
    except Exception as e:
        print(f"Gemini Error: {e}")
        return "Explanation unavailable"

def process_frame(frame):
    """Process frame for hand detection and classification"""
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    current_predictions = []

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Get hand bounding box
            h, w, _ = frame.shape
            x_coords = [lm.x * w for lm in hand_landmarks.landmark]
            y_coords = [lm.y * h for lm in hand_landmarks.landmark]
            
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)
            
            # Expand bounding box
            expansion = 0.2
            x_min = max(0, x_min - (x_max - x_min) * expansion)
            x_max = min(w, x_max + (x_max - x_min) * expansion)
            y_min = max(0, y_min - (y_max - y_min) * expansion)
            y_max = min(h, y_max + (y_max - y_min) * expansion)

            # Crop and predict
            hand_image = frame[int(y_min):int(y_max), int(x_min):int(x_max)]
            if hand_image.size == 0:
                continue
                
            # Preprocess for TFLite
            processed_image = cv2.resize(hand_image, (64, 64))
            processed_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
            processed_image = processed_image.astype(np.float32) / 255.0
            processed_image = np.expand_dims(processed_image, axis=0)

            # TFLite inference
            interpreter.set_tensor(input_details[0]['index'], processed_image)
            interpreter.invoke()
            pred = interpreter.get_tensor(output_details[0]['index'])[0]
            
            pred_class = CATEGORIES[np.argmax(pred)]
            confidence = np.max(pred)

            if confidence > st.session_state.confidence_threshold:
                # Draw bounding box
                cv2.rectangle(frame, (int(x_min), int(y_min)),
                            (int(x_max), int(y_max)), (0, 255, 0), 2)
                cv2.putText(frame, f"{pred_class} ({confidence:.2f})",
                          (int(x_min), int(y_min)-10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                
                current_predictions.append(pred_class)

    # Update detection buffer
    if current_predictions:
        st.session_state.buffer.extend(current_predictions)
        st.session_state.buffer = st.session_state.buffer[-st.session_state.buffer_size:]
    
    return frame

# Streamlit UI
st.title("Indian Sign Language Translator 🤟")

# Sidebar controls
with st.sidebar:
    st.header("Controls")
    st.session_state.confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.7)
    st.session_state.buffer_size = st.slider("Buffer Size", 1, 10, 5)
    
    if st.button("Clear Buffer"):
        st.session_state.buffer = []
        st.session_state.gemini_text = "Buffer cleared!"
        
    st.markdown("---")
    st.write("Detection Buffer:")
    st.write(st.session_state.buffer)

# Main interface
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Live Detection")
    run_button = st.empty()
    
    if st.session_state.running:
        if run_button.button("Stop Detection"):
            st.session_state.running = False
    else:
        if run_button.button("Start Detection"):
            st.session_state.running = True

    camera = st.camera_input("Webcam Feed")

with col2:
    st.subheader("Explanation")
    explanation = st.empty()
    explanation.markdown(st.session_state.gemini_text)

# Processing loop
if st.session_state.running and camera is not None:
    bytes_data = camera.getvalue()
    frame = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
    
    # Process frame
    processed_frame = process_frame(frame)
    
    # Update display
    col1.image(processed_frame, channels="BGR")
    
    # Update Gemini text every 2 seconds
    if time.time() - st.session_state.last_update > 2 and st.session_state.buffer:
        st.session_state.gemini_text = get_gemini_response(' '.join(st.session_state.buffer))
        st.session_state.last_update = time.time()
        explanation.markdown(st.session_state.gemini_text)
