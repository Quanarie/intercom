"""
Voice Intercom - Streamlit UI
Allows users to:
1. Select a trained model
2. Load a reference audio file
3. Record or load a test audio file
4. Run inference and view results
"""

import os
import io
import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
import librosa.display
import soundfile as sf
import matplotlib.pyplot as plt
from pathlib import Path
import sounddevice as sd
import scipy.io.wavfile as wavfile
import tempfile
from datetime import datetime


# ==================== MODEL ARCHITECTURE ====================
class CNN(nn.Module):
    """
    Simplified CNN model matching the actual saved model architecture.
    - 2 Conv2d layers with implicit pooling/downsampling
    - 1 FC output layer
    """
    def __init__(self, n_mels=64):
        super(CNN, self).__init__()
        self.c1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.c2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        
        # The saved model has pooling or striding that reduces dimensions
        # 64 x 200 -> 16 x 50 after pooling by 4
        # So flattened size: 32 * 16 * 50 = 25,600
        self.pool = nn.MaxPool2d(2)
        
        self.fc = nn.Linear(32 * 16 * 50, 2)
    
    def forward(self, x):
        # x shape: (batch, 1, 64, 200)
        x = F.relu(self.c1(x))           # (batch, 16, 64, 200)
        x = self.pool(x)                  # (batch, 16, 32, 100)
        
        x = F.relu(self.c2(x))           # (batch, 32, 32, 100)
        x = self.pool(x)                  # (batch, 32, 16, 50)
        
        # Flatten
        x = x.view(x.size(0), -1)        # (batch, 25600)
        x = self.fc(x)                    # (batch, 2)
        return x


# ==================== AUDIO PROCESSING ====================
def load_audio(file_path, sr=16000):
    """Load audio file and return waveform and sample rate"""
    try:
        audio, sr = librosa.load(file_path, sr=sr)
        return audio, sr
    except Exception as e:
        st.error(f"Error loading audio: {e}")
        return None, None


def record_audio(duration=5, sr=16000):
    """Record audio from microphone"""
    try:
        st.info(f"Recording for {duration} seconds...")
        audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype='float32')
        sd.wait()
        audio = audio.flatten().copy()  # Explicitly copy to avoid memory issues
        return audio, sr
    except Exception as e:
        st.error(f"Error recording audio: {e}")
        return None, None


def audio_to_mel_spectrogram(audio, sr=16000, n_mels=64, fixed_length=200):
    """Convert audio to mel-spectrogram"""
    try:
        # Compute mel-spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio, sr=sr, n_mels=n_mels, n_fft=1024, hop_length=512
        )
        
        # Convert to dB scale
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Normalize
        mel_spec_db = (mel_spec_db - mel_spec_db.mean()) / (mel_spec_db.std() + 1e-8)
        
        # Pad or truncate to fixed length
        if mel_spec_db.shape[1] < fixed_length:
            pad_width = ((0, 0), (0, fixed_length - mel_spec_db.shape[1]))
            mel_spec_db = np.pad(mel_spec_db, pad_width, mode='constant', constant_values=0)
        else:
            mel_spec_db = mel_spec_db[:, :fixed_length]
        
        return mel_spec_db
    except Exception as e:
        st.error(f"Error converting to mel-spectrogram: {e}")
        return None


def plot_spectrogram(mel_spec, title="Mel-Spectrogram"):
    """Plot mel-spectrogram"""
    fig, ax = plt.subplots(figsize=(10, 4))
    img = librosa.display.specshow(
        mel_spec, x_axis='time', y_axis='mel', sr=16000, hop_length=512, ax=ax
    )
    ax.set_title(title)
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    plt.tight_layout()
    return fig


# ==================== MODEL OPERATIONS ====================
@st.cache_resource
def load_model(model_path, device='cpu'):
    """Load trained model"""
    try:
        model = CNN(n_mels=64)
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


def predict(model, mel_spec, device='cpu'):
    """Run inference on mel-spectrogram"""
    try:
        # Convert to tensor: (1, 1, 64, 200)
        mel_tensor = torch.from_numpy(mel_spec[np.newaxis, np.newaxis, :, :]).float()
        mel_tensor = mel_tensor.to(device)
        
        with torch.no_grad():
            logits = model(mel_tensor)
            probs = F.softmax(logits, dim=1)
            pred_class = torch.argmax(probs, dim=1).item()
            confidence = probs[0, pred_class].item()
        
        return pred_class, confidence, probs[0].cpu().numpy()
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return None, None, None


def get_available_models(models_dir='./models'):
    """Get list of available model checkpoints"""
    if not os.path.exists(models_dir):
        return []
    models = sorted([f for f in os.listdir(models_dir) if f.endswith('.pth')])
    return models


# ==================== STREAMLIT UI ====================
def main():
    st.set_page_config(
        page_title="Voice Intercom Classification",
        page_icon="🎤",
        layout="wide"
    )
    
    st.title("🎤 Voice Intercom Classification System")
    st.markdown("**Classify audio as Allowed or Not Allowed using trained CNN models**")
    
    # Sidebar for model and device selection
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Device selection
        device = "cuda" if torch.cuda.is_available() else "cpu"
        st.info(f"Using device: **{device.upper()}**")
        
        # Model selection
        available_models = get_available_models()
        if not available_models:
            st.error("No models found in ./models directory")
            return
        
        selected_model = st.selectbox(
            "Select Model",
            available_models,
            index=len(available_models) - 1  # Default to latest model
        )
        
        model_path = f"./models/{selected_model}"
        st.success(f"Model selected: {selected_model}")
        
        # Load model
        model = load_model(model_path, device=device)
        if model is None:
            st.error("Failed to load model")
            return
    
    # Main content area
    
    # ==================== TEST AUDIO ====================
        st.subheader("🎙️ Test Audio")
        st.markdown("Choose how to provide test audio")
        
        test_mode = st.radio(
            "Test audio source",
            ["Upload File", "Record Audio"],
            horizontal=True
        )
        
        test_audio = None
        test_mel_spec = None
        
        if test_mode == "Upload File":
            test_file = st.file_uploader(
                "Upload test audio",
                type=["wav", "mp3", "ogg", "flac"],
                key="test_audio"
            )
            
            if test_file is not None:
                try:
                    test_audio, sr = librosa.load(test_file, sr=16000)
                    test_mel_spec = audio_to_mel_spectrogram(test_audio, sr=sr)
                    
                    st.success(f"✓ Test audio loaded ({len(test_audio)/16000:.2f}s)")
                    
                except Exception as e:
                    st.error(f"Error loading test audio: {e}")
        
        else:  # Record Audio
            col_record = st.columns([3, 1])
            
            with col_record[0]:
                duration = st.slider("Recording duration (seconds)", 1, 10, 5)
            
            with col_record[1]:
                if st.button("🔴 Record"):
                    test_audio, sr = record_audio(duration=duration, sr=16000)
                    if test_audio is not None:
                        test_mel_spec = audio_to_mel_spectrogram(test_audio, sr=sr)
                        st.success("✓ Audio recorded successfully")
                        
                        # Save to session state for later playback
                        st.session_state['recorded_audio'] = test_audio
                        st.session_state['recorded_sr'] = sr
            
            # Playback recorded audio if available
            if 'recorded_audio' in st.session_state:
                test_audio = st.session_state['recorded_audio']
                sr = st.session_state['recorded_sr']
                test_mel_spec = audio_to_mel_spectrogram(test_audio, sr=sr)
    
    # ==================== PREDICTION ====================
    st.divider()
    st.subheader("🔍 Classification Results")
    
    if test_mel_spec is not None:
        col_result_1, col_result_2 = st.columns([1, 1])
        
        with col_result_1:
            # Run prediction
            pred_class, confidence, probs = predict(model, test_mel_spec, device=device)
            
            if pred_class is not None:
                class_names = ["❌ Not Allowed", "✅ Allowed"]
                pred_label = class_names[pred_class]
                
                # Display main result
                if pred_class == 1:
                    st.success(f"**Prediction: {pred_label}**", icon="✅")
                else:
                    st.error(f"**Prediction: {pred_label}**", icon="❌")
                
                st.metric("Confidence", f"{confidence*100:.1f}%")
        
        with col_result_2:
            # Plot test spectrogram
            fig = plot_spectrogram(test_mel_spec, "Test Audio Spectrogram")
            st.pyplot(fig)
        
        # Detailed probabilities
        st.divider()
        st.subheader("📊 Detailed Scores")
        
        col_prob_1, col_prob_2 = st.columns(2)
        
        with col_prob_1:
            st.write("**Class Probabilities:**")
            prob_df = {
                "Class": ["Not Allowed", "Allowed"],
                "Probability": [f"{probs[0]*100:.2f}%", f"{probs[1]*100:.2f}%"]
            }
            st.dataframe(prob_df, use_container_width=True)
        
        with col_prob_2:
            # Probability bar chart
            fig, ax = plt.subplots(figsize=(8, 3))
            colors = ['#ff6b6b', '#51cf66']
            bars = ax.barh(['Not Allowed', 'Allowed'], probs, color=colors)
            ax.set_xlabel('Probability')
            ax.set_xlim(0, 1)
            
            # Add percentage labels
            for i, (bar, prob) in enumerate(zip(bars, probs)):
                ax.text(prob + 0.02, i, f'{prob*100:.1f}%', va='center')
            
            plt.tight_layout()
            st.pyplot(fig)
    
    else:
        st.info("👆 Please upload or record test audio to see predictions")
    
    # ==================== FOOTER ====================
    st.divider()
    st.markdown("---")
    col_footer_1, col_footer_2, col_footer_3 = st.columns(3)
    
    with col_footer_1:
        st.caption(f"**Model:** {selected_model}")
    
    with col_footer_2:
        st.caption(f"**Device:** {device.upper()}")
    
    with col_footer_3:
        st.caption(f"**Timestamp:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
