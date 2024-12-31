import os
os.environ["KERAS_BACKEND"] = "jax"
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import logging
from pathlib import Path
import numpy as np
import librosa
import tensorflow_hub as hub
from flask import Flask, render_template, request, jsonify, session
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import keras
import torch
from werkzeug.utils import secure_filename
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Environment setup


class AudioProcessor:
    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(AudioProcessor, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not AudioProcessor._initialized:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
            self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            self.initialize_models()
            AudioProcessor._initialized = True

    def initialize_models(self):
        try:
            logger.info("Initializing models...")
            # Initialize transcription model
            model_id = "distil-whisper/distil-large-v3"
            self.transcription_model = AutoModelForSpeechSeq2Seq.from_pretrained(
                model_id, torch_dtype=self.torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
            )
            self.transcription_model.to(self.device)
            self.processor = AutoProcessor.from_pretrained(model_id)
            
            # Initialize classification model
            self.classification_model = keras.saving.load_model("hf://datasciencesage/attentionaudioclassification")
            
            # Initialize pipeline
            self.pipe = pipeline(
                "automatic-speech-recognition",
                model=self.transcription_model,
                tokenizer=self.processor.tokenizer,
                feature_extractor=self.processor.feature_extractor,
                max_new_tokens=128,
                chunk_length_s=25,
                batch_size=16,
                torch_dtype=self.torch_dtype,
                device=self.device,
            )
            
            # Initialize YAMNet model
            self.yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')
            
            logger.info("Models initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing models: {str(e)}")
            raise

    def load_wav_16k_mono(self, filename):
        try:
            wav, sr = librosa.load(filename, mono=True, sr=None)
            if sr != 16000:
                wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
            return wav
        except Exception as e:
            logger.error(f"Error loading audio file: {str(e)}")
            raise

    def get_features_yamnet_extract_embedding(self, wav_data):
        try:
            scores, embeddings, spectrogram = self.yamnet_model(wav_data)
            return np.mean(embeddings.numpy(), axis=0)
        except Exception as e:
            logger.error(f"Error extracting YAMNet embeddings: {str(e)}")
            raise

# Initialize Flask application
app = Flask(__name__)
app.secret_key = 'your_secret_key_here'
app.config['UPLOAD_FOLDER'] = Path('uploads')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Create upload folder
app.config['UPLOAD_FOLDER'].mkdir(exist_ok=True)

# Initialize audio processor (will only happen once)
audio_processor = AudioProcessor()

@app.route('/')
def index():
    session.clear()
    return render_template('terminal.html')

@app.route('/process', methods=['POST'])
def process():
    try:
        data = request.json
        command = data.get('command', '').strip().lower()

        if command in ['classify', 'transcribe']:
            session['operation'] = command
            return jsonify({
                'result': f'root@math:~$ Upload a .mp3 file for {command} operation.',
                'upload': True
            })
        else:
            return jsonify({
                'result': 'root@math:~$ Please specify an operation: "classify" or "transcribe".'
            })
    except Exception as e:
        logger.error(f"Error in process route: {str(e)}\n{traceback.format_exc()}")
        session.pop('operation', None)
        return jsonify({'result': f'root@math:~$ Error: {str(e)}'})

@app.route('/upload', methods=['POST'])
def upload():
    filepath = None
    try:
        operation = session.get('operation')
        if not operation:
            return jsonify({
                'result': 'root@math:~$ Please specify an operation first: "classify" or "transcribe".'
            })

        if 'file' not in request.files:
            return jsonify({'result': 'root@math:~$ No file uploaded.'})

        file = request.files['file']
        if file.filename == '' or not file.filename.lower().endswith('.mp3'):
            return jsonify({'result': 'root@math:~$ Please upload a valid .mp3 file.'})

        filename = secure_filename(file.filename)
        filepath = app.config['UPLOAD_FOLDER'] / filename
        
        file.save(filepath)
        wav_data = audio_processor.load_wav_16k_mono(filepath)
        
        if operation == 'classify':
            embeddings = audio_processor.get_features_yamnet_extract_embedding(wav_data)
            embeddings = np.reshape(embeddings, (-1, 1024))
            result = np.argmax(audio_processor.classification_model.predict(embeddings))
        elif operation == 'transcribe':
            result = audio_processor.pipe(str(filepath))['text']
        else:
            result = 'Invalid operation'

        return jsonify({
            'result': f'root@math:~$ Result is: {result}\nroot@math:~$ Please specify an operation: "classify" or "transcribe".',
            'upload': False
        })

    except Exception as e:
        logger.error(f"Error in upload route: {str(e)}\n{traceback.format_exc()}")
        return jsonify({
            'result': f'root@math:~$ Error: {str(e)}\nroot@math:~$ Please specify an operation: "classify" or "transcribe".'
        })
    finally:
        session.pop('operation', None)
        if filepath and Path(filepath).exists():
            try:
                Path(filepath).unlink()
            except Exception as e:
                logger.error(f"Error deleting file {filepath}: {str(e)}")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)