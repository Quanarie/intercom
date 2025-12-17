Рецепт домашних пряников. Сука, они настолько хорошие, что после 1 штуки твой муж точно сойдёт с ума. Теперь слушай внимательно, я открою тебе секрет:

Взяв 4 стакана муки,

Смешай её с 2 стаканами сахара,

Добавь половину чайной ложки корицы и щепоткуимбиря.

Потом, блять перемешай всё это тесто и засунь его в духовку на чуть больше 35 минут с температурой 200 градусов, охлада 20 минут и готово блять.

Пряники, рецепт домашних пряников. Пряники домашние.

Нахуй магазин, домашние пряники, дома, ждут пряники.

Домашний пряник, пряник.

Домашнее видео прянички, вкусно и пряник, пряники в доме, пряники с плеткой. Пряники боевик. Пряник фото, пряник видео. Голый пряник фотографии, картина с пряником.


# Voice Access Recognition – Milestone 1

This project trains a simple convolutional neural network (CNN) to classify
voice recordings as **allowed** or **not allowed** based on spectrograms.

The notebook:
- Loads and cleans audio data
- Converts recordings into mel-spectrograms
- Balances the dataset between allowed/not_allowed classes
- Trains a CNN in an infinite loop (until you stop it)
- Saves model checkpoints to `/models`
- Saves training plots (loss + F1) to `/outputs`
- Allows testing your own audio file in the last cell

---

## 📦 1. Installation

### **Python version**
This project uses:
3.10.12


Make sure to install this version if you are using pyenv or similar tools.

---

## 📦 2. Install dependencies

Inside your virtual environment run:

pip install -r requirements.txt


---

## 📁 3. Dataset Structure

Your dataset folder must look like:

dataset/
│
├── allowed/
│ ├── Speaker_0001/
│ ├── Speaker_0002/
│ └── ...
│
└── not_allowed/
├── Speaker_0001/
├── Speaker_0002/
└── ...


Each speaker contains WAV audio files.

---

## ▶️ 4. Running the Notebook

1. Open **Jupyter Notebook** or **JupyterLab**
2. Run the notebook **top to bottom**
3. Training begins in the last training cell  
   (it runs **forever** until you stop it with the Stop button)

This cell will save:

### ✔ Model checkpoints  
Saved in:
./models/model_epoch_XX.pth


### ✔ Training plots  
Saved in:
./outputs/epoch_XXX.png


---

## 🎤 5. Testing Your Own Audio Recording

At the very bottom of the notebook, you will find this test cell:

```python
import librosa
import numpy as np
import torch

# Make sure the model class is defined above in the notebook:
model = SimpleConvNet()
model.load_state_dict(torch.load("./models/model_epoch_XX.pth"))  # <---- CHANGE THIS 
model.eval()

def preprocess_audio(path):
    y, sr = librosa.load(path, sr=args.sr)

    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=args.n_mels)
    S = librosa.power_to_db(S, ref=np.max)

    # Standardize
    S = (S - S.mean()) / (S.std() + 1e-9)

    # Fix length to 200 frames
    if S.shape[1] < 200:
        pad = 200 - S.shape[1]
        S = np.pad(S, ((0, 0), (0, pad)), mode='constant')
    else:
        S = S[:, :200]

    return torch.tensor(S).unsqueeze(0).unsqueeze(0).float()

test_file = "my_voice.wav" # <---- CHANGE THIS 

X = preprocess_audio(test_file)

with torch.no_grad():
    out = model(X)
    pred = out.argmax(1).item()

print("Prediction:", "ALLOWED" if pred == 1 else "NOT ALLOWED")

✔ How to use it:

1)Put a WAV file in the project folder (e.g., my_voice.wav)

2)Modify:
test_file = "my_voice.wav"

3)Modify which model to load:
model.load_state_dict(torch.load("./models/model_epoch_10.pth"))

4)Run the cell