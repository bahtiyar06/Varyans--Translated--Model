import streamlit as st
from transformers import MarianMTModel, MarianTokenizer
import torch
import os

# Model ve tokenizer'ı yükleyin
model_directory = "C:/Users/bahti/CeviriModel/translated-model"
tokenizer = MarianTokenizer.from_pretrained(model_directory)
model = MarianMTModel.from_pretrained(model_directory)

# Cihaz ayarı
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()

# Model ağırlıklarını yükle
checkpoint_path = os.path.join(model_directory, 'model_best.pt')
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])

# Metin çevirisi için fonksiyon
def translate_text(text, model, tokenizer, device, max_length=256):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=max_length)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        translated_tokens = model.generate(**inputs, max_length=max_length, num_beams=4, early_stopping=True)

    translated_text = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
    return translated_text

# Streamlit arayüzü
st.title("Varyans Çeviri")
input_text = st.text_area("Çevirilecek metni girin:", "")

if st.button("ÇEVİR"):
    if input_text:
        texts = [input_text]

        # Çeviri işlemi
        hypotheses = [translate_text(text, model, tokenizer, device) for text in texts]

        # Sonuçları göster
        st.text_area("Çeviri:", f"{hypotheses[0]}", height=200)        
    
    else:
        st.warning("Lütfen bir metin girin.") 

# HTML ve CSS kullanarak geliştirici bilgilerini sağ altta küçük boyutta göster
st.markdown(
    """
    <style>
    .footer {
        position: fixed;
        bottom: 10px;
        right: 10px;
        font-size: 12px;
        color: gray;
    }
    </style>
    <div class="footer">
        Geliştiriciler:
        \<a href="https://github.com/bahtiyar06" target="_blank">BahtiyarASLAN</a>
        \<a href="https://github.com/AhmetKLBS" target="_blank">AhmetKULABAŞ</a> &bull; 
    </div>
    """,
    unsafe_allow_html=True
)
