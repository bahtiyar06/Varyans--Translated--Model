from transformers import MarianMTModel, MarianTokenizer, AutoModelForSeq2SeqLM, AutoTokenizer
import torch
import os
import sacrebleu
from googletrans import Translator
import requests

model_directory = '/content/drive/MyDrive/Colab Notebooks/translated-model'
tokenizer = MarianTokenizer.from_pretrained(model_directory)
model = MarianMTModel.from_pretrained(model_directory)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()

checkpoint_path = os.path.join(model_directory, 'model_best.pt')
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])

def translate_text(text, model, tokenizer, device, max_length=256):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=max_length)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        translated_tokens = model.generate(**inputs, max_length=max_length, num_beams=4, early_stopping=True)

    translated_text = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
    return translated_text

def compute_bleu_score(hypotheses, references):
    bleu = sacrebleu.corpus_bleu(hypotheses, [references])
    return bleu.score

def translate_with_google(texts, src_lang='tr', dest_lang='en'):
    translator = Translator()
    translations = [translator.translate(text, src=src_lang, dest=dest_lang).text for text in texts]
    return translations

def translate_with_mymemory(texts, src_lang='tr', dest_lang='en'):
    translations = []
    for text in texts:
        response = requests.get(
            "https://api.mymemory.translated.net/get",
            params={"q": text, "langpair": f"{src_lang}|{dest_lang}"}
        )
        data = response.json()
        translated_text = data['responseData']['translatedText']
        translations.append(translated_text)
    return translations

def translate_with_opus_mt(texts, src_lang='tr', dest_lang='en'):
    model_name = f"Helsinki-NLP/opus-mt-{src_lang}-{dest_lang}"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    translations = []

    for text in texts:
        inputs = tokenizer.encode(text, return_tensors="pt", truncation=True)
        outputs = model.generate(inputs, max_length=256)
        translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        translations.append(translated_text)

    return translations

texts = [
    "Benim adım Bahtiyar, ben mekatronik mühendisliği öğrencisiyim."
]

google_references = translate_with_google(texts)

mymemory_references = translate_with_mymemory(texts)

opus_mt_references = translate_with_opus_mt(texts, src_lang='tr', dest_lang='en')

hypotheses = [translate_text(text, model, tokenizer, device) for text in texts]

google_bleu_score = compute_bleu_score(hypotheses, google_references)
mymemory_bleu_score = compute_bleu_score(hypotheses, mymemory_references)
opus_mt_bleu_score = compute_bleu_score(hypotheses, opus_mt_references)

print(f"\n\n Google Translate  BLEU skoru: {google_bleu_score:.4f}")
print(f"         MyMemory  BLEU skoru: {mymemory_bleu_score:.4f}")
print(f"          OPUS-MT  BLEU skoru: {opus_mt_bleu_score:.4f}")

for i, (hypothesis, google_reference, mymemory_reference, opus_mt_reference) in enumerate(zip(hypotheses, google_references, mymemory_references, opus_mt_references)):
    print(f"\n\n***Metin: {texts[i]}***")
    print(f"\n***Çeviri: {hypothesis}***")
    print(f"\n\nGoogle Translate Referans:* {google_reference}")
    print(f"\nMyMemory Referans:* {mymemory_reference}")
    print(f"\nOPUS-MT Referans:* {opus_mt_reference}")