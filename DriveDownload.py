import requests

def download_file_from_google_drive(file_id, destination):
    # Doğru indirme URL'si Google Drive dosya indirme linkidir
    URL = "https://drive.google.com/drive/folders/1Y52notwUMqnNSsNQxmn-GU2G9mApJyR7?usp=sharing/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params={'id': file_id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': file_id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)

def get_confirm_token(response):
    # Google Drive dosyaları için geçici bir indirme onayı gerekebilir
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768  # Dosyayı parça parça indir

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # Boş olmayan parçaları yaz
                f.write(chunk)

# Örnek kullanım
file_id = "1Y52notwUMqnNSsNQxmn-GU2G9mApJyR7"  # İndirilecek dosyanın kimliğini girin
destination = "YOUR_DESTINATION_PATH"  # Dosyanın kaydedileceği yeri girin
download_file_from_google_drive(file_id, destination)
