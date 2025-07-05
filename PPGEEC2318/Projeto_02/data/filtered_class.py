filtered: Realizado no Google Colab com GPU 
library: zipfile,os
url: https://www.kaggle.com/datasets/gpiosenka/sports-classification/data

Data Generation

This dataset was filtered to classify 10 different sports:
baseball, basketball, bmx, football, formula 1 racing, nascar racing, rugby, tennis, track bicycle, volleyball

# Caminho do arquivo zip
arquivo_zip = '/content/imagem_sport_10.zip'

# Pasta onde o conteúdo será extraído
pasta_destino = '/content/imagem_sport'

# Criar a pasta de destino, caso não exista
os.makedirs(pasta_destino, exist_ok=True)

# Abrir o arquivo zip e extrair o conteúdo
with zipfile.ZipFile(arquivo_zip, 'r') as zip_ref:
    zip_ref.extractall(pasta_destino)

print("Extração concluída!")


