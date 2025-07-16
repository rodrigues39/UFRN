filtered: Made on Google Colab with GPU
library: zipfile,os
url: https://www.kaggle.com/datasets/gpiosenka/sports-classification/data

Data Generation

This dataset was filtered to classify 10 different sports:
baseball, basketball, bmx, football, formula 1 racing, nascar racing, rugby, tennis, track bicycle, volleyball

# Caminho do arquivo zip
arquivo_zip = '/content/imagem_sport_10_split.zip'

# Pasta onde o conteúdo será extraído
main_folder = '/content/imagem_sport'

# Pasta onde o conteúdo será extraído
temp_folder = '/content/imagem_sport/train'

classes_desejadas = ['baseball', 'basketball', 'bmx', 'football', 'formula 1 racing',
                     'nascar racing', 'rugby', 'tennis', 'track bicycle', 'volleyball']

# Criar a pasta de destino, caso não exista
os.makedirs(main_folder, exist_ok=True)

# Abrir o arquivo zip e extrair o conteúdo
with zipfile.ZipFile(arquivo_zip, 'r') as zip_ref:
    zip_ref.extractall(main_folder)

print("Extração concluída!")

Conclusion
Despite the limited number of samples, the filtered dataset proved suitable for validating neural network architectures, applying data augmentation techniques, and analyzing overfitting. 
However, its small size poses a high risk of overfitting, which requires the use of regularization methods and/or effective data augmentation strategies to ensure the generalization capability of the trained models.
