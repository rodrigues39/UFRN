filtered: Made on Google Colab with GPU
library: zipfile,os
url: https://www.kaggle.com/datasets/gpiosenka/sports-classification/data

The original dataset consists of 100 distinct sports classes, covering a wide variety of modalities. For the specific purposes of this study, a targeted selection of 10 sports classes was performed, chosen based on criteria of representativeness and diversity of movements. 
This filtering resulted in the creation of a new compressed file (.zip), containing only the samples corresponding to the 10 selected classes.

During the extraction process, the original structure of the dataset was maintained, organized into three main subsets: train(1205), validation(259) and test(263), total(1727). 
This standardization ensures the integrity of the training and evaluation flow of the proposed models.

All images were standardized to RGB format with a resolution of 224×224 pixels (224×224×3), in order to ensure compatibility with convolutional neural network architectures widely used in the literature.


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
