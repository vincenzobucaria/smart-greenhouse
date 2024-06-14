## INIZIO CODICE
# Import delle librerie necessarie
import os
import json
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import time
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Definizione della funzione per caricare i dati da file JSON e immagini
def load_data(data_folder):
    data = []
    num_files = 0
    for subdir, dirs, files in os.walk(data_folder):
        for file in files:
            if file.endswith('.json'):
                num_files += 1
                json_file = os.path.join(subdir, file)
                print("Processing JSON file:", json_file)  # Stampa il nome del file in elaborazione
                with open(json_file, 'r') as f:
                    json_data = json.load(f)  # Carica i dati JSON
                    img_name = os.path.splitext(file)[0]  # Estrae il nome dell'immagine
                    img_path = os.path.join(subdir.replace('ann', 'img'), img_name)  # Costruisce il percorso dell'immagine
                    data.append((json_data, img_path))  # Aggiunge i dati JSON e il percorso dell'immagine alla lista
    print("Number of files loaded:", num_files)  # Stampa il numero di file caricati
    return data  # Restituisce i dati

# Definizione della funzione per preprocessare le immagini
def preprocess_image(image_path):
    image = cv2.imread(image_path)  # Carica l'immagine usando OpenCV
    resized_image = cv2.resize(image, (224, 224))  # Ridimensiona l'immagine a 224x224 pixel
    flattened_image = resized_image.flatten()  # Appiattisce l'immagine in un vettore
    return flattened_image  # Restituisce l'immagine preprocessata

# Caricamento e preparazione dei dati
start_time = time.time()  # Avvia il timer
dataset = load_data('dataset')  # Carica i dati dal dataset specificato

# Estrazione di X e y dai dati
X = []
y = []
for json_data, img_path in dataset:
    image = preprocess_image(img_path)  # Preprocessa l'immagine
    X.append(image)  # Aggiunge l'immagine preprocessata a X
    if 'objects' in json_data and json_data['objects']:
        object_data = json_data['objects'][0]  # Assume che il primo oggetto contenga l'etichetta
        if 'classTitle' in object_data:
            class_title = object_data['classTitle']  # Estrae il titolo della classe
            # Mappatura delle classi
            if class_title.startswith('b_fully_ripened'):
                label = 0
            elif class_title.startswith('b_half_ripened'):
                label = 1
            elif class_title.startswith('b_green'):
                label = 2
            elif class_title.startswith('l_fully_ripened'):
                label = 3
            elif class_title.startswith('l_half_ripened'):
                label = 4
            elif class_title.startswith('l_green'):
                label = 5
            else:
                label = -1  # Assegna un'etichetta predefinita se il titolo della classe non viene riconosciuto
        else:
            print("Warning: 'classTitle' key not found or empty in JSON data.")  # Avviso se 'classTitle' non è presente nei dati JSON
            label = -1  # Assegna un'etichetta predefinita se 'classTitle' non viene trovato o è vuoto
    else:
        print("Warning: 'objects' key not found or empty in JSON data.")  # Avviso se 'objects' non è presente nei dati JSON
        label = -1  # Assegna un'etichetta predefinita se 'objects' non viene trovato o è vuoto
    y.append(label)  # Aggiunge l'etichetta a y

end_time = time.time()  # Termina il timer
data_loading_time = end_time - start_time  # Calcola il tempo impiegato per il caricamento e la preparazione dei dati
print("Time taken for loading and preprocessing data: {:.3f} seconds".format(data_loading_time))  # Stampa il tempo impiegato

# Mischiamento dei dati
X = np.array(X)
y = np.array(y)
np.random.seed(42)
shuffle_index = np.random.permutation(len(X))
X_shuffled, y_shuffled = X[shuffle_index], y[shuffle_index]

# Divisione dei dati in set di addestramento e test bilanciati
X_train, X_test, y_train, y_test = train_test_split(X_shuffled, y_shuffled, test_size=0.2, stratify=y_shuffled, random_state=42)

# Unione dei dati di addestramento e test
X_total = np.concatenate((X_train, X_test))
y_total = np.concatenate((y_train, y_test))

# Suddivisione dei dati in set di addestramento e test con stratificazione
X_train_combined, X_test_combined, y_train_combined, y_test_combined = train_test_split(X_total, y_total, test_size=0.2, stratify=y_total, random_state=42)

# Costruzione e addestramento del classificatore Random Forest
start_time = time.time()  # Avvia il timer
rf_classifier = RandomForestClassifier(n_estimators=300, random_state=42)  # Crea un classificatore Random Forest con 300 alberi
rf_classifier.fit(X_train_combined, y_train_combined)  # Addestra il classificatore sui dati di addestramento combinati
end_time = time.time()  # Termina il timer
training_time = end_time - start_time  # Calcola il tempo impiegato per l'addestramento
print("Time taken for training: {:.3f} seconds".format(training_time))  # Stampa il tempo impiegato per l'addestramento

# Salva il modello addestrato
joblib.dump(rf_classifier, 'random_forest_model.joblib')

# Salva lo scaler
scaler = StandardScaler()
scaler.fit(X_total)  # Adatta lo scaler sull'intero dataset
joblib.dump(scaler, 'scaler.joblib')

# Valutazione del classificatore sui dati di test
start_time = time.time()  # Avvia il timer
y_pred_test = rf_classifier.predict(X_test_combined)  # Effettua le previsioni sui dati di test combinati
end_time = time.time()  # Termina il timer
test_prediction_time = end_time - start_time  # Calcola il tempo impiegato per le previsioni sui dati di test

# Valutazione del classificatore sull'intero dataset
start_time = time.time()  # Avvia il timer
y_pred_total = rf_classifier.predict(X_total)  # Effettua le previsioni sull'intero dataset
end_time = time.time()  # Termina il timer
total_prediction_time = end_time - start_time  # Calcola il tempo impiegato per le previsioni sull'intero dataset

# Calcolo delle matrici di confusione
conf_matrix_test = confusion_matrix(y_test_combined, y_pred_test)  # Calcola la matrice di confusione per i dati di test combinati
conf_matrix_total = confusion_matrix(y_total, y_pred_total)  # Calcola la matrice di confusione per l'intero dataset

class_names = ['b_fully_ripened', 'b_half_ripened', 'b_green', 'l_fully_ripened', 'l_half_ripened', 'l_green']

# Stampaggio e salvataggio della matrice di confusione per i dati di test
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_test, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix - Test Data")
plt.savefig("confusion_matrix_test.png")
plt.show()

# Stampaggio e salvataggio della matrice di confusione per l'intero dataset
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_total, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix - Total Data")
plt.savefig("confusion_matrix_total.png")
plt.show()

# Stampaggio e salvataggio del classification report per i dati di test
print("Classification Report - Test Data:")
print(classification_report(y_test_combined, y_pred_test, digits=4))  # Imposta digits a 4 per stampare 4 cifre decimali

# Stampaggio e salvataggio del classification report per l'intero dataset
print("Classification Report - Total Data:")
print(classification_report(y_total, y_pred_total, digits=4))  # Imposta digits a 4 per stampare 4 cifre decimali


# Stampa il tempo impiegato per le previsioni sui dati di test e sull'intero dataset
print("Time taken for test predictions: {:.3f} seconds".format(test_prediction_time))
print("Time taken for total predictions: {:.3f} seconds".format(total_prediction_time))

## FINE CODICE