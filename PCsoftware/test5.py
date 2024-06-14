# Import delle librerie necessarie
import cv2
import numpy as np
import joblib

# Carica il modello addestrato
rf_classifier = joblib.load('random_forest_model.joblib')

# Definisci i nomi delle classi
class_names = ['b_fully_ripened', 'b_half_ripened', 'b_green', 'l_fully_ripened', 'l_half_ripened', 'l_green']

# Funzione per preprocessare l'immagine
def preprocess_image(image):
    height, width, _ = image.shape
    center_x, center_y = width // 2, height // 2
    crop_size = 130
    half_crop_size = crop_size // 2

    # Calcola le coordinate per il ritaglio centrato
    start_x = center_x - half_crop_size
    start_y = center_y - half_crop_size
    end_x = center_x + half_crop_size
    end_y = center_y + half_crop_size

    # Estrai la regione centrale del frame originale
    cropped_image = image[start_y:end_y, start_x:end_x]

    # Ridimensiona l'immagine a 224x224 pixel
    resized_image = cv2.resize(cropped_image, (224, 224))
    flattened_image = resized_image.flatten()  # Appiattisci l'immagine in un vettore
    return flattened_image, resized_image

# Inizializza la webcam
cap = cv2.VideoCapture(1)

while True:
    # Acquisisci frame dalla webcam
    ret, frame = cap.read()
    
    # Preprocessa l'immagine
    processed_image, resized_cropped_image = preprocess_image(frame)
    
    # Effettua la previsione utilizzando il modello
    predicted_probs = rf_classifier.predict_proba([processed_image])[0]
    max_prob = np.max(predicted_probs)
    predicted_class_index = np.argmax(predicted_probs)
    predicted_class_name = class_names[predicted_class_index]
    
    # Messaggio di debug per visualizzare le probabilità predette
    print("Probabilità predette:", predicted_probs)
    
    # Imposta una soglia dinamica in base alla probabilità massima rilevata
    threshold = max_prob * 0.9  # Ad esempio, il 90% della probabilità massima
    
    # Messaggio di debug per visualizzare la soglia dinamica
    print("Soglia dinamica:", threshold)
    
    # Se la probabilità massima supera la soglia dinamica, considera che ci sia un pomodoro
    if max_prob > 0.3:
        # Converte l'immagine in scala di grigi
        gray = cv2.cvtColor(resized_cropped_image, cv2.COLOR_BGR2GRAY)
        # Applica una sogliatura per ottenere un'immagine binaria
        _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
        # Trova i contorni nell'immagine binaria
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Per ogni contorno trovato, disegna il contorno e mostra il nome della classe associata
        for contour in contours:
            # Calcola il rettangolo del contorno
            x, y, w, h = cv2.boundingRect(contour)
            # Estrai l'area del contorno
            contour_area = cv2.contourArea(contour)
            # Se l'area del contorno è sufficientemente grande, consideralo un pomodoro
            if contour_area > 100:
                # Disegna il rettangolo intorno al pomodoro
                cv2.rectangle(resized_cropped_image, (x, y), (x + w, y + h), (255, 0, 0), 2)
                # Mostra il nome della classe associata al pomodoro
                cv2.putText(resized_cropped_image, predicted_class_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    else:
        # Se la probabilità massima è inferiore alla soglia dinamica, non c'è un pomodoro
        cv2.putText(resized_cropped_image, "No tomato detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # Visualizza il frame ridimensionato 224x224
    cv2.imshow('Tomato Detection', resized_cropped_image)
    
    # Interrompi il ciclo quando si preme 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Rilascia la risorsa della webcam e chiudi le finestre
cap.release()
cv2.destroyAllWindows()
