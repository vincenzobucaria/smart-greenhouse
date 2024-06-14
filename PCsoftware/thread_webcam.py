import socket
import cv2
import base64
import numpy as np
import pandas as pd
import joblib
from threading import Event
from move_robot import move_robot


# Funzione per preprocessare l'immagine
def preprocess_image(image):
    resized_image = cv2.resize(image, (224, 224))  # Ridimensiona l'immagine a 224x224 pixel
    flattened_image = resized_image.flatten()  # Appiattisci l'immagine in un vettore
    return flattened_image

def camera_handle(socket: socket.socket,
                  camera: cv2.VideoCapture,
                  event: Event) -> None:
    while True:
        print("true webcam")
        msg, addr = socket.recvfrom(65535)
        while True:
            ret, frame = camera.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                _, buffer = cv2.imencode('.jpg', frame_rgb, [cv2.IMWRITE_JPEG_QUALITY, 80])
                jpg_as_text = base64.b64encode(buffer)
                socket.sendto(jpg_as_text, addr)
                if event.is_set():
                    event.clear()
                    break

def robot_control(socket_1: socket.socket,
                  socket_2: socket.socket,
                  pos: np.array,
                  event: Event,
                  cam: cv2.VideoCapture) -> None:
    while True:
        print("true robot control")
        print(socket_1.accept())
        robot_conn, _ = socket_1.accept() #accept è una chiamata bloccante. Non viene fatto l'handshake con il PLC
        print("Robot motion control connected.")
        conveyor_conn, _ = socket_2.accept()
        print("Conveyor belt control connected.")
        print("PLC connected. System started.")
        n = 0
        while True:
            while event.is_set():
                msg = conveyor_conn.recv(1024)
                
                if "START" in str(msg): #questa cosa corrisponde al nostro ArmTriggerOut
                    #print("Object detected by the photosensor.")
                    while True:
                        ret, frame = cam.read()
                        if ret:
                            break

                    # Ridimensiona il frame per una visualizzazione più rapida
                    frame = cv2.resize(frame, (640, 480))

                    # Converte il frame in HSV
                    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

                    # Definisce il range HSV per il bianco
                    lower_white = np.array([0, 0, 200])
                    upper_white = np.array([180, 55, 255])

                    # Crea una maschera per isolare il bianco
                    mask_white = cv2.inRange(hsv, lower_white, upper_white)

                    # Invert the mask to filter out white
                    mask = cv2.bitwise_not(mask_white)

                    # Applica la maschera al frame originale
                    filtered_frame = cv2.bitwise_and(frame, frame, mask=mask)

                    # Estrai la regione centrale dell'immagine
                    height, width = filtered_frame.shape[:2]
                    center_x, center_y = width // 2, height // 2
                    roi_size = 150
                    roi_x1 = max(0, center_x - roi_size // 2)
                    roi_y1 = max(0, center_y - roi_size // 2)
                    roi_x2 = min(width, center_x + roi_size // 2)
                    roi_y2 = min(height, center_y + roi_size // 2)
                    roi = filtered_frame[roi_y1:roi_y2, roi_x1:roi_x2]

                    if ret:

                        
                        # Carica il modello addestrato
                        best_rf_classifier = joblib.load('random_forest_model_optimized.joblib')

                        # Prepara l'immagine per la predizione (ridimensionamento)
                        resized_roi = cv2.resize(roi, (224, 224))

                        # Effettua la predizione utilizzando il modello addestrato
                        classe_predetta = best_rf_classifier.predict([resized_roi.flatten()])[0]

                        # Ottieni le probabilità di predizione per ciascuna classe
                        probabilita_predizione = best_rf_classifier.predict_proba([resized_roi.flatten()])[0]

                        # Imposta una soglia di probabilità per determinare la presenza del pomodoro
                        soglia_probabilita = 0.8

                        # Determina se è presente un pomodoro in base alla probabilità di predizione
                        pomodoro_presente = probabilita_predizione[classe_predetta] > soglia_probabilita

                        print(f'Probabilità predetta: {probabilita_predizione}')

                        #viene gestito il robot
                        
                        # Se la probabilità massima supera la soglia dinamica, considera che ci sia un pomodoro
                        if pomodoro_presente:
                                if classe_predetta == 0:
                                    print("Pomodoro maturo rilevato.")
                                    pos = move_robot(robot_conn, pos, rotation = -3.5, horizontal = 20, vertical = -0.02, gripper = 0)
                                    pos = move_robot(robot_conn, pos, rotation = -3.5, horizontal = 20, vertical = -0.02, gripper = 1)
                                    pos = move_robot(robot_conn, pos, rotation = 0, horizontal = 0, vertical = 0, gripper = 1)
                                    pos = move_robot(robot_conn, pos, rotation = 160, horizontal = 0, vertical = 0, gripper = 1)
                                    pos = move_robot(robot_conn, pos, rotation = 160, horizontal = 20, vertical = -0.13, gripper = 1)
                                    pos = move_robot(robot_conn, pos, rotation = 160, horizontal = 20, vertical = -0.13, gripper = 0)
                                    pos = move_robot(robot_conn, pos, rotation = 160, horizontal = 0, vertical = 0, gripper = 0)
                                    pos = move_robot(robot_conn, pos, rotation = 0, horizontal = 0, vertical = 0, gripper = 0)
                                    
                                else:
                                    print("Nessun pomodoro maturo rilevato.")
                        else:
                            print("Nessun pomodoro rilevato.")
                            
                            
                    print("Conveyor belt in ripartenza")        
                    codified_var = np.array([[1], [0]], dtype=np.uint8).tobytes()
                    conveyor_conn.sendall(codified_var)        
                            
                        # !!!!!!!!!!!!!!!!!!!!!!!! QUI DOBBIAMO NECESSARIAMENTE INFORMARE IL PLC CHE ABBIAMO FINITO, ALTRIMENTI IL CONVEYOR BELT NON RIPARTE! !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

                else:
                    print("Messaggio errato.")

              