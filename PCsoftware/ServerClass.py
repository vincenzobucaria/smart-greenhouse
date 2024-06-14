import socket
import threading
import cv2
import numpy as np
#from keras.models import load_model
from Communication import send, receive
from thread_webcam import robot_control, camera_handle


class ServerClass():
    
    def __init__(self, params: dict):

        
        #inizializzazione degli attributi della classe 
        BUFF_SIZE = 65535 
        
        self.host_pc = params["host_pc"] #IP di loopback dell'host
        self.comm_port = params["comm_port"] #la porta comune dovrebbe essere quella che permette a client e server di comunicare
        self.video_port = params["video_port"] #porta della prima videocamera
        self.video2port = params["video2_port"] #porta della seconda videocamera
        self.host_plc = params["host_plc"] #ip del PLC
        self.plc_port_1 = params["plc_port_1"] #porta relativa alla socket che permette di gestire il nastro trasportatore
        self.plc_port_2 = params["plc_port_2"] #porta relativa alla socket che permette di gestire il controllo del robot
        
        """
        
        Guardando lo schema ladder è possibile notare come ci siano due blocchi TCON, questo perché vengono create due socket TCP/IP.
        La prima per gestire il conveyor belt, la seconda per gestire il braccio robotico
        
        """
        
        
        self._footage_camera = cv2.VideoCapture(params["footage_cam_ID"])
        self._AI_camera = cv2.VideoCapture(params["AI_cam_ID"])
        #self._AI_camera.set(cv2.CAP_PROP_BRIGHTNESS,)
        
        #self._AI_model = load_model(params["AI_model_path"])
        
        
        # _____________ CREAZIONE SOCKETs ________________________ #
        
        self._video_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._video_socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, BUFF_SIZE)
        self._video2_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._video2_socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, BUFF_SIZE)
        self._comm_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._plc_socket_1 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._plc_socket_2 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        
        # _________________________________________________________ #
        
        
        
        # ___________________ GESTIONE DEI THREADs _______________________ #
        
        print(self._plc_socket_2)
        self._client_sync = threading.Event() #Gli eventi servono a richimare l'attenzione dei thread
        self._video_sync = threading.Event()
        self._system_state = "STOP"
        self._initial_position = np.array([[0],[0],[0],[0],[0],[0],[0],[0]])
        
        
        # LANCIO DEI THREAD
        
        self._video_thread = threading.Thread(target=camera_handle, args=(self._video_socket,
                                                                          self._footage_camera,
                                                                          self._video_sync))
        self._robot_thread = threading.Thread(target=robot_control, args=(self._plc_socket_1,
                                                                          self._plc_socket_2,
                                                                          self._initial_position,
                                                                          self._client_sync,
                                                                          self._footage_camera))

    def start(self):
        
        
        # ______________ BIND DELLE SOCKET ______________________ #
        
        self._video_socket.bind((self.host_pc, self.video_port))
        self._comm_socket.bind((self.host_pc, self.comm_port))
        print(self.plc_port_1, self.host_plc)
        self._plc_socket_1.bind((self.host_plc, self.plc_port_1))
        self._plc_socket_2.bind((self.host_plc, self.plc_port_2))
        self._video_thread.start()
        self._comm_socket.listen()
        self._plc_socket_1.listen()
        self._plc_socket_2.listen()
        self._robot_thread.start()
        print("Server listening.")
        
    
        
        
        # ___________ GESTIONE COMUNICAZIONE SERVER E CLIENT PYTHON (QUI IL PLC NON C'ENTRA NULLA!) _______________ #
        # La cosa interessante è che mediante gli eventi viene inibita o meno l'esecuzione dei thread.
        
        
        while True:
            conn, addr = self._comm_socket.accept()
            print(f"Client connected with address and port {addr}.")
            while True:
                print("true")
                message = receive(conn)
                if message is None:
                    if self._system_state == "RUNNING":
                        print("running")
                        self._client_sync.clear() #thread in attesa
                        self._system_state = "STOP"
                        print("Due to client disconnection, the system has been stopped.")
                    self._video_sync.set()
                    print("Client disconnected.\nServer listening.")
                    break

                if message["payload"] == "START":
                    if self._system_state == "STOP":
                        self._system_state = "RUNNING"
                        self._client_sync.set() #thread in esecuzione
                        send(conn, "message", "System started.\n")
                    else:
                        print("A Start command has been received, but the system is already running.")
                        send(conn, "message", "A Start command has been received, but the system is already running.\n")


                if message["payload"] == "STOP":
                    if self._system_state == "RUNNING":
                        self._client_sync.clear() #thread in attesa
                        self._system_state = "STOP"
                        print("System stopped.")
                        send(conn, "message", "System stopped.\n")
                    else:
                        print("A Stop command has been received, but the system is not running.")
                        send(conn, "message", "A Stop command has been received, but the system is not running.\n")
                    