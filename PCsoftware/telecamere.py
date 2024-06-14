import cv2

def find_cameras(max_cameras=10):
    available_cameras = []
    for i in range(0, max_cameras):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available_cameras.append(i)
            cap.release()
    return available_cameras

print("ID Telecamere disponibili:", find_cameras())
