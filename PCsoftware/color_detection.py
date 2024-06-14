import cv2
import pandas as pd

# Load the CSV file
csv_path = 'colors.csv'
index = ["color", "color_name", "hex", "R", "G", "B"]
csv = pd.read_csv(csv_path, names=index, header=None)

# function to calculate minimum distance from all colors and get the most matching color
def get_color_name(R, G, B):
    minimum = float('inf')
    cname = ""
    for i in range(len(csv)):
        d = abs(R - int(csv.loc[i, "R"])) + abs(G - int(csv.loc[i, "G"])) + abs(B - int(csv.loc[i, "B"]))
        if d < minimum:
            minimum = d
            cname = csv.loc[i, "color_name"]
    return cname

# function to get the color at the center of the frame
def get_center_color(frame):
    h, w, _ = frame.shape
    center_x, center_y = w // 2, h // 2
    b, g, r = frame[center_y, center_x]
    return int(r), int(g), int(b)

# Check if the color is a shade of red
def is_red_shade(r, g, b):
    return r > 100 and g < 100 and b < 100

# Open webcam stream
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_BRIGHTNESS, 1000)
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Get the color at the center of the frame
    r, g, b = get_center_color(frame)
    color_name = get_color_name(r, g, b)

    # Check for shades of red and print "rosso" if detected
    if is_red_shade(r, g, b):
        print("rosso")
    else:
        print("CUCULO")
    
    # Display the color name and RGB values
    text = f'{color_name} R={r} G={g} B={b}'
    
    # Draw a filled rectangle at the top to display text
    cv2.rectangle(frame, (20, 20), (750, 60), (b, g, r), -1)
    
    # Add text over the rectangle
    cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    
    # Draw a cross at the center
    center_x, center_y = frame.shape[1] // 2, frame.shape[0] // 2
    cv2.drawMarker(frame, (center_x, center_y), (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)
    
    # Display the frame
    cv2.imshow('Webcam', frame)
    
    # Break the loop when user hits 'esc' key
    if cv2.waitKey(20) & 0xFF == 27:
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
