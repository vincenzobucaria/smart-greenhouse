import numpy as np


def move_robot(socket, old_position, **kwargs):

    bit16rotation = np.array([old_position[1,0], old_position[0,0]], dtype=np.uint8)
    bit16vertical = np.array([old_position[3,0], old_position[2,0]], dtype=np.uint8)
    bit16horizontal = np.array([old_position[5,0], old_position[4,0]], dtype=np.uint8)
    gripper = np.array([old_position[6,0]], dtype=np.uint8)
    boolean_value = True
    reached_rotation = bit16rotation.view(dtype=np.uint16)
    reached_vertical = bit16vertical.view(dtype=np.uint16)
    reached_horizontal = bit16horizontal.view(dtype=np.uint16)

    rotation = bit16rotation.view(dtype=np.uint16)
    vertical = bit16vertical.view(dtype=np.uint16)
    horizontal = bit16horizontal.view(dtype=np.uint16)

    for key, value in kwargs.items():
        match key.lower():
            case 'rotation':
                temp_rotation = value
                rotation = round((temp_rotation+5)*9.85)
                if rotation > 3300:
                    rotation = 3300
                if rotation < 0:
                    rotation = 0
                rotation = np.array([rotation], dtype=np.uint16)
                bit16rotation= rotation.view(dtype=np.uint8)
            case 'horizontal':
                temp_horizontal = value
                horizontal = round(temp_horizontal*821)
                if horizontal > 78:
                    horizontal = 78
                if horizontal < 0:
                    horizontal = 0
                horizontal = np.array([horizontal], dtype=np.uint16)
                bit16horizontal= horizontal.view(dtype=np.uint8)
            case 'vertical':
                temp_vertical = value
                vertical = round(temp_vertical*(-12857))
                if vertical > 1800:
                    vertical = 1800
                if vertical < 0:
                    vertical = 0
                vertical = np.array([vertical], dtype=np.uint16)
                bit16vertical= vertical.view(dtype=np.uint8)
            case 'gripper':
                gripper = value
    print("GRIPPER: ", gripper)
    codified_var = np.array([[bit16rotation[1]], [bit16rotation[0]],
                             [bit16vertical[1]], [bit16vertical[0]],
                             [bit16horizontal[1]], [bit16horizontal[0]],
                             [gripper], 
                             [0]], dtype=np.uint8).tobytes()
    if gripper == old_position[6,0] and np.array_equal(reached_horizontal, horizontal) and ((reached_rotation >= np.double(rotation)-10).all() and (reached_rotation <= np.double(rotation)+10).all()) and ((reached_vertical >= np.double(vertical)-10).all() and (reached_vertical <= np.double(vertical)+10).all()):
        return old_position


    print(codified_var)
    print("codified var stampato")
    socket.sendall(codified_var)
    print("qui debug")
    old_position = np.frombuffer(socket.recv(8), dtype = np.uint8).reshape(-1,1)
    print("qui arrivo? probabilmente no")
    return old_position



#Se non ho capito male non c'è un feedback dal PLC a riguardo della posizione del braccio
