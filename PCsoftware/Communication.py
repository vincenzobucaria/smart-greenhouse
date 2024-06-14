import json


def receive(socket):
    message = bytes()
    n = 0

    while True:
        try:
            content = socket.recv(1024)
        except ConnectionResetError:
            return None
        if not content:
            return None
        message+=content

        for char in str(content, 'utf-8'):
            if char == '{':
                n+=1
            if char == '}':
                n-=1

        if n == 0:
            break

    print(message)
    message = json.loads(message)
    return message

def send(socket, type: str, payload: str):

    packet = {
        "type" : type,
        "payload" : payload
    }

    json_packet = bytes(json.dumps(packet), 'utf-8')
    socket.send(json_packet)
    return packet
        