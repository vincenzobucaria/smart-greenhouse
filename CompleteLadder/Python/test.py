import snap7
from snap7.util import *
from snap7.types import *

# Creare una connessione al PLC
plc = snap7.client.Client()
plc.connect('192.168.0.5', 0, 1)  # Sostituisci con l'indirizzo IP del tuo PLC, rack e slot

# Funzione per leggere un valore dal PLC
def read_data(plc, db_number, start, size):
    result = plc.db_read(db_number, start, size)
    return result

# Funzione per scrivere un valore nel PLC
def write_data(plc, db_number, start, data):
    plc.db_write(db_number, start, data)

# Leggere un valore dal DB 1, partendo dal byte 0, di 4 byte (ad esempio, un float)
db_number = 1
start = 0
size = 4
data = read_data(plc, db_number, start, size)
value = get_real(data, 0)  # Leggere un float (REAL) dal byte 0

print(f"Valore letto dal DB {db_number}: {value}")

# Scrivere un valore nel DB 1, partendo dal byte 0
new_value = 12.34
data = bytearray(size)
set_real(data, 0, new_value)
write_data(plc, db_number, start, data)
print(f"Valore scritto nel DB {db_number}: {new_value}")

# Disconnettersi dal PLC
plc.disconnect()