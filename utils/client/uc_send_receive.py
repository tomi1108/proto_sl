import zlib
import pickle

def send(
    connection,
    data
) -> None:
    
    chunk_size = 4096
    start_message = b"START"
    finish_message = b"FINISH"
    compressed_data = b""
    send_progress = 0

    while True:
        receive_message = connection.recv(chunk_size)
        if int(receive_message) == 0:
            break
        elif int(receive_message) > 0:
            raise ValueError("Connection Error.")
    
    serialized_data = pickle.dumps(data)
    compressed_data = zlib.compress(serialized_data)

    data_size = len(compressed_data)
    connection.sendall(str(data_size).encode())

    while True:
        receive_message = connection.recv(chunk_size)
        if receive_message == start_message:
            break
    
    if len(compressed_data) > chunk_size:
        while send_progress < len(compressed_data):
            sent = connection.send(compressed_data[send_progress:send_progress+chunk_size])
            send_progress += sent
    else:
        connection.send(compressed_data)
    
    while True:
        receive_message = connection.recv(chunk_size)
        if receive_message == finish_message:
            break


def receive(
    connection
):
    
    chunk_size = 4096
    start_message = b"START"
    finish_message = b"FINISH"
    compressed_data = b""

    while True:
        receive_message = connection.recv(chunk_size)
        if int(receive_message) > 0:
            break
        elif int(receive_message) == 0:
            raise ValueError("Connection Error.")
    
    connection.sendall(start_message)

    while True:
        compressed_data += connection.recv(chunk_size)
        if len(compressed_data) == int (receive_message):
            break
    
    connection.sendall(finish_message)

    uncompressed_data = zlib.decompress(compressed_data)
    data = pickle.loads(uncompressed_data)

    return data