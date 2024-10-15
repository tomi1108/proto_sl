import zlib
import pickle

def send(
    connection,
    data
) -> None:
    
    chunk_size = 4096
    send_progress = 0
    start_message = b"START"
    finish_message = b"FINISH"

    serialized_data = pickle.dumps(data)
    compressed_data = zlib.compress(serialized_data)
    send_data_size = len(compressed_data)

    # データサイズを送る
    connection.sendall(str(send_data_size).encode())

    while True:
        receive_message = connection.recv(chunk_size)
        if receive_message == start_message:
            break
    
    while send_progress < len(compressed_data):
        sent = connection.send(compressed_data[send_progress:send_progress+chunk_size])
        send_progress += sent
    
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

    connection.sendall(str(0).encode())

    while True:
        recieve_message = connection.recv(chunk_size)
        if int(recieve_message) > 0:
            break
    
    connection.sendall(start_message)

    while True:
        chunk = connection.recv(chunk_size)
        if not chunk:
            raise ConnectionError("Connection lost.")
        compressed_data += chunk
        if len(compressed_data) == int(recieve_message):
            break
    
    uncompressed_data = zlib.decompress(compressed_data)
    data = pickle.loads(uncompressed_data)

    connection.sendall(finish_message)

    return data