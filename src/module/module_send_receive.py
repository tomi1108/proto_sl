import zlib
import pickle

# データを送るのがどっちでも、最初はサーバからクライアントにアクセスする
def server(connection, condition, data=None):
    # send_progress = 0
    chunk_size = 4096
    start_message = b"START"
    end_message = b"END"
    finish_message = b"VERYOK"
    compressed_data = b""

    if condition == b"SEND":
        serialized_data = pickle.dumps(data)
        compressed_data = zlib.compress(serialized_data) + end_message
        
        connection.sendall(b"SEND")
        while True:
            receive_message = connection.recv(chunk_size)
            if receive_message == start_message:
                break
        if len(compressed_data) > chunk_size:
            send_progress = 0
            while send_progress < len(compressed_data):
                sent = connection.send(compressed_data[send_progress:send_progress+chunk_size])
                send_progress += sent
        else:
            connection.send(compressed_data)
        while True:
            receive_message = connection.recv(chunk_size)
            if receive_message == finish_message:
                break
        
        return None

    elif condition == b"REQUEST":
        connection.sendall(b"REQUEST")
        while True:
            chunk = connection.recv(chunk_size)
            compressed_data += chunk
            if compressed_data.endswith(end_message):
                compressed_data = compressed_data[:-len(end_message)]
                break
        
        uncompressed_data = zlib.decompress(compressed_data)
        data = pickle.loads(uncompressed_data)

        return data

def client(connection, data=None):
    chunk_size = 4096
    start_message = b"START"
    end_message = b"END"
    finish_message = b"VERYOK"
    compressed_data = b""
    send_progress = 0
    
    while True:
        receive_message = connection.recv(chunk_size)
        if receive_message == b"SEND":
            break
        elif receive_message == b"REQUEST":
            break
    
    if receive_message == b"SEND":
        connection.sendall(start_message)
        while True:
            chunk = connection.recv(chunk_size)
            compressed_data += chunk
            if compressed_data.endswith(end_message):
                compressed_data = compressed_data[:-len(end_message)]
                break
        connection.sendall(finish_message)
        uncompressed_data = zlib.decompress(compressed_data)
        data = pickle.loads(uncompressed_data)

        return data
    
    elif receive_message == b"REQUEST":
        serialized_data = pickle.dumps(data)
        compressed_data = zlib.compress(serialized_data) + end_message

        if len(compressed_data) > chunk_size:
            while send_progress < len(compressed_data):
                send_progress += chunk_size
                connection.send(compressed_data[send_progress-chunk_size:send_progress])
        else:
            connection.send(compressed_data)
        
        return None