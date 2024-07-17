# import zlib
# import pickle

# # データを送るのがどっちでも、最初はサーバからクライアントにアクセスする
# def server(connection, condition, data=None):
#     # send_progress = 0
#     chunk_size = 4096
#     start_message = b"START"
#     end_message = b"END"
#     finish_message = b"VERYOK"
#     compressed_data = b""

#     if condition == b"SEND":
#         serialized_data = pickle.dumps(data)
#         compressed_data = zlib.compress(serialized_data) + end_message
        
#         connection.sendall(b"SEND")
#         while True:
#             receive_message = connection.recv(chunk_size)
#             if receive_message == start_message:
#                 break
#         send_progress = 0
#         while send_progress < len(compressed_data):
#             sent = connection.send(compressed_data[send_progress:send_progress+chunk_size])
#             send_progress += sent
#         # if len(compressed_data) > chunk_size:
#         #     send_progress = 0
#         #     while send_progress < len(compressed_data):
#         #         sent = connection.send(compressed_data[send_progress:send_progress+chunk_size])
#         #         send_progress += sent
#         # else:
#         #     connection.send(compressed_data)
#         while True:
#             receive_message = connection.recv(chunk_size)
#             if receive_message == finish_message:
#                 break
        
#         return None

#     elif condition == b"REQUEST":
#         connection.sendall(b"REQUEST")
#         while True:
#             chunk = connection.recv(chunk_size)
#             if not chunk:
#                 raise ConnectionError("Connection lost")
#             compressed_data += chunk
#             if compressed_data.endswith(end_message):
#                 compressed_data = compressed_data[:-len(end_message)]
#                 break
        
#         uncompressed_data = zlib.decompress(compressed_data)
#         data = pickle.loads(uncompressed_data)

#         return data

# def client(connection, data=None):
#     chunk_size = 4096
#     start_message = b"START"
#     end_message = b"END"
#     finish_message = b"VERYOK"
#     compressed_data = b""
#     send_progress = 0
    
#     while True:
#         receive_message = connection.recv(chunk_size)
#         if receive_message == b"SEND":
#             break
#         elif receive_message == b"REQUEST":
#             break
    
#     if receive_message == b"SEND":
#         connection.sendall(start_message)
#         while True:
#             chunk = connection.recv(chunk_size)
#             compressed_data += chunk
#             if compressed_data.endswith(end_message):
#                 compressed_data = compressed_data[:-len(end_message)]
#                 break
#         connection.sendall(finish_message)
#         uncompressed_data = zlib.decompress(compressed_data)
#         data = pickle.loads(uncompressed_data)

#         return data
    
#     elif receive_message == b"REQUEST":
#         serialized_data = pickle.dumps(data)
#         compressed_data = zlib.compress(serialized_data) + end_message

#         if len(compressed_data) > chunk_size:
#             while send_progress < len(compressed_data):
#                 connection.send(compressed_data[send_progress:send_progress+chunk_size])
#                 send_progress += chunk_size
#         else:
#             connection.send(compressed_data)
        
#         return None

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
        compressed_data = zlib.compress(serialized_data)
        send_data_size = len(compressed_data)
        
        # データサイズを送る
        connection.sendall(str(send_data_size).encode()) # データサイズを文字列に変換して送信
        while True:
            receive_message = connection.recv(chunk_size)
            if receive_message == start_message:
                break
        send_progress = 0
        while send_progress < len(compressed_data):
            sent = connection.send(compressed_data[send_progress:send_progress+chunk_size])
            send_progress += sent
        # if len(compressed_data) > chunk_size:
        #     send_progress = 0
        #     while send_progress < len(compressed_data):
        #         sent = connection.send(compressed_data[send_progress:send_progress+chunk_size])
        #         send_progress += sent
        # else:
        #     connection.send(compressed_data)
        while True:
            receive_message = connection.recv(chunk_size)
            if receive_message == finish_message:
                break
        
        return None

    elif condition == b"REQUEST":
        # 値0を送る
        connection.sendall(str(0).encode())
        while True:
            receive_message = connection.recv(chunk_size)
            if int(receive_message) > 0:
                break
        
        connection.sendall(start_message)
            
        while True:
            chunk = connection.recv(chunk_size)
            if not chunk:
                raise ConnectionError("Connection lost")
            compressed_data += chunk
            if len(compressed_data) == int(receive_message):
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
        # 受信データがデータサイズの場合
        if int(receive_message) > 0:
            break
        elif int(receive_message) == 0:
            break
    
    if int(receive_message) > 0:
        connection.sendall(start_message)
        while True:
            chunk = connection.recv(chunk_size)
            compressed_data += chunk
            # 受信サイズがデータサイズに達した場合
            if len(compressed_data) == int(receive_message):
                break
        connection.sendall(finish_message)
        uncompressed_data = zlib.decompress(compressed_data)
        data = pickle.loads(uncompressed_data)

        return data
    
    elif int(receive_message) == 0:
        serialized_data = pickle.dumps(data)
        compressed_data = zlib.compress(serialized_data)

        data_size = len(compressed_data)
        connection.sendall(str(data_size).encode())

        while True:
            receive_message = connection.recv(chunk_size)
            if receive_message == b"START":
                break

        if len(compressed_data) > chunk_size:
            while send_progress < len(compressed_data):
                connection.send(compressed_data[send_progress:send_progress+chunk_size])
                send_progress += chunk_size
        else:
            connection.send(compressed_data)
        
        return None