import socket
import pickle
import numpy as np
import sys

def send_with_length(sock, data):
    """Helper function to send data preceded by its length."""
    length = len(data)
    sock.sendall(length.to_bytes(4, byteorder='big'))
    sock.sendall(data)

def receive_all(sock, length):
    """Helper function to receive exact number of bytes."""
    data = b''
    while len(data) < length:
        packet = sock.recv(length - len(data))
        if not packet:
            return None
        data += packet
    return data

def send_numpy_array(sock, array):
    """Serialize and send a NumPy array."""
    serialized_data = pickle.dumps(array)
    send_with_length(sock, serialized_data)

def receive_numpy_array(sock):
    """Receive and deserialize a NumPy array."""
    data_length_bytes = receive_all(sock, 4)
    if not data_length_bytes:
        print("[-] Failed to receive data length.")
        return None
    data_length = int.from_bytes(data_length_bytes, byteorder='big')
    print(f"[+] Expecting {data_length} bytes of data.")
    serialized_data = receive_all(sock, data_length)
    if not serialized_data:
        print("[-] Failed to receive data.")
        return None
    array = pickle.loads(serialized_data)
    return array

def get_inference_results(server_host, server_port, rgb_array, depth_array):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
        try:
            client_socket.connect((server_host, server_port))
            print(f"[+] Connected to server at {server_host}:{server_port}")

            # Send RGB data
            print("[*] Sending RGB data...")
            send_numpy_array(client_socket, rgb_array)
            print("[+] RGB data sent.")

            # Send Depth data
            print("[*] Sending Depth data...")
            send_numpy_array(client_socket, depth_array)
            print("[+] Depth data sent.")

            # Receive inference results
            print("[*] Waiting to receive inference results...")
            output_length_bytes = receive_all(client_socket, 4)
            if not output_length_bytes:
                print("[-] Failed to receive output length.")
                return None
            output_length = int.from_bytes(output_length_bytes, byteorder='big')
            print(f"[+] Expecting {output_length} bytes of inference results.")

            serialized_output = receive_all(client_socket, output_length)
            if not serialized_output:
                print("[-] Failed to receive inference results.")
                return None

            # Deserialize the output
            inference_results = pickle.loads(serialized_output)
            print("[+] Inference results received successfully.")

            return inference_results

        except Exception as e:
            print(f"[!] An error occurred: {e}")
            return None
