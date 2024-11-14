import socket
import numpy as np

server_host = 'localhost'
server_port = 9875

def send_data(sock, data_bytes):
    size_bytes = len(data_bytes).to_bytes(4, byteorder='big')
    sock.sendall(size_bytes)
    sock.sendall(data_bytes)

def get_grasps(colors,depth):

    print("Sending data to server...")

    # Serialize data to bytes
    colors_data = colors.tobytes()
    depths_data = (depth*1000.0).tobytes()

    # Create socket and connect to server
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((server_host, server_port))

    # Send colors data
    send_data(client_socket, colors_data)

    # Send depths data
    send_data(client_socket, depths_data)

    # Receive grasp poses data
    size_bytes = client_socket.recv(4)
    data_size = int.from_bytes(size_bytes, byteorder='big')
    grasp_poses_data = b""
    while len(grasp_poses_data) < data_size:
        chunk = client_socket.recv(data_size - len(grasp_poses_data))
        if not chunk:
            raise RuntimeError("Socket connection broken")
        grasp_poses_data += chunk
    grasp_poses = np.frombuffer(grasp_poses_data, dtype=np.float32).reshape(-1, 4, 4)

    # Receive grasp scores data
    size_bytes = client_socket.recv(4)
    data_size = int.from_bytes(size_bytes, byteorder='big')
    grasp_scores_data = b""
    while len(grasp_scores_data) < data_size:
        chunk = client_socket.recv(data_size - len(grasp_scores_data))
        if not chunk:
            raise RuntimeError("Socket connection broken")
        grasp_scores_data += chunk
    grasp_scores = np.frombuffer(grasp_scores_data, dtype=np.float32)

    # Receive grasp widths data
    size_bytes = client_socket.recv(4)
    data_size = int.from_bytes(size_bytes, byteorder='big')
    grasp_widths_data = b""
    while len(grasp_widths_data) < data_size:
        chunk = client_socket.recv(data_size - len(grasp_widths_data))
        if not chunk:
            raise RuntimeError("Socket connection broken")
        grasp_widths_data += chunk
    grasp_widths = np.frombuffer(grasp_widths_data, dtype=np.float32)

    client_socket.close()

    # Now you have grasp_poses, grasp_scores, grasp_widths
    # You can process or print them
    print("Received grasp poses:", grasp_poses)
    print("Received grasp scores:", grasp_scores)
    print("Received grasp widths:", grasp_widths)

    return grasp_poses, grasp_scores, grasp_widths

if __name__ == '__main__':
    colors = (np.random.rand(640, 480, 3) * 255).astype(np.uint8)  # Random RGB image (640x480)
    depth = np.random.rand(640, 480).astype(np.float32)  # Random depth map (640x480)

    # Call the get_grasps function
    grasp_poses, grasp_scores, grasp_widths = get_grasps(colors, depth)

    # Print the results
    print("Test completed:")
    print("Grasp Poses:", grasp_poses)
    print("Grasp Scores:", grasp_scores)
    print("Grasp Widths:", grasp_widths)