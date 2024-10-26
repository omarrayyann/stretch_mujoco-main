import socket
import struct
import json
import base64
import numpy as np
from PIL import Image
import io
import cv2

def send_detection_request(host, port, prompt, rgb_image):
    """
    Sends a detection request to the server.

    Parameters:
    - host (str): Server IP address or hostname.
    - port (int): Server port.
    - prompt (str): Text prompt for object detection.
    - rgb_image (np.ndarray): RGB image as a NumPy array with shape (H, W, 3).

    Returns:
    - dict: Detection results or error message.
    """
    try:
        print("Sending detection request...")
        # Validate RGB image
        if rgb_image.ndim != 3 or rgb_image.shape[2] != 3:
            raise ValueError("RGB image must have shape (H, W, 3)")
        if rgb_image.dtype != np.uint8:
            raise ValueError("RGB image dtype must be uint8")
        
        # Serialize the image to bytes using numpy
        img_buffer = io.BytesIO()
        np.save(img_buffer, rgb_image)
        img_bytes = img_buffer.getvalue()
        
        # Encode the image bytes to base64
        img_b64 = base64.b64encode(img_bytes).decode('utf-8')
        
        # Create the message dictionary
        message = {
            'prompt': prompt,
            'rgb_image': img_b64
        }
        
        # Serialize the message to JSON
        message_json = json.dumps(message)
        message_bytes = message_json.encode('utf-8')
        
        # Pack the length of the message
        message_length = struct.pack('>I', len(message_bytes))
        
        # Create a TCP/IP socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.connect((host, port))
            
            # Send the length and message
            sock.sendall(message_length + message_bytes)
            
            # Receive the response length
            raw_response_len = recv_all(sock, 4)
            if not raw_response_len:
                print("Server closed the connection unexpectedly.")
                return None
            response_len = struct.unpack('>I', raw_response_len)[0]
            
            # Receive the actual response
            response_data = recv_all(sock, response_len)
            if not response_data:
                print("Server closed the connection unexpectedly.")
                return None
            
            # Decode JSON response
            response = json.loads(response_data.decode('utf-8'))
            return response
            
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def recv_all(sock, length):
    """
    Helper function to receive 'length' bytes from the socket.
    """
    data = b''
    while len(data) < length:
        packet = sock.recv(length - len(data))
        if not packet:
            return None
        data += packet
    return data

# if __name__ == '__main__':
#     # Server details
#     SERVER_HOST = 'localhost'  # Change to server IP if needed
#     SERVER_PORT = 4000

#     # Define your prompt
#     prompt = "drawer. cabinet. handle."

#     # Load or create an RGB image as a NumPy array
#     # For demonstration, we'll create a dummy image
#     rgb_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
#     image_path = 'kitchen.jpg'
#     image = Image.open(image_path)

#     # Convert to an RGB NumPy array
#     rgb_image = np.array(image)
    
#     # Alternatively, load an image from a file
#     # image = Image.open('path_to_image.jpg').convert('RGB')
#     # rgb_image = np.array(image)
    
#     # Send the detection request
#     response = send_detection_request(SERVER_HOST, SERVER_PORT, prompt, rgb_image)
    
#     if response:
#         if 'detections' in response:
#             print("Detections:")
#             for det in response['detections']:
#                 print(f"Label: {det['label']}, Score: {det['score']:.2f}, Box: {det['box']}")
#         else:
#             print(f"Error from server: {response.get('error', 'Unknown error')}")
#     else:
#         print("No response received.")
