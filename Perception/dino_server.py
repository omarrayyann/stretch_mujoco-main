import socket
import struct
import json
import base64
import numpy as np
from PIL import Image
import io
import torch
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

class ObjectDetection:
    def __init__(self, model_id="IDEA-Research/grounding-dino-tiny", device=None):
        # Initialize device
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize processor and model
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(self.device)

    def get_bounding_boxes(self, image, text_queries, box_threshold=0.4, text_threshold=0.3):
        """
        Perform object detection on the image with given text queries.
        """
        inputs = self.processor(images=image, text=text_queries, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Post-process to get bounding boxes
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=[image.size[::-1]]  # (height, width)
        )
        
        return results

    def run_detection(self, rgb_image, prompt):
        """
        Run object detection on the provided RGB image with the given prompt.
        """
        # Validate RGB image shape and type
        if rgb_image.ndim != 3 or rgb_image.shape[2] != 3:
            raise ValueError("RGB image must have shape (H, W, 3)")
        if rgb_image.dtype != np.uint8:
            raise ValueError("RGB image dtype must be uint8")

        # Convert NumPy array to PIL Image
        image = Image.fromarray(rgb_image)
        
        # Perform object detection
        results = self.get_bounding_boxes(image, prompt)
        
        # Process results
        detections = []
        for score, label, box in zip(results[0]['scores'], results[0]['labels'], results[0]['boxes']):
            box = box.cpu().numpy().tolist()  # [xmin, ymin, xmax, ymax]
            score = score.cpu().item()
            
            detections.append({
                'score': score,
                'label': label,
                'box': box
            })
        
        return detections

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

def handle_client_connection(client_socket, detector):
    try:
        # Receive the first 4 bytes indicating the length of the incoming JSON data
        raw_msglen = recv_all(client_socket, 4)
        if not raw_msglen:
            print("Client disconnected before sending data length.")
            return
        msglen = struct.unpack('>I', raw_msglen)[0]
        
        # Receive the actual JSON data
        data = recv_all(client_socket, msglen)
        if not data:
            print("Client disconnected before sending all data.")
            return
        
        # Decode JSON
        message = json.loads(data.decode('utf-8'))
        
        # Extract prompt and image
        prompt = message.get('prompt', '')
        rgb_image_b64 = message.get('rgb_image', '')
        
        if not prompt or not rgb_image_b64:
            response = {'error': 'Missing prompt or rgb_image in the request'}
        else:
            # Decode the base64 image
            image_bytes = base64.b64decode(rgb_image_b64)
            rgb_image = np.load(io.BytesIO(image_bytes))
            
            # Perform object detection
            detections = detector.run_detection(rgb_image, prompt)
            
            response = {'detections': detections}
        
        # Serialize response to JSON
        response_json = json.dumps(response)
        response_bytes = response_json.encode('utf-8')
        response_length = struct.pack('>I', len(response_bytes))
        
        # Send response length and data
        client_socket.sendall(response_length + response_bytes)
    
    except Exception as e:
        error_response = {'error': str(e)}
        response_json = json.dumps(error_response)
        response_bytes = response_json.encode('utf-8')
        response_length = struct.pack('>I', len(response_bytes))
        client_socket.sendall(response_length + response_bytes)
        print(f"Error handling client: {e}")
    finally:
        client_socket.close()

def start_server(host='0.0.0.0', port=4000):
    # Initialize the ObjectDetection instance
    detector = ObjectDetection()

    # Create a TCP/IP socket
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind((host, port))
    server.listen(5)  # Listen for up to 5 connections
    print(f"Server listening on {host}:{port}")

    try:
        while True:
            client_sock, address = server.accept()
            print(f"Accepted connection from {address}")
            handle_client_connection(client_sock, detector)
    except KeyboardInterrupt:
        print("\nShutting down the server.")
    finally:
        server.close()

if __name__ == '__main__':
    start_server()
