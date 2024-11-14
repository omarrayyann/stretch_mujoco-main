import socket
import numpy as np
import io
import re
import matplotlib.pyplot as plt
from PIL import Image

def send_image_and_prompt(host, port, image_array, prompt):
    """
    Sends an image (as a PIL image) and a text prompt to a server via a socket.
    
    Parameters:
    - host (str): The server address.
    - port (int): The server port.
    - image_array (numpy.ndarray): The image to be sent, as a numpy array.
    - prompt (str): The text prompt to be sent.
    
    Returns:
    - str: The response from the server.
    """
    
    # Convert the NumPy array to a PIL image
    image = Image.fromarray(image_array.astype('uint8'), 'RGB')
    
    # Save the image as PNG in bytes
    image_bytes_io = io.BytesIO()
    image.save(image_bytes_io, format='PNG')
    image_bytes = image_bytes_io.getvalue()

    # Prepare the prompt
    prompt_bytes = prompt.encode('utf-8')

    # Combine image and prompt with a separator
    separator = b'--SEPARATOR--'
    data = image_bytes + separator + prompt_bytes

    # Send data size first (4 bytes)
    data_size = len(data)
    data_size_bytes = data_size.to_bytes(4, 'big')

    # Create a socket and send the data
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((host, port))
        s.sendall(data_size_bytes + data)
        
        # Receive the response
        response = b''
        while True:
            packet = s.recv(4096)
            if not packet:
                break
            response += packet

    # Return the generated text
    return response.decode('utf-8')


def extract_points(molmo_output, image_w, image_h):
    all_points = []
    for match in re.finditer(r'x\d*="\s*([0-9]+(?:\.[0-9]+)?)"\s+y\d*="\s*([0-9]+(?:\.[0-9]+)?)"', molmo_output):
        try:
            point = [float(match.group(i)) for i in range(1, 3)]
        except ValueError:
            pass
        else:
            point = np.array(point)
            if np.max(point) > 100:
                # Treat as an invalid output
                continue
            point /= 100.0
            point = point * np.array([image_w, image_h])
            all_points.append(point)
    return all_points

def get_points(image_array, prompt, host='localhost', port=8080):
    """
    Takes an image and a prompt, sends them to a server, and returns the extracted points.

    Parameters:
    - image_array (numpy.ndarray): The image to be sent, as a numpy array.
    - prompt (str): The text prompt to be sent.
    - host (str): The server address. Default is 'localhost'.
    - port (int): The server port. Default is 8080.

    Returns:
    - list of numpy.ndarray: A list of points (each as a numpy array [x, y]).
    """
    # Ensure the image is in uint8 format (values 0-255)
    image_array = convert_to_uint8(image_array)
    
    molmo_output = send_image_and_prompt(host, port, image_array, prompt)
    image_h, image_w, _ = image_array.shape
    points = extract_points(molmo_output, image_w, image_h)

    plt.imshow(image_array)
    if points:
        points = np.array(points)
        plt.scatter(points[:, 0], points[:, 1], c='red', marker='x')
    plt.title("Molmo")
    plt.show()
    
    return points

def convert_to_uint8(image_array):
    """
    Converts a float image array (values between 0 and 1) to uint8 (values between 0 and 255).
    
    Parameters:
    - image_array (numpy.ndarray): The image array, can be float or uint8.
    
    Returns:
    - numpy.ndarray: The image converted to uint8.
    """
    if image_array.dtype == np.float32 or image_array.dtype == np.float64:
        # If the image is in float format (0 to 1), scale it to (0 to 255) and convert to uint8
        return (image_array * 255).astype(np.uint8)
    else:
        # If the image is already in uint8 format, return it as is
        return image_array.astype(np.uint8)
    
if __name__ == "__main__":
    # Example usage
    host = 'localhost'  # Server address
    port = 8080  # Server port

    # Randomly generate an image array (or load your actual image)
    image_path = 'hh.png'
    image_array = plt.imread(image_path)[:,:,:3]
    if image_array.dtype == np.float32 or image_array.dtype == np.float64:
        image_array = (image_array * 255).astype(np.uint8)
    else:
        image_array = image_array.astype(np.uint8)

    image_h, image_w, _ = image_array.shape

    # Example prompt
    prompt = "Point to all the points in the image where I need to pull to open the drawer."

    # Call the function
    molmo_output = send_image_and_prompt(host, port, image_array, prompt)
    print(molmo_output)
    points = extract_points(molmo_output, image_w, image_h)

    plt.imshow(image_array)
    if points:
        points = np.array(points)
        plt.scatter(points[:, 0], points[:, 1], c='red', marker='x')
    plt.title("Molmo")
    plt.show()
