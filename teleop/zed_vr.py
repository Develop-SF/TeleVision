"""
ZED VR Streaming Script

This script captures stereo images from a ZED camera and streams them to a VR headset.
It supports both WebRTC and image-based streaming modes, with head tracking capabilities.
"""

import numpy as np
import cv2
from constants_vuer import *
import argparse
import time
import pyzed.sl as sl
from TeleVision import OpenTeleVision
from multiprocessing import Array, Process, shared_memory, Queue, Manager, Event, Semaphore
from scipy.spatial.transform import Rotation as R
from pytransform3d import rotations
import os
import socket


class ZedVRStreamer:
    """
    Class for streaming ZED camera stereo images to VR headsets.
    Supports both WebRTC and image-based streaming with head tracking.
    """

    def __init__(self, stream_mode="webrtc", use_ngrok=True):
        """
        Initialize the ZED VR streamer with specified settings.
        
        Args:
            stream_mode (str): Streaming mode - either 'webrtc' or 'image'
            use_ngrok (bool): Whether to use ngrok for remote access
        """
        self.stream_mode = stream_mode
        self.use_ngrok = use_ngrok
        
        # Camera resolution settings
        self.resolution = (720, 1280)
        self.crop_size_w = 1
        self.crop_size_h = 0
        self.resolution_cropped = (self.resolution[0] - self.crop_size_h, 
                                   self.resolution[1] - 2 * self.crop_size_w)
        
        # Image and streaming variables
        self.img_shape = (self.resolution_cropped[0], 2 * self.resolution_cropped[1], 3)
        self.shm = None
        self.img_array = None
        self.image_queue = None
        self.toggle_streaming = None
        self.tv = None
        self.zed = None
        
        # Streaming state
        self.frame_count = 0
        self.stream_active = False
        self.last_stream_state = False
        self.max_queue_size = 2
        self.last_queue_warning = 0
        
        # Initialize resources
        self.setup_camera()
        self.setup_streaming()

    def setup_camera(self):
        """
        Initialize and configure the ZED camera with appropriate settings.
        """
        # Initialize ZED camera
        self.zed = sl.Camera()

        # Set camera parameters
        init_params = sl.InitParameters()
        init_params.camera_resolution = sl.RESOLUTION.HD720  # Use HD720 video mode
        init_params.camera_fps = 60  # Set fps at 60

        # Open the camera
        err = self.zed.open(init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            print("Camera Open : " + repr(err) + ". Exit program.")
            exit()

    def setup_streaming(self):
        """
        Set up shared memory, queues and OpenTeleVision for streaming.
        """
        # Initialize shared memory for image streaming
        self.shm = shared_memory.SharedMemory(create=True, size=np.prod(self.img_shape) * np.uint8().itemsize)
        self.img_array = np.ndarray(self.img_shape, dtype=np.uint8, buffer=self.shm.buf)

        # Initialize streaming
        self.image_queue = Manager().Queue(maxsize=2)  # Limit queue size to avoid memory issues
        self.toggle_streaming = Event()

        # Get certificate file paths
        current_dir = os.path.dirname(os.path.abspath(__file__))
        cert_file = os.path.join(current_dir, "cert.pem")
        key_file = os.path.join(current_dir, "key.pem")

        # Check if certificates exist
        if not os.path.exists(cert_file) or not os.path.exists(key_file):
            print("Warning: SSL certificates not found. Creating self-signed certificates...")
            # Implement certificate creation here or use existing ones
            try:
                # Use existing system certs if available
                import ssl
                print(f"Using system certificates if available.")
            except Exception as e:
                print(f"Certificate error: {e}")

        # Create OpenTeleVision instance with streaming enabled
        self.tv = OpenTeleVision(self.resolution_cropped, self.shm.name, 
                               self.image_queue, self.toggle_streaming, 
                               stream_mode=self.stream_mode, 
                               cert_file=cert_file, key_file=key_file, 
                               ngrok=self.use_ngrok)

    def process_head_tracking(self):
        """
        Process head tracking data from the VR headset.
        Returns the head rotation matrix.
        """
        try:
            head_mat = grd_yup2grd_zup[:3, :3] @ self.tv.head_matrix[:3, :3] @ grd_yup2grd_zup[:3, :3].T
            if np.sum(head_mat) == 0:
                head_mat = np.eye(3)
            head_rot = rotations.quaternion_from_matrix(head_mat[0:3, 0:3])
            ypr = rotations.euler_from_quaternion(head_rot, 2, 1, 0, False)
            return head_mat
        except Exception:
            # Don't print errors for tracking, just continue
            return np.eye(3)

    def get_local_ip(self):
        """
        Get the local IP address of the machine.
        Returns the IP address as a string.
        """
        local_ip = "127.0.0.1"
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            local_ip = s.getsockname()[0]
            s.close()
        except:
            pass
        return local_ip

    def print_connection_info(self):
        """
        Print connection information for the user.
        """
        local_ip = self.get_local_ip()
        print(f"\n*** ZED Stereo Streaming Setup ***")
        print(f"Streaming mode: {self.stream_mode}")
        print(f"Ngrok enabled: {self.use_ngrok}")
        print(f"1. For WebRTC streaming: Open your Quest browser and navigate to:")
        print(f"   - Connect to: http://{local_ip}:8080")
        print(f"   - Or try: http://127.0.0.1:8080 (if on the same machine)")
        print(f"2. For troubleshooting:")
        print(f"   - Make sure your Quest and computer are on the same network")
        print(f"   - Try the 'image' streaming mode if WebRTC fails")
        print(f"   - DO NOT use https:// protocol, use http:// instead\n")

    def stream_frames(self):
        """
        Main streaming loop that captures and streams frames from the ZED camera.
        """
        # Initialize ZED image containers
        image_left = sl.Mat()
        image_right = sl.Mat()
        runtime_parameters = sl.RuntimeParameters()
        
        self.print_connection_info()
        
        try:
            while True:
                start_time = time.time()

                # Check if WebRTC client has requested streaming
                stream_state = self.toggle_streaming.is_set()
                if stream_state != self.last_stream_state:
                    if stream_state:
                        print("WebRTC client connected, streaming frames...")
                        self.stream_active = True
                    else:
                        print("WebRTC client disconnected, waiting for new connection...")
                        self.stream_active = False
                    self.last_stream_state = stream_state

                # Process head tracking data
                self.process_head_tracking()

                # Grab new images
                if self.zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
                    # Retrieve left and right images
                    self.zed.retrieve_image(image_left, sl.VIEW.LEFT)
                    self.zed.retrieve_image(image_right, sl.VIEW.RIGHT)
                    
                    # Process and combine images
                    bgra = np.hstack((image_left.numpy()[self.crop_size_h:, self.crop_size_w:-self.crop_size_w],
                                     image_right.numpy()[self.crop_size_h:, self.crop_size_w:-self.crop_size_w]))
                    rgb = cv2.cvtColor(bgra, cv2.COLOR_BGRA2RGB)

                    # Save the image every 10 seconds for debugging
                    if self.frame_count % 600 == 0:
                        current_dir = os.path.dirname(os.path.abspath(__file__))
                        debug_path = os.path.join(current_dir, "debug")
                        os.makedirs(debug_path, exist_ok=True)
                        cv2.imwrite(os.path.join(debug_path, f"zed_frame_{self.frame_count}.jpg"), 
                                    cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                        print(f"Saved debug frame to {debug_path}")
                        
                    self.frame_count += 1
                    
                    # Update shared memory (for image streaming mode)
                    np.copyto(self.img_array, rgb)
                    
                    # Send frame to WebRTC if streaming is active or requested
                    if self.stream_active or stream_state:
                        try:
                            # Check queue status
                            if hasattr(self.image_queue, 'qsize'):
                                queue_size = self.image_queue.qsize()
                                if queue_size >= self.max_queue_size:
                                    # Queue is full, get most recent frame to avoid lag
                                    try:
                                        while self.image_queue.qsize() > 0:
                                            self.image_queue.get_nowait()
                                    except:
                                        pass
                            
                            # Add new frame to queue
                            self.image_queue.put_nowait(rgb.copy())
                        except Exception as e:
                            # Only print queue errors occasionally to avoid flooding the console
                            if time.time() - self.last_queue_warning > 5:
                                print(f"Queue warning: {e}")
                                self.last_queue_warning = time.time()
                
                # Throttle loop to maintain reasonable CPU usage
                elapsed = time.time() - start_time
                if elapsed < 1/60:  # Cap at 60fps
                    time.sleep(1/60 - elapsed)

        except KeyboardInterrupt:
            print("Stopping ZED streaming...")
        finally:
            self.cleanup()

    def cleanup(self):
        """
        Clean up resources when the streamer is stopped.
        """
        self.toggle_streaming.clear()
        self.zed.close()
        self.shm.close()
        self.shm.unlink() 
        print("ZED camera and resources closed")

    def run(self):
        """
        Start the ZED VR streaming process.
        """
        self.stream_frames()


def main():
    """
    Parse command line arguments and start the ZED VR streamer.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='ZED VR Streaming')
    parser.add_argument('--mode', type=str, choices=['webrtc', 'image'], default='webrtc',
                        help='Streaming mode: webrtc or image (default: webrtc)')
    parser.add_argument('--ngrok', action='store_true', default=True,
                        help='Use ngrok for remote access (default: True)')
    args = parser.parse_args()
    
    # Create and run the streamer
    streamer = ZedVRStreamer(stream_mode=args.mode, use_ngrok=args.ngrok)
    streamer.run()


if __name__ == "__main__":
    main() 