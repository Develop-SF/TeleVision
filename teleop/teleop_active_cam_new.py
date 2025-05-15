import math
import numpy as np

np.set_printoptions(precision=2, suppress=True)
# import matplotlib.pyplot as plt # Not used
from scipy.spatial.transform import Rotation as R
from pytransform3d import rotations

import time
import cv2
import os
from constants_vuer import *
from TeleVision import OpenTeleVision
import pyzed.sl as sl
from dynamixel.active_cam import DynamixelAgent, DynamixelRobotConfig
from multiprocessing import Array, Process, shared_memory, Queue, Manager, Event, Semaphore
import argparse # For potential future command-line arguments

class ActiveCamStreamer:
    def __init__(self, stream_mode="webrtc", use_ngrok=True, dynamixel_port="/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTA7NOM6-if00-port0"):
        self.stream_mode = stream_mode
        self.use_ngrok = use_ngrok
        self.dynamixel_port = dynamixel_port

        # Camera and image settings
        self.resolution = (720, 1280)
        self.crop_size_w = 1
        self.crop_size_h = 0
        self.resolution_cropped = (self.resolution[0] - self.crop_size_h, self.resolution[1] - 2 * self.crop_size_w)
        self.img_shape = (self.resolution_cropped[0], 2 * self.resolution_cropped[1], 3)
        self.img_height, self.img_width = self.resolution_cropped[:2]

        # Components
        self.zed = None
        self.agent = None
        self.tv = None
        
        # Shared resources for streaming
        self.shm = None
        self.img_array = None
        self.image_queue = None
        self.toggle_streaming = None
        
        # Streaming state
        self.frame_count = 0
        self.stream_active = False
        self.last_stream_state = False
        self.max_queue_size = 2 # From zed_vr.py
        self.last_queue_warning = 0 # From zed_vr.py

        # Setup components
        self.setup_camera()
        self.setup_actuator()
        self.setup_streaming()

    def setup_camera(self):
        self.zed = sl.Camera()
        init_params = sl.InitParameters()
        init_params.camera_resolution = sl.RESOLUTION.HD720
        init_params.camera_fps = 60
        err = self.zed.open(init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            print(f"Camera Open Error: {repr(err)}. Exiting.")
            exit(1)
        # Set camera exposure to 50%
        self.zed.set_camera_settings(sl.VIDEO_SETTINGS.EXPOSURE, 80)
        print("ZED Camera opened successfully.")

    def setup_actuator(self):
        custom_config = DynamixelRobotConfig(
            joint_ids=(1, 2),
            joint_offsets=(2 * np.pi / 2, 2 * np.pi / 2),
            joint_signs=(-1, -1),
            gripper_config=None,
        )
        self.agent = DynamixelAgent(
            port=self.dynamixel_port, 
            dynamixel_config=custom_config
        )
        self.agent._robot.set_torque_mode(True)
        print("Dynamixel Agent configured and torque mode set.")

    def setup_streaming(self):
        self.shm = shared_memory.SharedMemory(create=True, size=np.prod(self.img_shape) * np.uint8().itemsize)
        self.img_array = np.ndarray(self.img_shape, dtype=np.uint8, buffer=self.shm.buf)
        
        # Using Manager().Queue() for consistency with zed_vr.py if OpenTeleVision expects it across processes
        self.image_queue = Manager().Queue(maxsize=self.max_queue_size) 
        self.toggle_streaming = Event()

        current_dir = os.path.dirname(os.path.abspath(__file__))
        cert_file = os.path.join(current_dir, 'cert.pem')
        key_file = os.path.join(current_dir, 'key.pem')

        # Basic check for certs, can be expanded
        if not (os.path.exists(cert_file) and os.path.exists(key_file)):
            print(f"Warning: SSL cert '{cert_file}' or key '{key_file}' not found. WebRTC might have issues if not behind a proxy handling SSL.")
            
        self.tv = OpenTeleVision(
            self.resolution_cropped, 
            self.shm.name, 
            self.image_queue, 
            self.toggle_streaming,
            stream_mode=self.stream_mode, 
            cert_file=cert_file, 
            key_file=key_file, 
            ngrok=self.use_ngrok
        )
        print(f"OpenTeleVision initialized in '{self.stream_mode}' mode.")

    def process_head_tracking_and_command_actuator(self):
        try:
            # Original logic from teleop_active_cam.py
            head_mat_raw = self.tv.head_matrix[:3, :3]
            # head_mat = grd_yup2grd_zup[:3, :3].T @ head_mat_raw
            head_mat = grd_yup2grd_zup[:3, :3] @ self.tv.head_matrix[:3, :3] @ grd_yup2grd_zup[:3, :3].T
            if np.sum(head_mat) == 0:
                head_mat = np.eye(3)
            
            head_rot_quat = rotations.quaternion_from_matrix(head_mat[0:3, 0:3]) # pytransform3d uses matrix directly
            ypr = rotations.euler_from_quaternion(head_rot_quat, 2, 1, 0, False) # Assuming ZYX order for euler
            
            # Command the robot - [pitch, -yaw] based on original script's ypr[1] and -ypr[2]
            # Ensure ypr indices match expected: yaw, pitch, roll. Original code used ypr[1] (pitch) and ypr[2] (roll as yaw proxy?)
            # If ypr is [yaw, pitch, roll]: command should be [pitch, -roll] or similar.
            # Original: agent._robot.command_joint_state([ypr[1], -ypr[2]])
            # Assuming ypr = (yaw, pitch, roll) from pytransform3d euler output with 'Z', 'Y', 'X' axes and intrinsic.
            # Here, euler_from_quaternion(q, 2,1,0, False) means Extrinsic ZYX, or Intrinsic XYZ.
            # So ypr[0] is around Z (yaw-ish), ypr[1] is around Y (pitch-ish), ypr[2] is around X (roll-ish)
            # The command was agent._robot.command_joint_state([ypr[1], -ypr[2]])
            # This means joint1 gets ypr[1] (rotation about Y axis in intermediate frame)
            # And joint2 gets -ypr[2] (negative rotation about X axis in final frame)
            # This depends highly on the robot's kinematics and how ypr is interpreted.
            # Sticking to original command for now
            print(f"Head Rotation: {ypr * 180 / np.pi}")
            
            # offset the pitch angle 20 degree positive
            ypr[1] += 0.349066
            
            # setup the joint limit to avoid collision
            # yaw: -90~90deg
            if ypr[0] < -1.5708:
                ypr[0] = -1.5708
            elif ypr[0] > 1.5708:
                ypr[0] = 1.5708
            # pitch: -20~90deg 
            if ypr[1] < -0.349066:
                ypr[1] = -0.349066
            elif ypr[1] > 1.5708:
                ypr[1] = 1.5708
            print(f"Joint Command: {ypr * 180 / np.pi}")
            # the order is: yaw, pitch, roll
            # for robot, we only control the yaw and pitch
            self.agent._robot.command_joint_state([ypr[0], ypr[1]])

        except Exception as e:
            # print(f"Head tracking/actuator command error: {e}") # Avoid flooding console
            pass # Continue silently as in original

    def stream_frames(self):
        image_left = sl.Mat()
        image_right = sl.Mat()
        runtime_parameters = sl.RuntimeParameters()

        print("Starting frame streaming and actuator control loop...")
        # self.tv.print_connection_info() # OpenTeleVision does this internally now

        try:
            while True:
                start_time = time.time()

                # Process head tracking and command actuator first
                self.process_head_tracking_and_command_actuator()

                # Check streaming state (from zed_vr.py)
                stream_state = self.toggle_streaming.is_set()
                if stream_state != self.last_stream_state:
                    if stream_state:
                        print("WebRTC client connected, streaming frames...")
                        self.stream_active = True
                    else:
                        print("WebRTC client disconnected, waiting for new connection...")
                        self.stream_active = False
                    self.last_stream_state = stream_state

                if self.zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
                    self.zed.retrieve_image(image_left, sl.VIEW.LEFT)
                    self.zed.retrieve_image(image_right, sl.VIEW.RIGHT)
                    
                    bgra_left = image_left.get_data() 
                    bgra_right = image_right.get_data()

                    # Apply cropping
                    cropped_left = bgra_left[self.crop_size_h:self.resolution[0]-self.crop_size_h, self.crop_size_w:self.resolution[1]-self.crop_size_w]
                    cropped_right = bgra_right[self.crop_size_h:self.resolution[0]-self.crop_size_h, self.crop_size_w:self.resolution[1]-self.crop_size_w]
                    
                    # Hstack and convert to RGB
                    # Note: original hstack was on image_left.numpy(), etc.
                    # get_data() returns a memory view that can be treated as a numpy array
                    bgra_combined = np.hstack((cropped_left, cropped_right))
                    rgb_combined = cv2.cvtColor(bgra_combined, cv2.COLOR_BGRA2RGB)

                    # Update shared memory (for 'image' mode or direct access)
                    np.copyto(self.img_array, rgb_combined)

                    # Send frame to WebRTC if streaming is active/requested (from zed_vr.py)
                    if self.stream_mode == 'webrtc' and self.stream_active:
                        try:
                            if self.image_queue.qsize() >= self.max_queue_size:
                                try:
                                    while self.image_queue.qsize() > 0:
                                        self.image_queue.get_nowait()
                                except: # pylint: disable=bare-except
                                    pass # Queue might have emptied between qsize and get
                            self.image_queue.put_nowait(rgb_combined.copy())
                        except Exception as e: # pylint: disable=broad-except
                            if time.time() - self.last_queue_warning > 5:
                                print(f"WebRTC image queue warning: {e}")
                                self.last_queue_warning = time.time()
                    
                    self.frame_count += 1
                    # Debug: Save frame periodically
                    # if self.frame_count % 600 == 0:
                    #     debug_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "debug_active_cam")
                    #     os.makedirs(debug_path, exist_ok=True)
                    #     cv2.imwrite(os.path.join(debug_path, f"active_cam_frame_{self.frame_count}.jpg"), cv2.cvtColor(rgb_combined, cv2.COLOR_RGB2BGR))
                    #     print(f"Saved debug frame to {debug_path}")


                # Throttle loop (optional, but good for CPU management)
                elapsed = time.time() - start_time
                sleep_duration = (1/60) - elapsed # Target 60 FPS
                if sleep_duration > 0:
                    time.sleep(sleep_duration)

        except KeyboardInterrupt:
            print("Stopping ActiveCam streaming...")
        finally:
            self.cleanup()

    def run(self):
        self.stream_frames()

    def cleanup(self):
        print("Cleaning up resources...")
        if self.zed and self.zed.is_opened():
            self.zed.close()
            print("ZED Camera closed.")
        if self.shm:
            self.shm.close()
            try:
                self.shm.unlink() # Ensure unlinking only if it exists
                print("Shared memory unlinked.")
            except FileNotFoundError:
                print("Shared memory already unlinked or never created linkable.")
        if self.agent:
            # Add any necessary agent cleanup, e.g., disabling torque
            try:
                self.agent._robot.set_torque_mode(False) # Example cleanup
                print("Dynamixel torque mode disabled.")
            except Exception as e: # pylint: disable=broad-except
                print(f"Error during Dynamixel cleanup: {e}")
        print("Cleanup finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ZED Active Camera VR Streaming')
    parser.add_argument('--mode', type=str, choices=['webrtc', 'image'], default='webrtc',
                        help='Streaming mode: webrtc or image (default: webrtc)')
    parser.add_argument('--ngrok', action='store_true', default=True, # Original script had ngrok=True hardcoded for tv
                        help='Use ngrok for remote access (default: True if available in OpenTeleVision)')
    parser.add_argument('--port', type=str, default="/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTA7NOM6-if00-port0",
                        help='Dynamixel serial port')
    args = parser.parse_args()

    streamer = ActiveCamStreamer(stream_mode=args.mode, use_ngrok=args.ngrok, dynamixel_port=args.port)
    try:
        streamer.run()
    except Exception as e: # pylint: disable=broad-except
        print(f"Unhandled error in ActiveCamStreamer run: {e}")
        streamer.cleanup() # Attempt cleanup on unhandled error too