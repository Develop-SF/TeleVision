import math
import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from pytransform3d import rotations
import logging

from dynamixel.active_cam import DynamixelAgent, DynamixelRobotConfig
from dynamixel_sdk.packet_handler import PacketHandler
from dynamixel_sdk.port_handler import PortHandler

# Constants from driver.py
ADDR_PRESENT_POSITION = 132
LEN_PRESENT_POSITION = 4

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("dynamixel_test.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("dynamixel_test")

# Create a custom config with the specified joint offsets as requested (2*np.pi/2 which equals pi)
custom_config = DynamixelRobotConfig(
    joint_ids=(1, 2),
    joint_offsets=(2*np.pi/2, 2*np.pi/2),  # pi, pi
    joint_signs=(1, 1),
    gripper_config=None,
)

# Define a direct read function that doesn't use the problematic get_joints method
def read_motor_positions_directly(port_handler, packet_handler, motor_ids):
    """Read motor positions directly using the low-level API instead of get_joints"""
    positions = []
    for motor_id in motor_ids:
        try:
            # Direct read of present position using the packet handler
            position_value, result, error = packet_handler.read4ByteTxRx(
                port_handler, motor_id, ADDR_PRESENT_POSITION
            )
            if result != 0 or error != 0:
                logger.error(f"Failed to read position from motor {motor_id}: result={result}, error={error}")
                positions.append(float('nan'))
            else:
                # Convert to radians (same conversion as in the driver)
                position_rad = position_value / 2048.0 * np.pi
                positions.append(position_rad)
        except Exception as e:
            logger.error(f"Exception reading motor {motor_id}: {str(e)}")
            positions.append(float('nan'))
    
    return np.array(positions)

# Initialize the Dynamixel agent
try:
    agent = DynamixelAgent(
        port="/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTA7NOM6-if00-port0",
        dynamixel_config=custom_config
    )
    logger.info("Successfully connected to Dynamixel agent")
    
    # Get direct access to the port and packet handlers for our direct reads
    port_handler = agent._robot._driver._portHandler
    packet_handler = agent._robot._driver._packetHandler
    motor_ids = agent._robot._joint_ids
    
    # Enable torque
    logger.info("Enabling torque mode...")
    agent._robot.set_torque_mode(True)
    logger.info("Torque mode enabled successfully")
except Exception as e:
    logger.error(f"Failed to initialize Dynamixel agent: {str(e)}")
    logger.error("Check if the motor is properly connected and powered")
    raise

def test_single_motor_rotation(motor_idx, angles):
    """Test rotation of a single motor and log commands vs positions."""
    motor_name = f"Motor {motor_idx+1}"
    logger.info(f"Testing {motor_name} rotation through {len(angles)} positions")
    
    results = []
    for angle in angles:
        try:
            # Create command array (zeros except for the tested motor)
            cmd = np.zeros(2)
            cmd[motor_idx] = angle
            
            # Log command
            logger.info(f"{motor_name} Command: {angle:.4f} rad")
            
            # Send command to motor
            logger.info(f"Sending command to motor...")
            agent._robot.command_joint_state(cmd)
            logger.info(f"Command sent successfully")
            
            # Wait for motor to reach position
            time.sleep(0.5)
            
            # Read directly from the motors
            logger.info(f"Reading joint position directly...")
            try:
                positions = read_motor_positions_directly(port_handler, packet_handler, motor_ids)
                # Apply offset and sign correction (similar to what get_joint_state does)
                positions = (positions - agent._robot._joint_offsets) * agent._robot._joint_signs
                logger.info(f"Motor positions: {positions}")
                position = positions[motor_idx]
                logger.info(f"{motor_name} Position: {position:.4f} rad")
                results.append((angle, position))
            except Exception as e:
                logger.error(f"Error reading position directly: {str(e)}")
                results.append((angle, float('nan')))
        
        except Exception as e:
            logger.error(f"Exception in motor test loop: {str(e)}")
            results.append((angle, float('nan')))
    
    return results

def apply_rotation_transform(yaw, pitch):
    """Apply rotation transform to set target rotation."""
    # This simulates what happens in teleop_active_cam.py when applying
    # rotation transform from head matrix
    
    logger.info(f"Applying rotation transform - yaw: {yaw:.4f}, pitch: {pitch:.4f}")
    
    try:
        # Create rotation command (similar to what's done with head rotation in teleop_active_cam.py)
        cmd = np.array([yaw, pitch])
        
        # Log command
        logger.info(f"Combined rotation command: {cmd}")
        
        # Apply command
        logger.info("Sending rotation command...")
        agent._robot.command_joint_state(cmd)
        logger.info("Rotation command sent successfully")
        
        # Wait for motors to reach position
        time.sleep(1.0)
        
        # Read directly from the motors
        logger.info("Reading final position directly...")
        try:
            positions = read_motor_positions_directly(port_handler, packet_handler, motor_ids)
            # Apply offset and sign correction (similar to what get_joint_state does)
            positions = (positions - agent._robot._joint_offsets) * agent._robot._joint_signs
            logger.info(f"Final position: {positions}")
            return positions
        except Exception as e:
            logger.error(f"Error reading final position directly: {str(e)}")
            return np.array([float('nan'), float('nan')])
    
    except Exception as e:
        logger.error(f"Exception in apply_rotation_transform: {str(e)}")
        return np.array([float('nan'), float('nan')])

def plot_results(motor1_results, motor2_results):
    """Plot the command vs position results."""
    plt.figure(figsize=(12, 6))
    
    # Plot Motor 1 results
    plt.subplot(1, 2, 1)
    commands, positions = zip(*motor1_results)
    plt.plot(commands, positions, 'bo-')
    plt.plot(commands, commands, 'r--')  # Ideal line
    plt.xlabel('Command (rad)')
    plt.ylabel('Position (rad)')
    plt.title('Motor 1: Command vs Position')
    plt.grid(True)
    
    # Plot Motor 2 results
    plt.subplot(1, 2, 2)
    commands, positions = zip(*motor2_results)
    plt.plot(commands, positions, 'go-')
    plt.plot(commands, commands, 'r--')  # Ideal line
    plt.xlabel('Command (rad)')
    plt.ylabel('Position (rad)')
    plt.title('Motor 2: Command vs Position')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('dynamixel_test_results.png')
    logger.info("Results plot saved to dynamixel_test_results.png")

def main():
    try:
        # Check if motors are responsive
        logger.info("Checking motor communication...")
        try:
            initial_positions = read_motor_positions_directly(port_handler, packet_handler, motor_ids)
            # Apply offset and sign correction (similar to what get_joint_state does)
            initial_positions = (initial_positions - agent._robot._joint_offsets) * agent._robot._joint_signs
            logger.info(f"Initial motor positions: {initial_positions}")
        except Exception as e:
            logger.error(f"Error checking initial motor positions: {str(e)}")
            logger.error("Motor communication failed, check hardware connections")
            return

        # Test Motor 1 (yaw) rotation
        logger.info("=== Testing Motor 1 (Yaw) ===")
        test_angles = np.linspace(-np.pi/4, np.pi/4, 10)
        motor1_results = test_single_motor_rotation(0, test_angles)
        
        # Reset position
        logger.info("Resetting position to [0, 0]...")
        agent._robot.command_joint_state([0, 0])
        time.sleep(1.0)
        
        # Test Motor 2 (pitch) rotation
        logger.info("=== Testing Motor 2 (Pitch) ===")
        test_angles = np.linspace(-np.pi/4, np.pi/4, 10)
        motor2_results = test_single_motor_rotation(1, test_angles)
        
        # Reset position
        logger.info("Resetting position to [0, 0]...")
        agent._robot.command_joint_state([0, 0])
        time.sleep(1.0)
        
        # Test applying rotation transform
        logger.info("=== Testing Rotation Transform ===")
        target_yaw = np.pi/6    # 30 degrees
        target_pitch = np.pi/4  # 45 degrees
        final_position = apply_rotation_transform(target_yaw, target_pitch)
        
        # Check if there are valid results to plot
        all_nan_motor1 = all(np.isnan(pos) for _, pos in motor1_results)
        all_nan_motor2 = all(np.isnan(pos) for _, pos in motor2_results)
        
        if not all_nan_motor1 and not all_nan_motor2:
            # Plot results
            logger.info("Plotting results...")
            plot_results(motor1_results, motor2_results)
            logger.info("Test completed successfully")
        else:
            logger.error("Cannot plot results, no valid position data collected")
    
    except Exception as e:
        logger.error(f"Unexpected error in main routine: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
    
    finally:
        # Always disable torque when done
        logger.info("Disabling torque...")
        try:
            agent._robot.set_torque_mode(False)
            logger.info("Torque disabled successfully")
        except Exception as e:
            logger.error(f"Error disabling torque: {str(e)}")

if __name__ == "__main__":
    main()
