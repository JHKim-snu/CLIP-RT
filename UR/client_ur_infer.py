import sys
import cv2
import base64
import socket
import numpy as np
from RT.ur_controller import RobotController
import time
import urx
import argparse

# socket
HOST = ""  # Server Host
PORT = 9998

# Bounding Area (m) Info for the robot safety
max_x = 0.27
min_x = -0.2
max_y = 0.27
min_y = -0.5
max_z = 0.2
min_z = -0.27


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", dest="mode", action="store", default="safe")
    args = parser.parse_args()
    
    robot = urx.Robot("192.168.1.117")  # Network IP Address
    time.sleep(0.2)
    print("roobt successfully connected")
    cli_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    cli_sock.connect((HOST, PORT))
    # rospy.init_node('rt0_client', anonymous=False, disable_signals=True)

    rc = RobotController()
    # rospy.loginfo('Start')

    while 1:
        rc.arm.to_home_pose()
        rc.arm.finger_open()

        del_xyz_sum = [0, 0, 0]

        input("Press any key to start the task")

        while 1:
            try:
                capture = rc.camera.get_capture()
                cv_img = capture.color[:, :, :3]

                # send captured image
                _, buf = cv2.imencode(
                    ".png", cv_img
                )  # , [cv2.IMWRITE_PNG_STRATEGY_DEFAULT]
                data = np.array(buf)
                #
                b64_data = base64.b64encode(data)
                length = str(len(b64_data))
                cli_sock.sendall(length.encode("utf-8").ljust(64))
                cli_sock.sendall(b64_data)
                # rospy.loginfo('Wait for the server')

                # get action
                data = cli_sock.recv(1024)
                str_data = data.decode()
                action = str_data.strip().split(";")
                action = [float(v) for v in action]
                del_xyz_sum += action[:3]

                if min_x > del_xyz_sum[0]:
                    action[0] += min_x - del_xyz_sum[0]
                    del_xyz_sum[0] = min_x
                if max_x > del_xyz_sum[0]:
                    action[0] += max_x - del_xyz_sum[0]
                    del_xyz_sum[0] = max_x
                if min_y > del_xyz_sum[1]:
                    action[1] += min_y - del_xyz_sum[1]
                    del_xyz_sum[1] = min_y
                if max_y > del_xyz_sum[1]:
                    action[1] += max_y - del_xyz_sum[1]
                    del_xyz_sum[1] = max_y
                if min_z > del_xyz_sum[2]:
                    action[2] += min_z - del_xyz_sum[2]
                    del_xyz_sum[2] = min_z
                if max_z > del_xyz_sum[2]:
                    action[2] += max_z - del_xyz_sum[2]
                    del_xyz_sum[2] = max_z

                rc.arm.move_gripper(action)
                time.sleep(1)
                
                if args.mode == "safe":
                    terminate = input("press any key to continue or press "t" to terminate")
                    if terminate == "t":
                        break
                
                elif args.mode == "cont":
                    continue
                

            except:  # KeyboardInterrupt
                rc.camera.stop()
                break


if __name__ == "__main__":
    main()
