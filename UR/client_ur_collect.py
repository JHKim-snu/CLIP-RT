import sys
import cv2
import base64
import socket
import numpy as np
from ur_controller import RobotController
import time
import urx
import json
from datetime import datetime
import os


# socket
HOST = "147.47.200.171"
PORT = 9997

# Bounding Area (m) Info for the robot safety
max_x = 0.27
min_x = -0.2
max_y = 0.27
min_y = -0.5
max_z = 0.2
min_z = -0.27

TASK_NUM = input("Insert task number...")

image_root_path = "./data/train/images/task_{}/".format(TASK_NUM)
if not os.path.exists(image_root_path):
    os.makedirs(image_root_path)
    print(f"Directory '{image_root_path}' created.")

steps_root_path = "./data/train/steps/task_{}/".format(TASK_NUM)
if not os.path.exists(steps_root_path):
    os.makedirs(steps_root_path)
    print(f"Directory '{steps_root_path}' created.")


def main():

    existing_epi = os.listdir(image_root_path)
    if len(existing_epi) == 0:
        print("Starting new task! Good luck")
        EPISODE_NUM = 0
    else:
        EPISODE_NUM = max([int(k[-1]) for k in existing_epi]) + 1

    robot = urx.Robot("192.168.1.117")  # Network IP Address
    time.sleep(0.2)
    print("roobt successfully connected")

    cli_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    cli_sock.connect((HOST, PORT))
    # rospy.init_node('rt0_client', anonymous=False, disable_signals=True)

    rc = RobotController()
    # rospy.loginfo('Start')
    rc.arm.to_home_pose()
    rc.arm.finger_open()

    del_xyz_sum = [0, 0, 0]

    while 1:
        single_data = {}
        try:
            input("Press any key to get images")
            capture = rc.camera.get_capture()
            cv_img = capture.color[:, :, :3]
            dt = datetime.now()
            IMAGE_PATH = os.path.join(image_root_path, "episode_{}".format(EPISODE_NUM))
            save_image_path = os.path.join(
                IMAGE_PATH,
                "{}_{}_{}_{}_{}_{}.png".format(
                    dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second
                ),
            )

            STEPS_PATH = os.path.join(steps_root_path, "episode_{}".format(EPISODE_NUM))
            save_steps_path = os.path.join(
                STEPS_PATH,
                "{}_{}_{}_{}_{}_{}.json".format(
                    dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second
                ),
            )

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
            print("data received from server... \n{}".format(str_data))
            data_list = str_data.strip().split(";")
            instruction = data_list[-2]
            supervision = data_list[-1]
            action = data_list[:-2]
            assert len(action) == 7 or 8
            action = [float(v) for v in action]
            del_xyz_sum = [a + b for a, b in zip(del_xyz_sum, action[:3])]

            prev_eef_pose = rc.arm.get_pose()
            prev_joint = rc.arm.get_joint()

            print("supervision: {}".format(action))
            print("expected delta coord.: {}".format(del_xyz_sum))

            if min_x > del_xyz_sum[0]:
                print("required action out-of-box (min_x)")
                action[0] += min_x - del_xyz_sum[0]
                del_xyz_sum[0] = min_x
            if max_x < del_xyz_sum[0]:
                print("required action out-of-box (max_x)")
                action[0] += max_x - del_xyz_sum[0]
                del_xyz_sum[0] = max_x
            if min_y > del_xyz_sum[1]:
                print("required action out-of-box (min_y)")
                action[1] += min_y - del_xyz_sum[1]
                del_xyz_sum[1] = min_y
            if max_y < del_xyz_sum[1]:
                print("required action out-of-box (max_y)")
                action[1] += max_y - del_xyz_sum[1]
                del_xyz_sum[1] = max_y
            if min_z > del_xyz_sum[2]:
                print("required action out-of-box (min_z)")
                action[2] += min_z - del_xyz_sum[2]
                del_xyz_sum[2] = min_z
            if max_z < del_xyz_sum[2]:
                print("required action out-of-box (max_z)")
                action[2] += max_z - del_xyz_sum[2]
                del_xyz_sum[2] = max_z

            print("bounded supervised action: {}".format(action))
            print("accumulated delta coord.: {}".format(del_xyz_sum))

            if supervision != "done":
                rc.arm.move_gripper(action)
            else:
                print("task_{}, episode_{} done \n".format(TASK_NUM, EPISODE_NUM))
                EPISODE_NUM += 1
                del_xyz_sum = [0, 0, 0]
                rc.arm.to_home_pose()
                rc.arm.finger_open()
                # time.sleep(1.0)

            eef_pose = rc.arm.get_pose()
            joint = rc.arm.get_joint()

            single_data["prev_eef_pose"] = prev_eef_pose
            single_data["prev_joint"] = prev_joint
            single_data["eef_pose"] = eef_pose
            single_data["joint"] = joint
            single_data["instruction"] = instruction
            single_data["supervision"] = supervision
            single_data["action"] = action
            single_data["image_path"] = save_image_path

            save_flag = input("Save data? [y/n]")
            if save_flag == "y":
                if not os.path.exists(STEPS_PATH):
                    os.makedirs(STEPS_PATH)
                    print(f"Directory '{STEPS_PATH}' created.")
                if not os.path.exists(IMAGE_PATH):
                    os.makedirs(IMAGE_PATH)
                    print(f"Directory '{IMAGE_PATH}' created.")
                with open(save_steps_path, "w") as f:
                    json.dump(single_data, f)
                cv2.imwrite(save_image_path, cv_img)
        except:  # KeyboardInterrupt
            rc.camera.stop()
            break

    cli_sock.close()


if __name__ == "__main__":
    main()
