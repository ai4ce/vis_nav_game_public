__version__ = '1.2.2'

import os
import sys
import time
import json
import gdown
import zipfile
import hashlib
from enum import Enum
import argparse
import logging
import ntplib

import pybullet as PB
import pybullet_data
import numpy as np
import cv2

from vis_nav_game.interface import Action, Phase, Player
import pub as nopub


class KeyboardPlayerPyBullet(Player):
    def __init__(self):
        self.keymap = {}
        self.fpv = None
        self.last_act = Action.IDLE
        super(KeyboardPlayerPyBullet, self).__init__()

    def reset(self):
        self.fpv = None
        self.last_act = Action.IDLE

        forward_n_left = PB.B3G_UP_ARROW + PB.B3G_LEFT_ARROW
        forward_n_right = PB.B3G_UP_ARROW + PB.B3G_RIGHT_ARROW
        backward_n_left = PB.B3G_DOWN_ARROW + PB.B3G_LEFT_ARROW
        backward_n_right = PB.B3G_DOWN_ARROW + PB.B3G_RIGHT_ARROW

        self.keymap = {
            PB.B3G_LEFT_ARROW: Action.LEFT,
            PB.B3G_RIGHT_ARROW: Action.RIGHT,
            PB.B3G_UP_ARROW: Action.FORWARD,
            PB.B3G_DOWN_ARROW: Action.BACKWARD,
            PB.B3G_SPACE: Action.CHECKIN,
            forward_n_left: Action.FORWARD | Action.LEFT,
            forward_n_right: Action.FORWARD | Action.RIGHT,
            backward_n_left: Action.BACKWARD | Action.LEFT,
            backward_n_right: Action.BACKWARD | Action.RIGHT
        }

    def pre_exploration(self):
        logging.info('pre exploration')
        K = self.get_camera_intrinsic_matrix()
        logging.info(f'Camera Intrinsic Matrix K={K}')

    def act(self):
        q_key = ord('q')
        keys = PB.getKeyboardEvents()
        if q_key in keys and keys[q_key] & PB.KEY_WAS_TRIGGERED:
            return Action.QUIT

        action = Action.IDLE
        for k, v in keys.items():
            if k in self.keymap:
                if v & PB.KEY_WAS_RELEASED:
                    action = action ^ self.keymap[k]  # remove this action
                else:
                    action = action | self.keymap[k]  # add this action
            else:
                self.show_target_images()  # show target image if pressed any other keys outside the keymap
                action = Action.IDLE
        return action

    def set_target_images(self, images: list[np.ndarray]) -> None:
        super(KeyboardPlayerPyBullet, self).set_target_images(images)
        self.show_target_images()

    def show_target_images(self):
        targets = self.get_target_images()
        if targets is None or len(targets) <= 0:
            return
        concat_img = cv2.hconcat(targets)
        cv2.imshow(f'KeyboardPlayer:target_images', concat_img)
        cv2.waitKey(1)
        
    def see(self, fpv):
        self.fpv = fpv
        cv2.imshow('KeyboardPlayer:fpv', fpv)
        cv2.waitKey(1)
        return


def capture_fpv(robot_id, camera_K, bot_pos=None, bot_orn=None, camera_height=0.15, look_distance=5,
                img_width=320, img_height=240):
    def cvK2BulletP(K, w, h, near, far):
        """
        cvKtoPulletP convert the K intrinsic matrix as calibrated using Opencv
        and ROS to the projection matrix used in openGL and Pybullet.

        :param K:  OpenCV 3x3 camera intrinsic matrix
        :param w:  Image width
        :param h:  Image height
        :near:     The nearest objects to be included in the render
        :far:      The furthest objects to be included in the render
        :return:   4x4 projection matrix as used in openGL and pybullet

        note: copied from https://stackoverflow.com/a/75354854
        projectionMatrix = cvK2BulletP(K, w, h, near, far)
        viewMatrix = cvPose2BulletView(q, t)
        _, _, rgb, depth, segmentation = b.getCameraImage(W, H, viewMatrix, projectionMatrix, shadow = True)
        """
        f_x = K[0, 0]
        f_y = K[1, 1]
        c_x = K[0, 2]
        c_y = K[1, 2]
        A = (near + far) / (near - far)
        B = 2 * near * far / (near - far)

        projection_matrix = [
            [2 / w * f_x, 0, (w - 2 * c_x) / w, 0],
            [0, 2 / h * f_y, (2 * c_y - h) / h, 0],
            [0, 0, A, B],
            [0, 0, -1, 0]]
        # The transpose is needed for respecting the array structure of the OpenGL
        return np.array(projection_matrix).T.reshape(16).tolist()

    if bot_pos is None or bot_orn is None:
        if robot_id is None:
            raise ValueError('robot_id cannot be None when bot_pos or bot_orn is None!')
        bot_pos, bot_orn = PB.getBasePositionAndOrientation(robot_id)

    yaw = PB.getEulerFromQuaternion(bot_orn)[-1]
    x_a, y_a, z_a = bot_pos
    z_a = z_a + camera_height

    x_b = x_a + np.cos(yaw) * look_distance
    y_b = y_a + np.sin(yaw) * look_distance
    z_b = z_a

    view_matrix = PB.computeViewMatrix(
        cameraEyePosition=[x_a, y_a, z_a],
        cameraTargetPosition=[x_b, y_b, z_b],
        cameraUpVector=[0, 0, 1.0])

    # projection_matrix = PB.computeProjectionMatrixFOV(
    #     fov=90, aspect=1.5, nearVal=0.02, farVal=look_distance)
    if camera_K is None or camera_K.shape != (3, 3):
        raise ValueError(f'Invalid camera_K={camera_K}!')
    projection_matrix = cvK2BulletP(camera_K, img_width, img_height, 0.02, look_distance)

    # Get the camera image
    img = PB.getCameraImage(
        img_width, img_height,
        view_matrix,
        projection_matrix,
        flags=PB.ER_NO_SEGMENTATION_MASK)[2]

    # Convert the image to BGR format for opencv
    bot_fpv = np.array(img[:, :, 2::-1])  # RGBA -> BGR
    return bot_fpv, bot_pos, bot_orn


class Game:
    # following file IDs are changable during Game init
    ESSENTIAL_FILE_ID = '1CvIOxnKO8Z8NDBh-kKOpUuWEgPinLLSV'
    MAZE_FILE_ID = '1ns3RtMrL53jEERQimnUNmW5_mDqyGEe5'
    # This key is available after 'python maze.py'. The content is in the maze_md5.txt
    MAZE_FILE_MD5_KEY = '0d75aeeff036e2fe61ff32a0640a6e47'

    NAV_START_TIME = 0  # 0 means we do not enforce a common start of NAVIGATION phase
    NOISY_MOTION = 0  # 0 means no noise, 1 otherwise

    MAX_EXP_STEP = 5000000
    MAX_NAV_STEP = 10000000
    MAX_GAME_SECONDS = 60 * 60  # max 60min in total

    CAMERA_W = 320
    CAMERA_H = 240
    CAMERA_F = np.round(CAMERA_W/2.0/np.tan(np.deg2rad(60)))
    CAMERA_K = np.array([
        [CAMERA_F, 0, CAMERA_W/2.0],
        [0, CAMERA_F, CAMERA_H/2.0],
        [0, 0, 1]
    ])

    class State:
        # visible information to player
        bot_fpv: np.ndarray
        phase: Enum
        step: int
        time: float
        fps: float
        time_left: float
        # internal information hidden from player
        bot_pos: list | tuple
        bot_orn: list | tuple
        bot_action: Action

        def time_left_since(self, start_time):
            self.time = time.time()
            time_elapsed = self.time - start_time
            self.time_left = Game.MAX_GAME_SECONDS - time_elapsed
            return self.time_left

        def step_left(self):
            if self.phase == Phase.EXPLORATION:
                return Game.MAX_EXP_STEP - self.step
            else:
                return Game.MAX_NAV_STEP - self.step

        def is_done(self, start_time):
            if self.step_left() <= 0:
                return True  # player used all steps
            if self.time_left_since(start_time) <= 0:
                return True  # player time out
            if self.bot_action & Action.QUIT:
                return True  # player choose to stop
            return False

        def for_player(self):
            return self.bot_fpv, self.phase, self.step, self.time, self.fps, self.time_left

        def for_save(self):
            return [self.phase, self.step, self.time, self.fps, self.bot_action] + \
                   list(self.bot_pos) + list(self.bot_orn)

    def __init__(self, player=None, do_pybullet_gui=False, time_step=0.01, save_video=0):
        def define_robot(position):
            mass = 1.0  # Mass of the robot

            # Create a visual shape for the BOX (optional, for visualization)
            visual_shape_id = PB.createVisualShape(PB.GEOM_BOX,
                                                   halfExtents = [0.05, 0.05, 0.05],
                                                   rgbaColor=[0.1, 0.2, 0.2, 1],
                                                   specularColor=[0.4, 0.4, 0.4])

            # Create a collision shape for the BOX
            collision_shape_id = PB.createCollisionShape(PB.GEOM_BOX,
                                                         halfExtents=[0.06, 0.06, 0.06])

            # Create a rigid body (the actual object)
            robot_id = PB.createMultiBody(baseMass=mass,
                                          baseCollisionShapeIndex=collision_shape_id,
                                          baseVisualShapeIndex=visual_shape_id,
                                          basePosition=position)

            return robot_id

        logging.info(f'vis_nav_game.core version={__version__}')

        startup_file_path = os.path.join(os.getcwd(), 'startup.json')
        if os.path.exists(startup_file_path):
            with open(startup_file_path, 'r') as f:
                startup_dict = json.load(f)
                logging.info(f'init from {startup_file_path}: {startup_dict}')
                Game.MAZE_FILE_MD5_KEY = startup_dict['MAZE_FILE_MD5_KEY']
                Game.MAZE_FILE_ID = startup_dict['MAZE_FILE_ID']
                Game.ESSENTIAL_FILE_ID = startup_dict['ESSENTIAL_FILE_ID']

        self.data_dir = os.path.join(os.getcwd(), 'data')
        logging.info(f'data_dir={self.data_dir}')
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
        self.check_or_download_essential()

        if player is None:
            player = KeyboardPlayerPyBullet()
        self.player = player
        self.player.set_camera_intrinsic_matrix(Game.CAMERA_K)

        self.do_pybullet_gui = do_pybullet_gui
        self.save_video = save_video  # 0 means no, 1 means navigation phase only, 2 means both phases

        self.pid = PB.connect(PB.GUI if self.do_pybullet_gui else PB.DIRECT)
        if do_pybullet_gui:
            # set BEV view
            camera_target_position = [3, 3, -6]
            if 'CAMERA_TARGET_POSITION' in startup_dict:
                camera_target_position = startup_dict['CAMERA_TARGET_POSITION']
            PB.resetDebugVisualizerCamera(
                cameraDistance=10,
                cameraYaw=180,
                cameraPitch=-89.8,
                cameraTargetPosition=camera_target_position
            )
            PB.configureDebugVisualizer(PB.COV_ENABLE_GUI, 0)
            PB.configureDebugVisualizer(PB.COV_ENABLE_MOUSE_PICKING, 0)
            PB.configureDebugVisualizer(PB.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
            PB.configureDebugVisualizer(PB.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)
        PB.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.plane = PB.loadURDF("plane.urdf")

        self.bot_reset_pos = [0.2, 2.6, 0.05]
        self.bot = define_robot(self.bot_reset_pos)

        self.panel_sdf_path = os.path.join(self.data_dir, "models/panel_models/model_{}/model.sdf")
        self.textures_path = os.path.join(self.data_dir, 'textures')

        self.target_pose = None
        self.target_images = None

        PB.setGravity(0, 0, -9.8)

        self.state = Game.State()
        self.start_time = None

        self.time_step = time_step
        if time_step <= 0:
            PB.setRealTimeSimulation(1)
        else:
            PB.setTimeStep(self.time_step)

        self.data = []
        self.save_dir = os.path.join(self.data_dir, 'save')
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        logging.info(f'save_dir={self.save_dir}')

    def get_result(self, bot_pos, bot_orn):
        bot_pose = np.asarray([bot_pos[0], bot_pos[1], PB.getEulerFromQuaternion(bot_orn)[-1]])

        trans_error = np.linalg.norm(self.target_pose[:2] - bot_pose[:2])
        rot_error = np.abs(self.target_pose[-1] - bot_pose[-1])
        if rot_error > np.pi:
            rot_error = 2 * np.pi - rot_error

        return trans_error, rot_error * 180.0 / np.pi

    def check_or_download_essential(self):
        def check_folder(folder_name):
            return os.path.exists(folder_name) and os.path.isdir(folder_name)

        def check_file(file_name):
            return os.path.exists(file_name) and os.path.isfile(file_name)

        def download_essential():
            url = f'https://drive.google.com/uc?id={Game.ESSENTIAL_FILE_ID}'

            output_file = os.path.join(self.data_dir, 'data.zip')
            gdown.download(url, output_file, quiet=True)

            extract_path = '.'
            with zipfile.ZipFile(output_file, 'r') as zip_ref:
                # Extract all contents to the target directory
                zip_ref.extractall(extract_path)

            os.remove(output_file)

            logging.info("Essentials extraction complete.")

        def verify_md5(file_path):
            md5_hash = hashlib.md5()
            with open(file_path, 'rb') as file:
                for chunk in iter(lambda: file.read(4096), b''):
                    md5_hash.update(chunk)

            return Game.MAZE_FILE_MD5_KEY == md5_hash.hexdigest()

        def download_maze():
            file_url = f'https://drive.google.com/uc?id={Game.MAZE_FILE_ID}'
            output_file = os.path.join(self.data_dir, 'maze.json')
            gdown.download(file_url, output_file, quiet=True)

        check_status = True
        folder_names = ["models", "textures"]

        for folder_name in folder_names:
            if check_folder(os.path.join(self.data_dir, folder_name)):
                pass
            else:
                logging.debug(f"Folder '{folder_name}' does not exist.")
                check_status = False

        if check_status:
            logging.info("Pass checking essential files")
        else:
            logging.info("Missing essentials, downloading the files...")
            download_essential()

        #Check for maze file
        maze_file = os.path.join(self.data_dir, "maze.json")
        if not check_file(maze_file):
            logging.info(f"Missing {maze_file}, downloading it...")
            download_maze()
        elif not verify_md5(maze_file):
            logging.debug(f"{maze_file} did not pass MD5 test, downloading the correct one...")
            download_maze()
        else:
            logging.info(f"Verified {maze_file}")

    def replay(self, game_file_path):
        save_path = game_file_path[:-3]+'avi'

        video = None

        game = nopub.load_game_file(game_file_path)
        if len(game.shape) == 0:
            game = game.item()
            game_version = game['version']

            def warn_msg(s):
                return f'{s}={game[s]} saved in {game_file_path} does not match the one used in ' \
                       f'vis_nav_game.core ({getattr(self, s)})'

            if game['ESSENTIAL_FILE_ID'] != Game.ESSENTIAL_FILE_ID:
                raise Warning(warn_msg('ESSENTIAL_FILE_ID'))
            if game['MAZE_FILE_ID'] != Game.MAZE_FILE_ID:
                raise Warning(warn_msg('MAZE_FILE_ID'))
            if game['MAZE_FILE_MD5_KEY'] != Game.MAZE_FILE_MD5_KEY:
                raise Warning(warn_msg('MAZE_FILE_MD5_KEY'))

            game = np.asarray(game['data'])
        else:  # legacy from version 1.1.0
            game_version = '1.1.0'

        logging.info(f'reading data recorded from vis_nav_game.core version={game_version}')

        if game.shape[0] == 0:
            raise ValueError(f'No data loaded from {game_file_path}!')

        def draw_txt(txt, img_width):
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            font_color = (0, 0, 0)
            font_thickness = 1
            text_size, _ = cv2.getTextSize(txt, font, font_scale, font_thickness)
            text_height = text_size[1] + 10  # Add some padding

            text_image = np.zeros((text_height, img_width, 3), dtype=np.uint8) + 180
            text_x = 10
            text_y = text_size[1] + 5

            cv2.putText(text_image, txt, (text_x, text_y), font, font_scale, font_color, font_thickness,
                        lineType=cv2.LINE_AA)
            return text_image

        if self.save_video == 1:
            game_phase = game[:, 0]
            itemindex = np.where( game_phase==Phase.NAVIGATION )
            game = game[itemindex[0][0]:]

        for i in range(game.shape[0]):
            phase_i, step_i, time_i, fps_i, bot_action_i = game[i, :5]
            bot_pos_i, bot_orn_i = game[i, 5:8], game[i, 8:]
            self.set_pose(bot_pos_i, bot_orn_i)
            bot_fpv, _, _ = capture_fpv(None, Game.CAMERA_K, bot_pos=bot_pos_i, bot_orn=bot_orn_i)

            txt1 = time.strftime('%Y/%m/%d %H:%M:%S.', time.localtime(time_i)) + str(time_i).split('.')[1][:3]
            txt2 = f"{phase_i.name} | step={step_i}"
            txt3 = f'action={str(bot_action_i).split(".")[1]} | fps={fps_i:.1f}'

            img_height, img_width, _ = bot_fpv.shape

            txt1_img = draw_txt(txt1, img_width)
            txt2_img = draw_txt(txt2, img_width)
            txt3_img = draw_txt(txt3, img_width)

            # Concatenate the text_image with the original image vertically
            result_image = np.vstack((txt1_img, txt2_img, txt3_img, bot_fpv)).astype(np.uint8)

            if self.save_video:
                if video is None:
                    rimg_h, rimg_w, _ = result_image.shape
                    average_fps = np.round(game.shape[0] / np.abs(game[-1, 2] - game[0, 2]))
                    video = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'MJPG'), average_fps, (rimg_w, rimg_h))
                    if not video.isOpened():
                        raise RuntimeError('OpenCV Video Writer not opened!')
                video.write(result_image)

        if video is not None:
            video.release()

        return self.get_result(bot_pos_i, bot_orn_i)

    def set_texture(self, object_id, model_id, selected_texture_index):
        texture = PB.loadTexture(self.textures_path + "/pattern_{}.png".format(selected_texture_index))

        if model_id == 1:
            PB.changeVisualShape(objectUniqueId=object_id[0], linkIndex=-1, textureUniqueId=texture)
        elif 1 < model_id < 5:
            PB.changeVisualShape(objectUniqueId=object_id[0], linkIndex=-1, textureUniqueId=texture)
            PB.changeVisualShape(objectUniqueId=object_id[0], linkIndex=0, textureUniqueId=texture)

    def set_pose(self, pos, ori):
        if len(ori) == 3:  # if ori is not Quaternion, convert it
            ori = PB.getQuaternionFromEuler(ori)
        PB.resetBasePositionAndOrientation(self.bot, pos, ori)

    def capture_panorama(self):
        self.set_pose(self.target_pose, [0.0, 0.0, 0.0])
        time.sleep(0.5)
        self.target_images.append(capture_fpv(self.bot, Game.CAMERA_K)[0])

        self.set_pose(self.target_pose, [0.0, 0.0, np.deg2rad(90.0)])
        time.sleep(0.5)
        self.target_images.append(capture_fpv(self.bot, Game.CAMERA_K)[0])

        self.set_pose(self.target_pose, [0.0, 0.0, np.deg2rad(180.0)])
        time.sleep(0.5)
        self.target_images.append(capture_fpv(self.bot, Game.CAMERA_K)[0])

        self.set_pose(self.target_pose, [0.0, 0.0, np.deg2rad(270.0)])
        time.sleep(0.5)
        self.target_images.append(capture_fpv(self.bot, Game.CAMERA_K)[0])

        logging.info("Target Images processing completed.")
        # text = input("press enter to continue...")

        self.set_pose(self.bot_reset_pos, [0.0, 0.0, 0.0])

    def load_data(self):
        maze_path = os.path.join(self.data_dir, 'maze.json')
        if not os.path.exists(maze_path):
            logging.error(f"The file '{maze_path}' does not exist.")
            exit(-1)

        logging.info("Start loading the map")
        start = time.time()

        # Read the JSON data from the file
        with open(maze_path, 'r') as file:
            combined_json = file.read()

        combined_data = json.loads(combined_json)

        panel_decrypted_data, target_decrypted_data, rules_data = nopub.load_and_decrypt_data(combined_data)
        if rules_data is not None:
            rules_dict = json.loads(rules_data)
            msg = f'Init rules from {rules_dict}'
            logging.info(msg)
            print(msg)
            if 'NAV_START_TIME' in rules_dict:
                Game.NAV_START_TIME = rules_dict['NAV_START_TIME']  # this should be a floating point representing time
            if 'NOISY_MOTION' in rules_dict:
                Game.NOISY_MOTION = rules_dict['NOISY_MOTION']
            if Game.NAV_START_TIME > 0:
                try:
                    nav_start_time_str = time.strftime('%Y/%m/%d %H:%M:%S', time.localtime(Game.NAV_START_TIME))
                    logging.info('NAV_START_TIME set to ' + nav_start_time_str)
                except ... as e:
                    logging.error('Sometime wrong about the NAV_START_TIME: ' + str(e))
                    exit(-1)

                system_time = time.time()
                internet_time = 0
                try:
                    internet_time = ntplib.NTPClient().request('pool.ntp.org').tx_time
                except ... as e:
                    logging.error('Failed to get Internet time, due to: ' + str(e))
                    exit(-1)

                time_diff = np.abs(system_time - internet_time)
                logging.info(f'System time={system_time}, Internet time={internet_time}, diff={time_diff}')
                if np.abs(system_time - internet_time) >= 30:
                    logging.error('System time and Internet time differs more than 30 seconds!')
                    exit(-1)
            if Game.NOISY_MOTION != 0:
                logging.info('NOISY_MOTION set to ' + str(Game.NOISY_MOTION))
            logging.info("Loaded rules for competition!")
        else:
            logging.info('No rules loaded from the maze.json!')

        maze_info = np.array(json.loads(panel_decrypted_data))
        self.target_pose = np.array(json.loads(target_decrypted_data))

        for panel in maze_info:
            model_type = int(panel[0])
            texture_index = int(panel[1])
            setPos = panel[2:5]
            setOrientation = PB.getQuaternionFromEuler(panel[5:])
            model_x_ID = PB.loadSDF(sdfFileName=self.panel_sdf_path.format(model_type))

            PB.resetBasePositionAndOrientation(model_x_ID[0], setPos, setOrientation)
            self.set_texture(model_x_ID, model_type, texture_index)

        duration = time.time() - start
        logging.info("Finished loading the map: takes {} s".format(duration))

        self.target_images = []
        self.capture_panorama()

    def save(self, file_path=None) -> bool:
        """
        save the whole game with all GT states in each simulation step from begin to the end.
        :param file_path: if None, this just append the information to an internal list; otherwise, save list to file
        :return: True if no errors
        """
        if not file_path:
            self.data.append(self.state.for_save())
            return True
        else:
            try:
                logging.info("Saving the game, please wait...")
                data_dict = {
                    'data': self.data,
                    'version':__version__,
                    'MAZE_FILE_ID': Game.MAZE_FILE_ID,
                    'ESSENTIAL_FILE_ID': Game.ESSENTIAL_FILE_ID,
                    'MAZE_FILE_MD5_KEY': Game.MAZE_FILE_MD5_KEY
                }
                nopub.save_game_file(file_path, data_dict)
                logging.info(f"Game saved successfully at {file_path}.")
                return True
            except Exception as e:
                logging.error("Error saving game.")
                logging.error(f"Exception: {e}")
                return False

    def check_in(self):
        if self.state.phase == Phase.EXPLORATION:
            return  # no effect

        # Check-in during Navigation phase
        self.state.bot_action |= Action.QUIT
        return

    def perform(self, action: Action) -> None:
        """
        do the action selected by the player
        :param action:
        :return: None
        """
        if action & Action.QUIT:
            return

        if action & Action.CHECKIN:
            self.check_in()
            return

        mode = 'reset'
        do_add_noise = Game.NOISY_MOTION

        if mode == 'reset':
            linear_speed = np.random.randn()*0.3*do_add_noise + 3
            angular_speed = np.random.randn()*0.1*do_add_noise + 6
            v = 0
            w = 0
            _, orn = PB.getBasePositionAndOrientation(self.bot)
            yaw = PB.getEulerFromQuaternion(orn)[-1]

            V = np.array([np.cos(yaw), np.sin(yaw), 0.0])
            W = np.array([0.0, 0.0, 1.0])

            if action & Action.IDLE:
                pass
            if action & Action.FORWARD:
                v += linear_speed
            if action & Action.BACKWARD:
                v += -linear_speed
            if action & Action.LEFT:
                if action & Action.BACKWARD:
                    w += -angular_speed
                else:
                    w += angular_speed
            if action & Action.RIGHT:
                if action & Action.BACKWARD:
                    w += angular_speed
                else:
                    w += -angular_speed

            PB.resetBaseVelocity(self.bot, linearVelocity=v*V, angularVelocity=w*W)
        else:
            forward = 0
            turn = 0

            left = 0
            right = 0
            speed = 20
            forward_speed_factor = 1.5

            if action & Action.IDLE:
                turn = 0
                forward = 0
            if action & Action.FORWARD:
                forward = 1
            if action & Action.BACKWARD:
                forward = -1
            if action & Action.LEFT:
                if action & Action.BACKWARD:
                    turn = -0.6
                else:
                    turn = 0.6
            if action & Action.RIGHT:
                if action & Action.BACKWARD:
                    turn = 0.6
                else:
                    turn = -0.6

            right += (forward * forward_speed_factor + turn) * speed
            left += (forward * forward_speed_factor - turn) * speed

            PB.setJointMotorControlArray(self.bot, [0, 1], PB.VELOCITY_CONTROL, targetVelocities=[left, right],
                                         forces=[2000, 2000])

    def run(self):
        self.load_data()

        # ready to start the clock for players
        self.start_time = time.time()

        # phase 1: player explores environment
        # check if it's valid time for exploration, otherwise, proceed to nav check
        self.player.pre_exploration()

        curr_time = time.time()
        if (Game.NAV_START_TIME == 0) or (curr_time < Game.NAV_START_TIME):
            self.run_phase(Phase.EXPLORATION)
        else:  # past exploration time!
            msg = "EXPLORATION phase passed. Proceeding to NAVIGATION phase."
            logging.info(msg)
            print(msg)

        # wait until NAV_START_TIME to start NAVIGATION phase
        while (Game.NAV_START_TIME > 0) and (time.time() < Game.NAV_START_TIME):
            curr_time = time.time()
            time_left = np.ceil(Game.NAV_START_TIME - curr_time)
            msg = f"NAVIGATION phase has not start yet. " \
                  f"Wait for {time_left} seconds!"
            logging.info(msg)
            print(msg)
            time.sleep(time_left)

        self.set_pose(self.bot_reset_pos, [0.0, 0.0, 0.0])  # set bot back to where we start
        self.player.set_target_images(self.target_images)  # set target for navigation

        # phase 2: player navigates robot to targets
        self.player.pre_navigation()

        self.run_phase(Phase.NAVIGATION)

        # done, save game data
        PB.removeBody(self.bot)  # remove robot from simulation
        self.save(os.path.join(self.save_dir, 'game-'+time.strftime('%Y%m%d-%H%M%S')+'.npy'))

    def run_phase(self, phase: Phase):
        self.state.phase = phase
        self.state.step = 0
        self.state.fps = -1
        self.state.time = time.time()
        self.state.bot_action = Action.IDLE
        while not self.state.is_done(self.start_time):
            self.state.time = time.time()

            self.state.bot_fpv, self.state.bot_pos, self.state.bot_orn = capture_fpv(self.bot, Game.CAMERA_K)
            self.player.see(self.state.bot_fpv)
            self.state.bot_action = self.player.act()
            self.perform(self.state.bot_action)

            PB.stepSimulation()

            duration = time.time() - self.state.time
            self.state.fps = 1.0 / duration

            self.player.set_state(self.state.for_player())
            self.save()  # pushback the current step's game state
            logging.debug(f'{phase.name}'
                  f' step={self.state.step:>5}'
                  f' fps={self.state.fps:>7.2f}Hz'
                  f' time-left={self.state.time_left_since(self.start_time):>7.1f}s')

            self.state.step += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(sys.argv[0])
    parser.add_argument('-m', '--mode', type=str, default='player', help='mode to start the server in')
    parser.add_argument('-s', '--save', type=int, default=0, help='whether to save the video or not')
    parser.add_argument('-i', '--gui', type=int, default=1, help='whether to open GUI or not')
    parser.add_argument('-f', '--file', type=str, help='game.npy file for reply')
    opt = parser.parse_args(sys.argv[1:])

    logging.basicConfig(filename='vis_nav_game.log', filemode='w', level=logging.DEBUG,
                        format='%(asctime)s - %(levelname)s: %(message)s', datefmt='%d-%b-%y %H:%M:%S')

    if opt.mode == 'player':
        sim = Game(do_pybullet_gui=(opt.gui == 1), save_video=opt.save)
        sim.run()
    elif opt.mode == 'judge':
        sim = Game(do_pybullet_gui=(opt.gui == 1), save_video=opt.save)
        sim.load_data()
        trans_error, rot_error = sim.replay(opt.file)
        print(f"Translational Error : {trans_error}")
        print(f"Rotational Error : {rot_error}")
    else:
        print("Invalid Mode!")
