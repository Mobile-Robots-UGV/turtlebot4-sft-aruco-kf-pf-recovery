#!/usr/bin/env python3
import json
import math
from pathlib import Path

import cv2
import numpy as np
import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PoseStamped, Vector3Stamped, TransformStamped
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Bool, Int32MultiArray
from tf2_ros import TransformBroadcaster


DICT_MAP = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
    "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
    "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
    "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
    "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
    "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
    "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
    "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
    "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
}


def rvec_to_quaternion(rvec: np.ndarray) -> np.ndarray:
    rot_mtx, _ = cv2.Rodrigues(rvec)
    trace = np.trace(rot_mtx)

    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (rot_mtx[2, 1] - rot_mtx[1, 2]) / s
        qy = (rot_mtx[0, 2] - rot_mtx[2, 0]) / s
        qz = (rot_mtx[1, 0] - rot_mtx[0, 1]) / s
    elif rot_mtx[0, 0] > rot_mtx[1, 1] and rot_mtx[0, 0] > rot_mtx[2, 2]:
        s = math.sqrt(1.0 + rot_mtx[0, 0] - rot_mtx[1, 1] - rot_mtx[2, 2]) * 2.0
        qw = (rot_mtx[2, 1] - rot_mtx[1, 2]) / s
        qx = 0.25 * s
        qy = (rot_mtx[0, 1] + rot_mtx[1, 0]) / s
        qz = (rot_mtx[0, 2] + rot_mtx[2, 0]) / s
    elif rot_mtx[1, 1] > rot_mtx[2, 2]:
        s = math.sqrt(1.0 + rot_mtx[1, 1] - rot_mtx[0, 0] - rot_mtx[2, 2]) * 2.0
        qw = (rot_mtx[0, 2] - rot_mtx[2, 0]) / s
        qx = (rot_mtx[0, 1] + rot_mtx[1, 0]) / s
        qy = 0.25 * s
        qz = (rot_mtx[1, 2] + rot_mtx[2, 1]) / s
    else:
        s = math.sqrt(1.0 + rot_mtx[2, 2] - rot_mtx[0, 0] - rot_mtx[1, 1]) * 2.0
        qw = (rot_mtx[1, 0] - rot_mtx[0, 1]) / s
        qx = (rot_mtx[0, 2] + rot_mtx[2, 0]) / s
        qy = (rot_mtx[1, 2] + rot_mtx[2, 1]) / s
        qz = 0.25 * s

    q = np.array([qx, qy, qz, qw], dtype=float)
    q /= np.linalg.norm(q)
    return q


def rotation_matrix_to_rpy(rot_mtx: np.ndarray) -> tuple[float, float, float]:
    sy = math.sqrt(rot_mtx[0, 0] ** 2 + rot_mtx[1, 0] ** 2)
    singular = sy < 1e-6

    if not singular:
        roll = math.atan2(rot_mtx[2, 1], rot_mtx[2, 2])
        pitch = math.atan2(-rot_mtx[2, 0], sy)
        yaw = math.atan2(rot_mtx[1, 0], rot_mtx[0, 0])
    else:
        roll = math.atan2(-rot_mtx[1, 2], rot_mtx[1, 1])
        pitch = math.atan2(-rot_mtx[2, 0], sy)
        yaw = 0.0

    return roll, pitch, yaw


class BoardPoseNode(Node):
    def __init__(self) -> None:
        super().__init__("board_pose_node")

        pkg_share = Path(__file__).resolve().parents[1]

        self.declare_parameter("image_topic", "/robot_09/oakd/rgb/image_raw/compressed")
        self.declare_parameter("camera_frame", "oak_camera_frame")
        self.declare_parameter("board_frame", "board_frame")
        self.declare_parameter("calibration_file", str(pkg_share / "config" / "camera_calib_oak.npz"))
        self.declare_parameter("board_config_file", str(pkg_share / "config" / "board_config.json"))
        self.declare_parameter("log_pose", True)
        self.declare_parameter("log_every_n", 10)
        self.declare_parameter("log_rpy_degrees", False)

        self.image_topic = self.get_parameter("image_topic").get_parameter_value().string_value
        self.camera_frame = self.get_parameter("camera_frame").get_parameter_value().string_value
        self.board_frame = self.get_parameter("board_frame").get_parameter_value().string_value
        self.calibration_file = self.get_parameter("calibration_file").get_parameter_value().string_value
        self.board_config_file = self.get_parameter("board_config_file").get_parameter_value().string_value
        self.log_pose = self.get_parameter("log_pose").get_parameter_value().bool_value
        self.log_every_n = self.get_parameter("log_every_n").get_parameter_value().integer_value
        self.log_rpy_degrees = self.get_parameter("log_rpy_degrees").get_parameter_value().bool_value

        calib = np.load(self.calibration_file)
        self.camera_matrix = calib["camera_matrix"]
        self.dist_coeffs = calib["dist_coeffs"]

        with open(self.board_config_file, "r", encoding="utf-8") as f:
            self.board_cfg = json.load(f)

        dict_name = self.board_cfg["dictionary"]
        self.marker_size_m = float(self.board_cfg["marker_size_m"])
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(DICT_MAP[dict_name])
        self.detector_params = cv2.aruco.DetectorParameters()

        self.board_object_points = self._build_board_object_points()

        self.pose_pub = self.create_publisher(PoseStamped, "/robot_09/board_pose", 10)
        self.rpy_pub = self.create_publisher(Vector3Stamped, "/robot_09/board_rpy", 10)
        self.visible_pub = self.create_publisher(Bool, "/robot_09/board_visible", 10)
        self.ids_pub = self.create_publisher(Int32MultiArray, "/robot_09/board_used_ids", 10)
        self.ids_pub = self.create_publisher(Int32MultiArray, "/board_used_ids", 10)
        self.tf_broadcaster = TransformBroadcaster(self)

        self.sub = self.create_subscription(
            CompressedImage,
            self.image_topic,
            self.image_callback,
            10,
        )

        self.frame_count = 0
        self.get_logger().info(f"Subscribed to {self.image_topic}")
        self.get_logger().info(f"Calibration file: {self.calibration_file}")
        self.get_logger().info(f"Board config file: {self.board_config_file}")


    def _build_board_object_points(self) -> dict[int, np.ndarray]:
        result = {}
        size = self.marker_size_m

        for marker_id_str, marker_info in self.board_cfg["markers"].items():
            marker_id = int(marker_id_str)
            x_tl, y_tl = marker_info["top_left_xy_m"]
            rotation_deg = float(marker_info.get("rotation_deg", 0.0))

            corners = np.array(
                [
                    [x_tl, y_tl, 0.0],
                    [x_tl + size, y_tl, 0.0],
                    [x_tl + size, y_tl - size, 0.0],
                    [x_tl, y_tl - size, 0.0],
                ],
                dtype=np.float32,
            )

            if abs(rotation_deg) > 1e-6:
                center = np.array([x_tl + size / 2.0, y_tl - size / 2.0, 0.0], dtype=np.float32)
                theta = math.radians(rotation_deg)
                rot = np.array(
                    [
                        [math.cos(theta), -math.sin(theta), 0.0],
                        [math.sin(theta), math.cos(theta), 0.0],
                        [0.0, 0.0, 1.0],
                    ],
                    dtype=np.float32,
                )
                corners = ((corners - center) @ rot.T) + center

            result[marker_id] = corners

        return result

    def image_callback(self, msg: CompressedImage) -> None:
        np_arr = np.frombuffer(msg.data, dtype=np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None:
            self.get_logger().warning("Failed to decode compressed image")
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = cv2.aruco.detectMarkers(
            gray,
            self.aruco_dict,
            parameters=self.detector_params,
        )

        visible_msg = Bool()
        ids_msg = Int32MultiArray()

        if ids is None or len(ids) == 0:
            visible_msg.data = False
            self.visible_pub.publish(visible_msg)
            self.ids_pub.publish(ids_msg)
            return

        object_points = []
        image_points = []
        used_ids = []

        ids_flat = ids.flatten().tolist()
        for marker_corners, marker_id in zip(corners, ids_flat):
            if marker_id not in self.board_object_points:
                continue

            img_pts = marker_corners.reshape(4, 2).astype(np.float32)
            obj_pts = self.board_object_points[marker_id].astype(np.float32)

            image_points.append(img_pts)
            object_points.append(obj_pts)
            used_ids.append(marker_id)

        if len(used_ids) == 0:
            visible_msg.data = False
            self.visible_pub.publish(visible_msg)
            self.ids_pub.publish(ids_msg)
            return

        object_points = np.vstack(object_points)
        image_points = np.vstack(image_points)

        success, rvec, tvec = cv2.solvePnP(
            object_points,
            image_points,
            self.camera_matrix,
            self.dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        if not success:
            visible_msg.data = False
            self.visible_pub.publish(visible_msg)
            self.ids_pub.publish(ids_msg)
            return

        rot_mtx, _ = cv2.Rodrigues(rvec)
        roll, pitch, yaw = rotation_matrix_to_rpy(rot_mtx)
        quat = rvec_to_quaternion(rvec)

        header = msg.header
        if not header.frame_id:
            header.frame_id = self.camera_frame

        pose_msg = PoseStamped()
        pose_msg.header = header
        pose_msg.header.frame_id = self.camera_frame
        pose_msg.pose.position.x = float(tvec[0][0])
        pose_msg.pose.position.y = float(tvec[1][0])
        pose_msg.pose.position.z = float(tvec[2][0])
        pose_msg.pose.orientation.x = float(quat[0])
        pose_msg.pose.orientation.y = float(quat[1])
        pose_msg.pose.orientation.z = float(quat[2])
        pose_msg.pose.orientation.w = float(quat[3])
        self.pose_pub.publish(pose_msg)

        rpy_msg = Vector3Stamped()
        rpy_msg.header = pose_msg.header
        rpy_msg.vector.x = float(roll)
        rpy_msg.vector.y = float(pitch)
        rpy_msg.vector.z = float(yaw)
        self.rpy_pub.publish(rpy_msg)

        visible_msg.data = True
        self.visible_pub.publish(visible_msg)

        ids_msg.data = used_ids
        self.ids_pub.publish(ids_msg)

        tf_msg = TransformStamped()
        tf_msg.header = pose_msg.header
        tf_msg.child_frame_id = self.board_frame
        tf_msg.transform.translation.x = pose_msg.pose.position.x
        tf_msg.transform.translation.y = pose_msg.pose.position.y
        tf_msg.transform.translation.z = pose_msg.pose.position.z
        tf_msg.transform.rotation = pose_msg.pose.orientation
        self.tf_broadcaster.sendTransform(tf_msg)

        self.frame_count += 1
        if self.log_pose and self.frame_count % max(1, self.log_every_n) == 0:
            rx, ry, rz = roll, pitch, yaw
            if self.log_rpy_degrees:
                rx, ry, rz = map(math.degrees, (roll, pitch, yaw))
            self.get_logger().info(
                f"visible=True ids={used_ids} "
                f"x={tvec[0][0]:.4f} y={tvec[1][0]:.4f} z={tvec[2][0]:.4f} "
                f"rx={rx:.4f} ry={ry:.4f} rz={rz:.4f}"
            )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = BoardPoseNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
