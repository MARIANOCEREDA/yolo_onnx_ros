import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import LifecycleNode
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    model_path_arg = DeclareLaunchArgument(
        "model_path",
        default_value=PathJoinSubstitution(
            [FindPackageShare("yolo_onnx_ros"), "..", "..", "..", "..", "models", "yolov8n-seg.onnx"]
        ),
        description="Absolute path to the ONNX model file.",
    )

    device_type_arg = DeclareLaunchArgument(
        "device_type",
        default_value="CPU",
        description="Inference device: CPU or GPU.",
    )

    image_topic_arg = DeclareLaunchArgument(
        "image_topic",
        default_value="/image_rect",
        description="Input rectified image topic.",
    )

    output_mask_topic_arg = DeclareLaunchArgument(
        "output_mask_topic",
        default_value="/segmentation/mask",
        description="Topic on which the segmentation mask image is published.",
    )

    output_detections_topic_arg = DeclareLaunchArgument(
        "output_detections_topic",
        default_value="/segmentation/detections",
        description="Topic on which Detection2DArray messages are published.",
    )
    
    segmentor_node = LifecycleNode(
        package="yolo_onnx_ros",
        executable="yolo_onnx_ros_segmentor_node",
        name="inference_node",
        namespace="",
        parameters=[
            {
                "model_path": LaunchConfiguration("model_path"),
                "device_type": LaunchConfiguration("device_type"),
                "output_mask_topic": LaunchConfiguration("output_mask_topic"),
                "output_detections_topic": LaunchConfiguration("output_detections_topic"),
            }
        ],
        remappings=[
            ("/image_rect", LaunchConfiguration("image_topic")),
        ],
        output="screen",
    )

    return LaunchDescription(
        [
            model_path_arg,
            device_type_arg,
            image_topic_arg,
            output_mask_topic_arg,
            output_detections_topic_arg,
            segmentor_node,
        ]
    )
