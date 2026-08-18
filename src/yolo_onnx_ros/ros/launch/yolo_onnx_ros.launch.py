from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import LifecycleNode, Node
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
        default_value="/camera/image_raw",
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

    use_sim_time_arg = DeclareLaunchArgument(
        "use_sim_time",
        default_value="true",
        description="Use simulation (Gazebo) clock if true",
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
                "input_image_topic": LaunchConfiguration("image_topic"),
            }
        ],
        output="screen",
    )

    lifecycle_manager = Node(
        package="nav2_lifecycle_manager",
        executable="lifecycle_manager",
        name="lifecycle_manager_yolo",
        output="screen",
        parameters=[{
            'use_sim_time': LaunchConfiguration('use_sim_time'),
            'autostart': True,
            'node_names': ['inference_node'],
            'bond_timeout': 0.0,
        }],
    )

    return LaunchDescription(
        [
            use_sim_time_arg,
            model_path_arg,
            device_type_arg,
            image_topic_arg,
            output_mask_topic_arg,
            output_detections_topic_arg,
            segmentor_node,
            lifecycle_manager,
        ]
    )
