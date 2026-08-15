ROS_DISTRO   ?= jazzy
ROS_SETUP     = /opt/ros/$(ROS_DISTRO)/setup.bash

WS_ROOT      := $(shell pwd)
YOLO_ONNX_ONNX_CPP_DIR := $(WS_ROOT)/src/yolo_onnx_ros/cpp
YOLO_ONNX_CPP_BUILD_DIR := $(YOLO_ONNX_ONNX_CPP_DIR)/build

JOBS         ?= $(shell nproc)

.PHONY: help build-cpp-library build-ros2 build-all cleanONNX_-cpp clean-ros2 clean

help:
	@echo ""
	@echo "Available targets:"
	@echo "  build-cpp-library  Build the yolo_onnx_ros C++ library only (CMake)"
	@echo "  build-ros2         Build all ROS2 packages (colcon)"
	@echo "  build-all          Build C++ library then ROS2 packages"
	@echo "  clean-cpp          Remove the C++ library build directory"
	@echo "  clean-ros2         Remove colcon build/install/log directories"
	@echo "  clean              Remove all build artifacts"
	@echo "  install-rosdeps    Install ROS2 dONNX_ependencies"
	@echo ""
	@echo "Options (override with make <target> OPTION=value):"
	@echo "  ROS_DISTRO=<distro>   ROS2 distribution name  (default: jazzy)"
	@echo ""

build-cpp-library:
	@echo "==> Building yolo_onnx_ros C++ library..."
	@mkdir -p $(YOLO_ONNX_CPP_BUILD_DIR)
	@cd $(YOLO_ONNX_CPP_BUILD_DIR) && cmake .. -DCMAKE_BUILD_TYPE=Release && make -j$(JOBS)
	@echo "==> C++ library build complete."

clean-cpp:
	@echo "==> Cleaning C++ library build..."
	@rm -rf $(YOLO_ONNX_CPP_BUILD_DIR)
	@echo "==> Done."

build-ros2:
	@echo "==> Building ROS2 packages from workspace: $(WS_ROOT)"
	@bash -c "source $(ROS_SETUP) && \
	          cd $(WS_ROOT) && \
	          colcon build \
	            --symlink-install \
	            --parallel-workers $(JOBS) \
	            --packages-select yolo_onnx_ros image_file_publisher \
	            --cmake-args -DYOLO_ONNX_CPP_BUILD_DIR=$(YOLO_ONNX_CPP_BUILD_DIR) \
	            --event-handlers console_cohesion+"
	@echo "==> ROS2 build complete."

clean-ros2:
	@echo "==> Cleaning ROS2 build artifacts in $(WS_ROOT)..."
	@rm -rf $(WS_ROOT)/build $(WS_ROOT)/install $(WS_ROOT)/log
	@echo "==> Done."

build-all:
	@echo "==> Building all components..."
	@$(MAKE) build-cpp-library
	@$(MAKE) build-ros2
	@echo "==> All components built successfully."

install-rosdeps:
	@echo "==> Installing ROS2 dependencies..."
	@bash -c "source $(ROS_SETUP) && \
	          cd $(WS_ROOT) && \
			  sudo apt update && \
	          rosdep update && \
	          rosdep install --from-paths src --ignore-src -r -y"
	@echo "==> ROS2 dependencies installed."

clean:
	@$(MAKE) clean-cpp
	@$(MAKE) clean-ros2