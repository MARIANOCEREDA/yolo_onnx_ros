FROM ros:jazzy-ros-base-noble AS core_system

# Prevent interactive questions during image build
ENV DEBIAN_FRONTEND=noninteractive

# Remove the docker-auto-clean feature of debian docker images
RUN rm -f /etc/apt/apt.conf.d/docker-clean

RUN userdel ubuntu

# Install essential packages and ROS development tools
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt update \
    && apt install -y --no-install-recommends \
    apt-transport-https \
    bash-completion \
    build-essential \
    ca-certificates \
    software-properties-common \
    ccache \
    curl \
    gdb \
    git \
    gnupg \
    htop \
    iputils-ping \
    libglfw3-dev \
    locate \
    mc \
    nano \
    openssh-client \
    python-is-python3 \
    python3-colcon-argcomplete \
    python3-colcon-common-extensions \
    python3-pip \
    screen \
    sudo \
    tmux \
    vim \
    wget \
    zsh \
    libserial-dev \
    clang-tidy \
    clang-format \
    pre-commit \
    locales \
    libpcl-dev \
    libi2c-dev \
    python3-vcstool \
    nlohmann-json3-dev \
    ros-$ROS_DISTRO-rviz2 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /ros2_ws

COPY . /ros2_ws

RUN mkdir -p /tmp/ws/src \
    && find ./ -name "package.xml" | xargs cp --parents -t /tmp/ws/src \
    && find ./ -name "COLCON_IGNORE" | xargs cp --parents -t /tmp/ws/src \
    || true

FROM core_system AS builder

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt update \
    && apt install --no-install-recommends -y ros-$ROS_DISTRO-ros-base \
    && rm -rf /var/lib/apt/lists/*

ARG UID=1000
ARG GID=1000
ARG USER=dev
ARG GROUP=dev

RUN addgroup --gid $GID $GROUP \
    && adduser --uid $UID --ingroup $GROUP --home /home/$USER --shell /bin/bash --disabled-password --gecos "" $USER \
    && adduser $USER sudo \
    && echo "$USER ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers \
    && echo "source /opt/ros/$ROS_DISTRO/setup.bash" >> /home/$USER/.bashrc \
    && echo "source /usr/share/colcon_argcomplete/hook/colcon-argcomplete.bash" >> /home/$USER/.bashrc \
    && echo "export QT_X11_NO_MITSHM=1" >> /home/$USER/.bashrc

# Install OpenCV 4.6.0 (provided by Ubuntu 24.04 Noble apt)
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    sudo apt-get update \
    && sudo apt-get install -y libopencv-dev

# Install ONNX Runtime 1.22.0 GPU (x64)
ARG ONNXRUNTIME_VERSION=1.22.0
RUN cd /tmp \
    && wget -q https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-linux-x64-gpu-${ONNXRUNTIME_VERSION}.tgz \
    && tar -xzf onnxruntime-linux-x64-gpu-${ONNXRUNTIME_VERSION}.tgz \
    && sudo cp -r onnxruntime-linux-x64-gpu-${ONNXRUNTIME_VERSION}/include/* /usr/local/include/ \
    && sudo cp -r onnxruntime-linux-x64-gpu-${ONNXRUNTIME_VERSION}/lib/* /usr/local/lib/ \
    && sudo ldconfig \
    && rm -rf /tmp/onnxruntime-linux-x64-gpu-${ONNXRUNTIME_VERSION}*

USER $USER

ENV HOME=/home/$USER
ENV WS_DIR=$HOME/ws

RUN mkdir -p $HOME/.cache
RUN mkdir -p $HOME/.config

WORKDIR $WS_DIR

COPY --from=core_system --chown=$USER:$GROUP /tmp/ws/src /home/$USER/ws/src

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    sudo apt update \
    && rosdep update \
    && rosdep install --from-paths src --ignore-src --rosdistro $ROS_DISTRO -y

ENTRYPOINT [ "/bin/bash" ]