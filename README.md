# yolo_onnx_ros

A library that provides inference for YOLO models (versions 8, 9, and 11) using ONNX Runtime.

## Latency (ROS 2 node)

Measured with `display_latency: true`. The node logs per-stage averages over ~1 s windows
(`steady_clock`, host side). Numbers are ranges over several windows of the same bag.

**Setup:** `yolov11n-seg.onnx`, `device_type: GPU` (RTX 3060 Laptop), i7-11800H, Release build,
960x544 `rgba8` input (undistorted, same size as the raw image) at ~13 Hz, composable container with intra-process comms. Few or no detections in the
measured scene, so `masks` is a lower bound.

| Stage | What it covers | Before | Current |
|---|---|---|---|
| `pre` | Letterbox + alpha drop + blob + tensor wrap | 4.67 ms | 1.9–5.7 ms (typ. 2–3) |
| `run` | `session.Run` (includes H2D/D2H copies) | 8.00 ms | 7.6–10.1 ms |
| `decode` | Output tensor → boxes → confidence filter → NMS | 2.59 ms | 2.3–3.3 ms |
| `masks` | Proto masks → per-instance masks | 0.31 ms | ~0.0–0.2 ms |
| `copy` | Final boxes copied to the caller | — | ~0.00 ms |
| `node-mask` | Binary mask build + mask message + publish | — | 0.2–0.4 ms |
| `total` | Segment start → detections and mask published | — | 12–19 ms (typ. ~14) |
| `wait` | `ImageCallback` → worker dequeue | — | < 0.1 ms, 0 dropped frames |

"Before" is the first per-stage measurement. "Current" includes:

- Correct RGB input to the model. The R/B swap now happens inside the blob repack, replacing a
  separate `cvtColor`.
- Zero-copy `cv::Mat` view over the input message, with `rgb8`, `bgr8`, `rgba8` and `bgra8`
  supported.
- Letterbox written straight into a reused buffer, with only the padding filled. No allocation per
  frame.

Run-to-run spread in `pre` and `run` is mostly system load (other components in the same container).
Compare averages over several windows, not single lines.
