"""

Computer vision module for object detection and image processing.

This module handles cube detection and other CV tasks at the Application Layer.

"""

from typing import List, Optional, Tuple
from dataclasses import dataclass
import time
import struct
from threading import Event

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import logger
from . import protocol_encoder

# Try to import CubeFusionTracker if available
try:
    from .cubeDetector.cubeDetector import CubeFusionTracker, CozmoCubeDetector
    CUBE_DETECTOR_AVAILABLE = True
except ImportError:
    CUBE_DETECTOR_AVAILABLE = False
    logger.warning("CubeFusionTracker not available - advanced cube tracking disabled")


__all__ = [
    "CubeDetection",
    "VisionProcessor",
    "CameraCalibration",
    "CameraCalibrationRetriever",
]


@dataclass
class CubeDetection:
    """Represents a detected cube in the camera image."""

    # Bounding box (x, y, width, height) in pixels
    x: int
    y: int
    width: int
    height: int

    # Confidence score (0.0 to 1.0)
    confidence: float

    # Estimated distance from robot (if available)
    distance: Optional[float] = None

    # Cube ID (if tracking is enabled)
    cube_id: Optional[int] = None

    # Timestamp of detection
    timestamp: float = 0.0


@dataclass
class CameraCalibration:
    """
    Camera intrinsic calibration parameters for 3D localization.

    These parameters define the camera's internal geometry and are required
    for accurate 3D reconstruction from 2D images.
    """

    fx: float  # Focal length x (pixels)
    fy: float  # Focal length y (pixels)
    cx: float  # Principal point x (pixels)
    cy: float  # Principal point y (pixels)
    k1: float = 0.0  # Radial distortion
    k2: float = 0.0
    k3: float = 0.0
    p1: float = 0.0  # Tangential distortion
    p2: float = 0.0
    image_width: int = 320
    image_height: int = 240
    fov_x: Optional[float] = None
    fov_y: Optional[float] = None

    def __post_init__(self):
        """Compute FOV if not provided."""
        if self.fov_x is None:
            self.fov_x = 2.0 * np.arctan(self.image_width / (2.0 * self.fx))
        if self.fov_y is None:
            self.fov_y = 2.0 * np.arctan(self.image_height / (2.0 * self.fy))

    @property
    def camera_matrix(self) -> np.ndarray:
        """Get the camera intrinsic matrix K."""
        return np.array([
            [self.fx, 0.0, self.cx],
            [0.0, self.fy, self.cy],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)

    @property
    def distortion_coefficients(self) -> np.ndarray:
        """Get distortion coefficients [k1, k2, p1, p2, k3]."""
        return np.array([self.k1, self.k2, self.p1, self.p2, self.k3], dtype=np.float64)

    @classmethod
    def default_calibration(cls) -> 'CameraCalibration':
        """Get default/approximate calibration for Cozmo."""
        return cls(fx=340.0, fy=340.0, cx=160.0, cy=120.0,
                   image_width=320, image_height=240)


class CameraCalibrationRetriever:
    """Retrieves camera calibration data from Cozmo's NVRAM."""

    def __init__(self, client):
        self.client = client
        self._calib_data = []
        self._event = Event()
        self._success = False

    def get_calibration(self, timeout: float = 10.0) -> Optional[CameraCalibration]:
        """Retrieve camera calibration from robot."""
        logger.info("Requesting camera calibration...")
        self._calib_data = []
        self._event.clear()
        self._success = False

        self.client.add_handler(protocol_encoder.NvStorageOpResult,
                                 self._on_nv_storage_op_result)

        try:
            pkt = protocol_encoder.NvStorageOp(
                tag=protocol_encoder.NvEntryTag.NVEntry_CameraCalib,
                length=1,
                op=protocol_encoder.NvOperation.NVOP_READ)
            self.client.conn.send(pkt)

            if self._event.wait(timeout=timeout):
                if self._success and self._calib_data:
                    calib = self._parse_calibration_data(self._calib_data)
                    if calib:
                        logger.info(f"Calibration retrieved: {calib}")
                        return calib
            else:
                logger.error("Timeout waiting for calibration")
        except Exception as e:
            logger.error(f"Error retrieving calibration: {e}")

        logger.warning("Using default calibration")
        return CameraCalibration.default_calibration()

    def _on_nv_storage_op_result(self, cli, pkt):
        """Handler for NVRAM results."""
        if pkt.tag == protocol_encoder.NvEntryTag.NVEntry_CameraCalib:
            if pkt.data:
                self._calib_data.extend(pkt.data)
            if pkt.result != protocol_encoder.NvResult.NV_MORE:
                self._success = (pkt.result == protocol_encoder.NvResult.NV_OKAY)
                self._event.set()

    def _parse_calibration_data(self, data: list) -> Optional[CameraCalibration]:
        """Parse raw calibration data."""
        if not data:
            return None
        data_bytes = bytes(data)

        # Try float32 format (9 values = 36 bytes)
        if len(data_bytes) >= 36:
            try:
                vals = struct.unpack('<9f', data_bytes[:36])
                fx, fy, cx, cy, k1, k2, p1, p2, k3 = vals
                if 100 < fx < 1000 and 100 < fy < 1000:
                    return CameraCalibration(fx=fx, fy=fy, cx=cx, cy=cy,
                                            k1=k1, k2=k2, k3=k3, p1=p1, p2=p2)
            except:
                pass

        # Try float64 format (9 values = 72 bytes)
        if len(data_bytes) >= 72:
            try:
                vals = struct.unpack('<9d', data_bytes[:72])
                fx, fy, cx, cy, k1, k2, p1, p2, k3 = vals
                if 100 < fx < 1000 and 100 < fy < 1000:
                    return CameraCalibration(fx=fx, fy=fy, cx=cx, cy=cy,
                                            k1=k1, k2=k2, k3=k3, p1=p1, p2=p2)
            except:
                pass

        logger.warning(f"Could not parse {len(data_bytes)} bytes of calibration data")
        return None


class VisionProcessor:
    """
    Computer vision processor for detecting cubes and other objects.

    This class should be used at the Application Layer (Brain) to process
    camera images asynchronously.
    """

    def __init__(self, enable_tracking: bool = True,
                 camera_calibration: Optional[CameraCalibration] = None,
                 use_advanced_detector: bool = False):
        """
        Initialize the vision processor.

        Args:
            enable_tracking: Whether to track detected cubes across frames
            camera_calibration: Camera calibration for 3D pose estimation
            use_advanced_detector: Use CozmoCubeDetector with CubeFusionTracker for 3D tracking
        """
        self.enable_tracking = enable_tracking
        self.next_cube_id = 1
        self.tracked_cubes = {}  # Dict[int, CubeDetection]
        self.camera_calibration = camera_calibration or CameraCalibration.default_calibration()

        # Advanced 3D detector setup
        self.use_advanced_detector = use_advanced_detector and CUBE_DETECTOR_AVAILABLE
        self.cube_detector = None
        self.cube_tracker = None
        self.fused_cubes = {}  # 3D fused cube states

        if self.use_advanced_detector:
            if not CUBE_DETECTOR_AVAILABLE:
                logger.warning("Advanced detector requested but CubeFusionTracker not available")
                self.use_advanced_detector = False
            else:
                self._init_advanced_detector()

        # Detection parameters (can be tuned)
        self.min_cube_size = 20  # Minimum cube size in pixels
        self.max_cube_size = 200  # Maximum cube size in pixels
        self.confidence_threshold = 0.5

        logger.info("Vision processor initialized")

    def _init_advanced_detector(self):
        """Initialize the advanced CozmoCubeDetector with marker templates."""
        import cv2
        import glob
        import os

        # Load marker templates
        target_markers = []
        # Try to find marker templates in cubeDetector directory
        marker_dir = os.path.join(os.path.dirname(__file__), 'cubeDetector', 'cube_faces')
        if not os.path.exists(marker_dir):
            logger.warning(f"Marker directory not found: {marker_dir}")
            self.use_advanced_detector = False
            return

        template_files = sorted(glob.glob(os.path.join(marker_dir, '*.jpg')))
        if not template_files:
            logger.warning(f"No marker templates found in {marker_dir}")
            self.use_advanced_detector = False
            return

        logger.info(f"Loading {len(template_files)} marker templates:")
        for idx, img_path in enumerate(template_files):
            filename = os.path.basename(img_path)
            logger.info(f"  Marker {idx:2d}: {filename}")
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            # Cut offset from each side
            h, w = img.shape
            offset = 18
            img = img[offset:h - offset, offset:w - offset]
            # Resize to 26x26
            img = cv2.resize(img, (26, 26), interpolation=cv2.INTER_AREA)
            # Add 3px white border to make 32x32
            img_with_border = cv2.copyMakeBorder(img, 3, 3, 3, 3, cv2.BORDER_CONSTANT, value=255)
            target_markers.append(img_with_border)

        if not target_markers:
            logger.warning("No valid marker templates loaded")
            self.use_advanced_detector = False
            return

        # Create detector
        self.cube_detector = CozmoCubeDetector(
            self.camera_calibration.camera_matrix,
            self.camera_calibration.distortion_coefficients,
            target_markers,
            downscale_factor=1.0,
            use_clahe=True,
            max_detections=3  # Cozmo has 3 cubes
        )

        # Create fusion tracker
        self.cube_tracker = CubeFusionTracker(
            cube_size_mm=44.0,
            smoothing_alpha=0.3,
            max_age_frames=10
        )

        logger.info("Advanced cube detector initialized")

    def get_fused_cube_states(self):
        """
        Get the fused 3D cube states from the fusion tracker.

        Returns:
            Dict of {cube_id: cube_state} where cube_state contains:
            - position: 3D position (x, y, z) in mm
            - rotation_matrix: 3x3 rotation matrix
            - confidence: detection confidence
            - num_faces: number of faces detected
            - rotation_source: which face determined the rotation
        """
        return self.fused_cubes.copy() if self.use_advanced_detector else {}

    def detect_cubes(self, image: Image.Image) -> List[CubeDetection]:
        """
        Detect cubes in the given camera image.

        Args:
            image: PIL Image from the robot's camera

        Returns:
            List of detected cubes
        """
        timestamp = time.time()

        # Convert PIL Image to numpy array for processing
        img_array = np.array(image)

        # Use advanced detector if available
        if self.use_advanced_detector and self.cube_detector is not None:
            detections = self._detect_cubes_advanced(img_array)
        else:
            detections = self._detect_cubes_impl(img_array)

        # Add timestamps
        for det in detections:
            det.timestamp = timestamp

        # Update tracking if enabled (only for basic detection)
        if self.enable_tracking and not self.use_advanced_detector:
            detections = self._update_tracking(detections)

        return detections

    def _detect_cubes_advanced(self, img_array: np.ndarray) -> List[CubeDetection]:
        """
        Use advanced CozmoCubeDetector with 3D tracking.

        Args:
            img_array: Image as numpy array (RGB from PIL)

        Returns:
            List of detected cubes with 3D information
        """
        import cv2

        # Convert RGB to BGR for OpenCV
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        # Detect marker faces
        results = self.cube_detector.detect(img_bgr)

        # Fuse detections into coherent cube poses
        self.fused_cubes = self.cube_tracker.update(results)

        # Convert to CubeDetection format
        detections = []
        for cube_id, cube_state in self.fused_cubes.items():
            # Calculate approximate bounding box from 3D position
            # This is a simplified projection - could be improved
            pos = cube_state['position']
            distance = np.linalg.norm(pos)  # Distance in mm

            # Find detections for this cube to get bounding box
            cube_results = [r for r in results if r['cube_id'] == cube_id]
            if cube_results:
                # Use the bounding box from detections
                all_corners = np.vstack([r['corners'] for r in cube_results])
                x_min = int(np.min(all_corners[:, 0]))
                y_min = int(np.min(all_corners[:, 1]))
                x_max = int(np.max(all_corners[:, 0]))
                y_max = int(np.max(all_corners[:, 1]))

                width = x_max - x_min
                height = y_max - y_min

                det = CubeDetection(
                    x=x_min,
                    y=y_min,
                    width=width,
                    height=height,
                    confidence=cube_state['confidence'],
                    distance=distance / 1000.0,  # Convert mm to meters
                    cube_id=cube_id + 1  # 1-indexed for user display
                )
                detections.append(det)

        return detections

    def _detect_cubes_impl(self, img_array: np.ndarray) -> List[CubeDetection]:
        """
        Internal implementation of cube detection.

        This is a placeholder that should be replaced with actual detection logic.

        Args:
            img_array: Image as numpy array

        Returns:
            List of detected cubes (without IDs or tracking)
        """
        # Placeholder implementation
        # In a real implementation, you would:
        # 1. Pre-process the image (resize, normalize, etc.)
        # 2. Apply detection algorithm
        # 3. Post-process results (NMS, filtering, etc.)

        detections = []

        # Example: Simple color-based detection for demonstration
        # Cozmo cubes typically have red, green, and blue markers
        detections.extend(self._detect_by_color(img_array, "red"))
        detections.extend(self._detect_by_color(img_array, "green"))
        detections.extend(self._detect_by_color(img_array, "blue"))

        return detections

    def _detect_by_color(self, img_array: np.ndarray, color: str) -> List[CubeDetection]:
        """
        Detect cube-like regions by color.

        This is a simplified example. Real implementation would need more
        sophisticated algorithms.

        Args:
            img_array: Image as numpy array
            color: Color to detect ("red", "green", "blue")

        Returns:
            List of potential cube detections
        """
        # Placeholder - would implement color thresholding and contour detection
        # For example, using OpenCV:
        # - Convert to HSV color space
        # - Apply color threshold
        # - Find contours
        # - Filter by size and shape
        # - Return bounding boxes

        return []

    def _update_tracking(self, detections: List[CubeDetection]) -> List[CubeDetection]:
        """
        Update cube tracking across frames.

        Args:
            detections: List of current frame detections

        Returns:
            List of detections with cube IDs assigned
        """
        # Simple tracking based on position proximity
        # More sophisticated tracking could use Kalman filters, Hungarian algorithm, etc.

        tracked = []

        for det in detections:
            # Try to match with existing tracked cubes
            matched_id = None
            min_distance = float('inf')

            for cube_id, prev_det in self.tracked_cubes.items():
                # Calculate distance between current and previous detection
                dist = np.sqrt((det.x - prev_det.x)**2 + (det.y - prev_det.y)**2)

                # If close enough, consider it the same cube
                if dist < 50 and dist < min_distance:  # 50 pixels threshold
                    matched_id = cube_id
                    min_distance = dist

            if matched_id is not None:
                det.cube_id = matched_id
            else:
                # New cube detected
                det.cube_id = self.next_cube_id
                self.next_cube_id += 1

            self.tracked_cubes[det.cube_id] = det
            tracked.append(det)

        # Remove old tracked cubes that weren't detected
        current_ids = {det.cube_id for det in tracked}
        self.tracked_cubes = {k: v for k, v in self.tracked_cubes.items() if k in current_ids}

        return tracked

    def reset_tracking(self):
        """Reset the cube tracking state."""
        self.tracked_cubes.clear()
        self.next_cube_id = 1
        if self.cube_tracker:
            self.cube_tracker.cubes.clear()
        logger.info("Cube tracking reset")

    # Annotation settings
    annotation_colors = [
        (255, 0, 0),    # Red
        (0, 255, 0),    # Green
        (0, 0, 255),    # Blue
        (255, 255, 0),  # Yellow
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Cyan
    ]
    box_thickness = 2
    show_cube_id = True
    show_confidence = True
    show_distance = True

    def draw_detections(self, image: Image.Image, detections: List[CubeDetection]) -> Image.Image:
        """
        Draw detection bounding boxes and labels on an image.

        Args:
            image: PIL Image to annotate
            detections: List of cube detections to draw

        Returns:
            Annotated PIL Image (a copy of the original)
        """
        # Create a copy to avoid modifying the original
        annotated_image = image.copy()
        draw = ImageDraw.Draw(annotated_image)

        # Try to load a font, fall back to default if not available
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 12)
        except:
            try:
                font = ImageFont.truetype("arial.ttf", 12)
            except:
                font = ImageFont.load_default()

        for i, det in enumerate(detections):
            # Choose color based on cube ID or cycle through colors
            if det.cube_id is not None:
                color = self.annotation_colors[det.cube_id % len(self.annotation_colors)]
            else:
                color = self.annotation_colors[i % len(self.annotation_colors)]

            # Draw bounding box
            x1, y1 = det.x, det.y
            x2, y2 = det.x + det.width, det.y + det.height

            # Draw rectangle with specified thickness
            for offset in range(self.box_thickness):
                draw.rectangle(
                    [(x1 - offset, y1 - offset), (x2 + offset, y2 + offset)],
                    outline=color,
                    width=1
                )

            # Build label text
            label_parts = []
            if self.show_cube_id and det.cube_id is not None:
                label_parts.append(f"ID:{det.cube_id}")
            if self.show_confidence:
                label_parts.append(f"{det.confidence:.2f}")
            if self.show_distance and det.distance is not None:
                label_parts.append(f"{det.distance:.1f}m")

            if label_parts:
                label = " | ".join(label_parts)

                # Draw label background
                try:
                    # For newer Pillow versions
                    bbox = draw.textbbox((x1, y1 - 15), label, font=font)
                    text_width = bbox[2] - bbox[0]
                    text_height = bbox[3] - bbox[1]
                except AttributeError:
                    # For older Pillow versions
                    text_width, text_height = draw.textsize(label, font=font)
                    bbox = (x1, y1 - 15, x1 + text_width, y1 - 15 + text_height)

                # Draw background rectangle for text
                draw.rectangle(
                    [(bbox[0] - 2, bbox[1] - 2), (bbox[2] + 2, bbox[3] + 2)],
                    fill=(0, 0, 0, 180)
                )

                # Draw text
                draw.text((x1, y1 - 15), label, fill=color, font=font)

        return annotated_image

    def detect_and_annotate(self, image: Image.Image) -> Tuple[List[CubeDetection], Image.Image]:
        """
        Detect cubes and return both detections and an annotated image.

        This is a convenience method that combines detect_cubes() and draw_detections().

        Args:
            image: PIL Image from the robot's camera

        Returns:
            Tuple of (detections list, annotated image)
        """
        detections = self.detect_cubes(image)
        annotated_image = self.draw_detections(image, detections)
        return detections, annotated_image


class AnnotatedVisionProcessor(VisionProcessor):
    """
    Vision processor that automatically generates annotated frames.

    This class extends VisionProcessor and automatically dispatches annotated
    camera images with detection boxes drawn on them. Use this when you want
    to receive both raw camera frames and annotated versions with visualizations.

    Example:
        vision = AnnotatedVisionProcessor(client, enable_tracking=True)

        # This will automatically start dispatching EvtAnnotatedCameraImage events
        def on_annotated_image(cli, annotated_image, detections):
            annotated_image.save("annotated_frame.png")

        cli.add_handler(pycozmo.event.EvtAnnotatedCameraImage, on_annotated_image)
    """

    def __init__(self, client, enable_tracking: bool = True,
                 process_every_n_frames: int = 1):
        """
        Initialize the annotated vision processor.

        Args:
            client: PyCozmo Client instance to attach handlers to
            enable_tracking: Whether to track detected cubes across frames
            process_every_n_frames: Process every Nth frame (1 = all frames,
                                   2 = every other frame, etc.)
        """
        super().__init__(enable_tracking=enable_tracking)

        self.client = client
        self.process_every_n_frames = process_every_n_frames
        self._frame_counter = 0

        # Import here to avoid circular dependency
        from . import event

        # Register camera image handler
        self.client.add_handler(event.EvtNewRawCameraImage, self._on_camera_image)

        logger.info(f"Annotated vision processor initialized (processing every "
                   f"{process_every_n_frames} frame{'s' if process_every_n_frames > 1 else ''})")

    def _on_camera_image(self, cli, image: Image.Image):
        """
        Internal handler for camera images.

        This processes the image, detects cubes, generates annotations,
        and dispatches events.
        """
        # Frame skipping for performance
        self._frame_counter += 1
        if self._frame_counter % self.process_every_n_frames != 0:
            return

        # Import here to avoid circular dependency
        from . import event

        # Detect cubes and annotate
        detections, annotated_image = self.detect_and_annotate(image)

        # Dispatch annotated image event
        self.client.dispatch(event.EvtAnnotatedCameraImage, cli, annotated_image, detections)

        # Also handle cube detection events
        self._dispatch_cube_events(cli, detections)

    def _dispatch_cube_events(self, cli, detections: List[CubeDetection]):
        """
        Dispatch cube detection events (detected, observed, lost).

        Args:
            cli: Client instance
            detections: List of current detections
        """
        # Import here to avoid circular dependency
        from . import event

        # Track which cubes we've seen
        if not hasattr(self, '_last_cube_ids'):
            self._last_cube_ids = set()

        current_ids = {det.cube_id for det in detections if det.cube_id is not None}

        # Detect new cubes
        new_ids = current_ids - self._last_cube_ids
        if new_ids:
            new_detections = [d for d in detections if d.cube_id in new_ids]
            cli.dispatch(event.EvtCubeDetected, cli, new_detections)

        # Dispatch observation event if any cubes visible
        if detections:
            cli.dispatch(event.EvtCubeObserved, cli, detections)

        # Detect lost cubes
        lost_ids = self._last_cube_ids - current_ids
        for cube_id in lost_ids:
            cli.dispatch(event.EvtCubeLost, cli, cube_id)

        # Update state
        self._last_cube_ids = current_ids

    def stop(self):
        """
        Stop the annotated vision processor and remove handlers.
        """
        from . import event

        # Note: There's no built-in way to remove a specific handler in the current
        # implementation, but we can stop processing by setting a flag
        self._stopped = True
        logger.info("Annotated vision processor stopped")
