#!/usr/bin/env python
"""

Main test script for PyCozmo vision system.

Displays both normal and annotated camera frames side-by-side using OpenCV.
Includes 3D viewer for cube tracking.

Controls:
- Press 'q' to quit
- Press 's' to save current frames
- Press 'c' to toggle cube detection
- Press 'f' to toggle FPS display
- Press '3' to toggle 3D viewer

"""

import time
import cv2
import numpy as np
import sys
import os

import pycozmo
from pycozmo import vision
from pycozmo.camera_calibration import CameraCalibrationRetriever

# Import 3D viewer
try:
    sys.path.append(os.path.join(os.path.dirname(__file__), 'pycozmo', 'cubeDetector'))
    from world_3d_viewer import World3DViewer

    VIEWER_3D_AVAILABLE = True
except ImportError:
    VIEWER_3D_AVAILABLE = False
    print("⚠️  3D Viewer not available")


class VisionDisplay:
    """Handles display of normal and annotated frames."""

    def __init__(self, use_3d_viewer=True):
        self.latest_raw_frame = None
        self.latest_annotated_frame = None
        self.latest_detections = []
        self.latest_debug_images = {}
        self.running = True
        self.show_fps = True
        self.detection_enabled = True
        self.frame_count = 0
        self.start_time = time.time()
        self.fps = 0.0
        self.vision_processor = None

        # 3D viewer setup
        self.use_3d_viewer = use_3d_viewer and VIEWER_3D_AVAILABLE
        self.viewer_3d = None
        self.show_3d = True

        if self.use_3d_viewer:
            try:
                self.viewer_3d = World3DViewer(window_size=(800, 600))
                print("✅ 3D Viewer initialized")
            except Exception as e:
                print(f"⚠️  Failed to initialize 3D viewer: {e}")
                self.use_3d_viewer = False

        # Window names
        self.window_name = "PyCozmo Vision Test - Normal (Left) | Annotated (Right)"

        print("\n" + "=" * 70)
        print("PyCozmo Vision System - Live Display")
        print("=" * 70)
        print("\nControls:")
        print("  'q' - Quit")
        print("  's' - Save current frames")
        print("  'c' - Toggle cube detection")
        print("  'f' - Toggle FPS display")
        if self.use_3d_viewer:
            print("  '3' - Toggle 3D viewer")
        print("=" * 70 + "\n")

    def on_raw_camera_image(self, cli, image):
        """Handler for raw camera images."""
        self.latest_raw_frame = image
        self.frame_count += 1

        # Calculate FPS
        elapsed = time.time() - self.start_time
        if elapsed > 1.0:
            self.fps = self.frame_count / elapsed
            self.frame_count = 0
            self.start_time = time.time()

    def on_annotated_camera_image(self, cli, annotated_image, detections):
        """Handler for annotated camera images."""
        self.latest_annotated_frame = annotated_image
        self.latest_detections = detections

        # Update 3D viewer if enabled
        if self.use_3d_viewer and self.show_3d and self.viewer_3d:
            self.update_3d_viewer()

    def on_debug_images(self, cli, debug_images, detections):
        """Handler for debug images from cube detection pipeline."""
        self.latest_debug_images = debug_images

    def update_3d_viewer(self):
        """Update 3D viewer with current cube states."""
        if not self.vision_processor:
            return

        # Get fused cube states from vision processor
        fused_cubes = self.vision_processor.get_fused_cube_states()

        # Track which cubes are present
        if not hasattr(self, '_prev_cube_ids'):
            self._prev_cube_ids = set()

        current_cube_ids = set(fused_cubes.keys())

        # Only clear if the set of cubes changed (expensive operation)
        if current_cube_ids != self._prev_cube_ids:
            try:
                self.viewer_3d.clear_cubes()
                self._prev_cube_ids = current_cube_ids
            except Exception as e:
                print(f"⚠️  Error clearing 3D viewer: {e}")
                return

        # Update cube poses (cheap operation)
        try:
            for cube_id, cube_state in fused_cubes.items():
                cube_name = f"cube_{cube_id + 1}"
                self.viewer_3d.update_cube(
                    cube_name,
                    cube_state['position'],
                    cube_state['rotation_matrix'],
                    cube_size=44
                )
        except Exception as e:
            print(f"⚠️  Error updating 3D viewer: {e}")

    def pil_to_cv2(self, pil_image):
        """Convert PIL Image to OpenCV format."""
        if pil_image is None:
            return None
        # Convert PIL RGB to OpenCV BGR
        rgb = np.array(pil_image)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        return bgr

    def add_text_overlay(self, frame, text_lines, position=(10, 30)):
        """Add text overlay to frame."""
        x, y = position
        for i, line in enumerate(text_lines):
            y_pos = y + i * 25
            # Draw text background
            (text_width, text_height), _ = cv2.getTextSize(
                line, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            cv2.rectangle(
                frame,
                (x - 5, y_pos - text_height - 5),
                (x + text_width + 5, y_pos + 5),
                (0, 0, 0),
                -1
            )
            # Draw text
            cv2.putText(
                frame,
                line,
                (x, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

    def display_frames(self):
        """Display both raw and annotated frames side by side."""
        # Convert PIL images to OpenCV format
        raw_cv2 = self.pil_to_cv2(self.latest_raw_frame)
        annotated_cv2 = self.pil_to_cv2(self.latest_annotated_frame)

        # If no frames yet, show waiting message
        if raw_cv2 is None:
            waiting_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(
                waiting_frame,
                "Waiting for camera frames...",
                (100, 240),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2
            )
            cv2.imshow(self.window_name, waiting_frame)
            return

        # Use raw frame for both if no annotated frame yet
        if annotated_cv2 is None:
            annotated_cv2 = raw_cv2.copy()

        # Ensure both frames are the same size
        height, width = raw_cv2.shape[:2]
        if annotated_cv2.shape[:2] != (height, width):
            annotated_cv2 = cv2.resize(annotated_cv2, (width, height))

        # Add labels to frames
        raw_labeled = raw_cv2.copy()
        annotated_labeled = annotated_cv2.copy()

        # Add "Normal" label to raw frame
        cv2.rectangle(raw_labeled, (0, 0), (width, 30), (0, 0, 0), -1)
        cv2.putText(
            raw_labeled,
            "Normal Camera Feed",
            (10, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        # Add "Annotated" label to annotated frame
        cv2.rectangle(annotated_labeled, (0, 0), (width, 30), (0, 0, 0), -1)
        cv2.putText(
            annotated_labeled,
            "Annotated (Detections)",
            (10, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

        # Add FPS and detection info
        if self.show_fps:
            info_lines = [
                f"FPS: {self.fps:.1f}",
                f"Cubes: {len(self.latest_detections)}",
                f"Detection: {'ON' if self.detection_enabled else 'OFF'}"
            ]
            self.add_text_overlay(raw_labeled, info_lines, (10, height - 90))

            # Add detection details on annotated frame
            if self.latest_detections:
                det_lines = [f"Detected Cubes: {len(self.latest_detections)}"]
                for i, det in enumerate(self.latest_detections[:3]):  # Show max 3
                    det_lines.append(
                        f"  ID:{det.cube_id} Conf:{det.confidence:.2f}"
                    )
                self.add_text_overlay(annotated_labeled, det_lines, (10, height - 90))

        # Combine frames side by side
        combined = np.hstack([raw_labeled, annotated_labeled])

        # Display
        cv2.imshow(self.window_name, combined)

        # Display debug images
        for name, img in self.latest_debug_images.items():
            cv2.imshow(f"Debug: {name}", img)

    def save_frames(self):
        """Save current frames to disk."""
        if self.latest_raw_frame is None:
            print("⚠️  No frames to save yet")
            return

        timestamp = time.strftime("%Y%m%d_%H%M%S")

        # Save raw frame
        raw_filename = f"frame_raw_{timestamp}.png"
        self.latest_raw_frame.save(raw_filename)
        print(f"💾 Saved raw frame: {raw_filename}")

        # Save annotated frame
        if self.latest_annotated_frame is not None:
            annotated_filename = f"frame_annotated_{timestamp}.png"
            self.latest_annotated_frame.save(annotated_filename)
            print(f"💾 Saved annotated frame: {annotated_filename}")

        print(f"   Detections: {len(self.latest_detections)}")

    def run(self):
        """Main display loop."""
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

        while self.running:
            self.display_frames()

            # Render 3D viewer
            if self.use_3d_viewer and self.show_3d and self.viewer_3d:
                if not self.viewer_3d.render():
                    print("\n🛑 3D viewer closed")
                    self.running = False

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                print("\n🛑 Quit requested")
                self.running = False
            elif key == ord('s'):
                print("\n📸 Saving frames...")
                self.save_frames()
            elif key == ord('f'):
                self.show_fps = not self.show_fps
                print(f"\n📊 FPS display: {'ON' if self.show_fps else 'OFF'}")
            elif key == ord('c'):
                self.detection_enabled = not self.detection_enabled
                print(f"\n🎲 Cube detection: {'ON' if self.detection_enabled else 'OFF'}")
                # Note: This doesn't actually disable detection, just changes the flag
                # To fully disable, you'd need to stop the vision processor
            elif key == ord('3') and self.use_3d_viewer:
                self.show_3d = not self.show_3d
                print(f"\n🎮 3D viewer: {'ON' if self.show_3d else 'OFF'}")

        cv2.destroyAllWindows()
        if self.viewer_3d:
            self.viewer_3d.close()


def main():
    """Main entry point."""

    # Create display handler with 3D viewer
    display = VisionDisplay(use_3d_viewer=VIEWER_3D_AVAILABLE)

    try:
        # Connect to Cozmo
        print("🤖 Connecting to Cozmo...")
        with pycozmo.connect() as cli:
            print("✅ Connected!\n")

            # Position robot for better viewing
            print("📷 Positioning robot...")
            angle = (pycozmo.robot.MAX_HEAD_ANGLE.radians -
                     pycozmo.robot.MIN_HEAD_ANGLE.radians) / 80.0

            cli.set_head_angle(angle)
            time.sleep(1.0)

            # Retrieve camera calibration from Cozmo
            print("📐 Retrieving camera calibration from Cozmo...")
            calibration_retriever = CameraCalibrationRetriever(cli)
            camera_calibration = calibration_retriever.get_calibration(timeout=10.0)

            print(f"✅ Camera calibration retrieved:")
            print(f"   fx={camera_calibration.fx:.2f}, fy={camera_calibration.fy:.2f}")
            print(f"   cx={camera_calibration.cx:.2f}, cy={camera_calibration.cy:.2f}")

            # Register handler for raw frames
            cli.add_handler(
                pycozmo.event.EvtNewRawCameraImage,
                display.on_raw_camera_image
            )

            # Create vision processor with cubeDetector.py
            print("🎨 Initializing vision processor with cubeDetector.py...")
            vision_processor = vision.AnnotatedVisionProcessor(
                client=cli,
                camera_calibration=camera_calibration,
                process_every_n_frames=1  # Process every frame (fast computer)
            )

            print("✅ Cube detector initialized with camera calibration")

            # Store reference for 3D viewer updates
            display.vision_processor = vision_processor

            # Configure annotation display
            vision_processor.show_confidence = True
            vision_processor.show_cube_id = True
            vision_processor.box_thickness = 2

            # Register handler for annotated frames
            cli.add_handler(
                pycozmo.event.EvtAnnotatedCameraImage,
                display.on_annotated_camera_image
            )

            # Register handler for debug images
            cli.add_handler(
                pycozmo.event.EvtCubeDetectorDebugImages,
                display.on_debug_images
            )

            # Enable camera
            print("🎥 Enabling camera...")
            cli.enable_camera(enable=True, color=False)
            # Set camera params for auto exposure
            cli.set_camera_params(exposure_ms=100)

            # Wait for camera to stabilize
            time.sleep(2.0)

            print("✨ Starting display...\n")
            if display.use_3d_viewer:
                print("🎮 3D Viewer Controls:")
                print("   W/S/A/D - Move camera")
                print("   Q/E - Move camera up/down")
                print("   Arrow keys - Rotate camera\n")

            # Run display loop
            display.run()

            print("\n✅ Display closed")

    except KeyboardInterrupt:
        print("\n\n🛑 Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n👋 Goodbye!\n")


if __name__ == '__main__':
    main()
