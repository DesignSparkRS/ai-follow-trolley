from enum import Enum
import logging
import time
import cv2

from .detector import SSDDetector, YoloDetector, PublicDetector
from .feature_extractor import FeatureExtractor
from .tracker import MultiTracker
from .utils.visualization import draw_tracks, draw_detections
from .utils.visualization import draw_flow_bboxes, draw_background_flow


LOGGER = logging.getLogger(__name__)


class DetectorType(Enum):
    SSD = 0
    YOLO = 1
    PUBLIC = 2


class MOT:
    """
    This is the top level module that integrates detection, feature extraction,
    and tracking together.
    Parameters
    ----------
    size : (int, int)
        Width and height of each frame.
    cap_dt : float
        Time interval in seconds between each captured frame.
    config : Dict
        Tracker configuration.
    draw : bool
        Flag to toggle visualization drawing.
    verbose : bool
        Flag to toggle output verbosity.
    """

    def __init__(self, size, cap_dt, config, draw=False, verbose=False):
        self.size = size
        self.draw = draw
        self.verbose = verbose
        self.detector_type = DetectorType[config['detector_type']]
        self.detector_frame_skip = config['detector_frame_skip']

        LOGGER.info('Loading detector model...')
        if self.detector_type == DetectorType.SSD:
            self.detector = SSDDetector(self.size, config['ssd_detector'])
        elif self.detector_type == DetectorType.YOLO:
            self.detector = YoloDetector(self.size, config['yolo_detector'])
        elif self.detector_type == DetectorType.PUBLIC:
            self.detector = PublicDetector(self.size, self.detector_frame_skip,
                                           config['public_detector'])

        LOGGER.info('Loading feature extractor model...')
        self.extractor = FeatureExtractor(config['feature_extractor'])
        self.tracker = MultiTracker(self.size, cap_dt, self.extractor.metric,
                                    config['multi_tracker'])

        # reset counters
        self.frame_count = 0
        self.detector_frame_count = 0
        self.preproc_time = 0
        self.detector_time = 0
        self.extractor_time = 0
        self.association_time = 0
        self.tracker_time = 0

    @property
    def visible_tracks(self):
        # retrieve confirmed and active tracks from the tracker
        return [track for track in self.tracker.tracks.values()
                if track.confirmed and track.active]

    def initiate(self):
        """
        Resets multiple object tracker.
        """
        self.frame_count = 0

    def step(self, frame):
        """
        Runs multiple object tracker on the next frame.
        Parameters
        ----------
        frame : ndarray
            The next frame.
        """
        detections = []
        if self.frame_count == 0:
            detections = self.detector(frame)
            self.tracker.initiate(frame, detections)
        else:
            if self.frame_count % self.detector_frame_skip == 0:
                tic = time.perf_counter()
                self.detector.detect_async(frame)
                self.preproc_time += time.perf_counter() - tic
                tic = time.perf_counter()
                self.tracker.compute_flow(frame)
                detections = self.detector.postprocess()
                self.detector_time += time.perf_counter() - tic
                tic = time.perf_counter()
                self.extractor.extract_async(frame, detections)
                self.tracker.apply_kalman()
                embeddings = self.extractor.postprocess()
                self.extractor_time += time.perf_counter() - tic
                tic = time.perf_counter()
                self.tracker.update(self.frame_count, detections, embeddings)
                self.association_time += time.perf_counter() - tic
                self.detector_frame_count += 1
            else:
                tic = time.perf_counter()
                self.tracker.track(frame)
                self.tracker_time += time.perf_counter() - tic

        if self.draw:
            self._draw(frame, detections)
        self.frame_count += 1

    def _draw(self, frame, detections):
        '''
        tracks = self.visible_tracks
        fw = frame.shape[1]
        fwh = int(fw/2)
        fh = frame.shape[0]
        fhh = int(fh/2)
        cv2.drawMarker(frame, (fwh, fhh), color=(0, 0, 255), markerType=cv2.MARKER_CROSS, thickness=2)
        cv2.line(frame, (fwh-self.centreLimits, 0), (fwh-self.centreLimits, fh), color=(0, 0, 255), thickness=2)
        cv2.line(frame, (fwh+self.centreLimits, 0), (fwh+self.centreLimits, fh), color=(0, 0, 255), thickness=2)
        for track in tracks:
            tl, br = tuple(track.tlbr[:2]), tuple(track.tlbr[2:])
            tx = tl[0]
            ty = tl[1]
            bx = br[0]
            by = br[1]
            mx = int(((bx - tx) / 2) + tx)
            my = int(((by - ty) / 2) + ty)
            print("ID: {}, middleX: {}, middleY: {}, motorStatus: {}".format(track.trk_id, mx, my, self.motorStatus))
            cv2.drawMarker(frame, (mx, my), color=(0, 255, 0), markerType=cv2.MARKER_CROSS, thickness=2)

            if mx > fwh+self.centreLimits and self.motorStatus == 0:
                print("Right")
                self.motorStatus = 1
                self.t.turnRight(dutyCycle=4)

            if mx < fwh-self.centreLimits and self.motorStatus == 0:
                print("Left")
                self.motorStatus = -1
                self.t.turnLeft(dutyCycle=5)

            if mx < fwh+(self.centreLimits/2) and self.motorStatus == 1:
                print("Stop after right")
                self.motorStatus = 0
                self.t.stop()

            if mx > fwh-(self.centreLimits/2) and self.motorStatus == -1:
                print("Stop after left")
                self.motorStatus = 0
                self.t.stop()

        draw_tracks(frame, self.visible_tracks, show_flow=self.verbose)
        '''
        if self.verbose:
            draw_detections(frame, detections)
            draw_flow_bboxes(frame, self.tracker)
            draw_background_flow(frame, self.tracker)
        cv2.putText(frame, f'visible: {len(self.visible_tracks)}', (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, 0, 2, cv2.LINE_AA)
