#!/usr/bin/env python3

'''
MIT License
2021 GeekAlexis, 2022 RS Components Ltd
'''

from pathlib import Path
import argparse
import logging
import time
import json
import cv2
import websockets
import asyncio
import json
import threading
import copy

import fastmot
from fastmot.trolley import Trolley

websocketSendData = {}

async def websocketHandler(websocket, path):
    while True:
        ld = copy.deepcopy(websocketSendData)
        await websocket.send(json.dumps(ld))
        await asyncio.sleep(1)

def wsThread():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    ws = websockets.serve(websocketHandler, "0.0.0.0", 8765)
    asyncio.get_event_loop().run_until_complete(ws)
    asyncio.get_event_loop().run_forever()

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-m', '--mot', action='store_true', help='run multiple object tracker')
    parser.add_argument('-i', '--input_uri', metavar="URI", required=True, help=
                        'URI to input stream\n'
                        '1) image sequence (e.g. img_%%06d.jpg)\n'
                        '2) video file (e.g. video.mp4)\n'
                        '3) MIPI CSI camera (e.g. csi://0)\n'
                        '4) USB/V4L2 camera (e.g. /dev/video0)\n'
                        '5) RTSP stream (rtsp://<user>:<password>@<ip>:<port>/<path>)\n'
                        '6) HTTP stream (http://<user>:<password>@<ip>:<port>/<path>)\n')
    parser.add_argument('-o', '--output_uri', metavar="URI",
                        help='URI to output video (e.g. output.mp4)')
    parser.add_argument('-l', '--log', metavar="FILE",
                        help='output a MOT Challenge format log (e.g. eval/results/mot17-04.txt)')
    parser.add_argument('-g', '--gui', action='store_true', help='enable display')
    parser.add_argument('-v', '--verbose', action='store_true', help='verbose output for debugging')
    args = parser.parse_args()

    # set up logging
    logging.basicConfig(format='[%(levelname)s] %(message)s')
    logger = logging.getLogger(fastmot.__name__)
    logger.setLevel(logging.DEBUG if args.verbose else logging.INFO)

    '''
    ws = websockets.serve(websocketHandler, "0.0.0.0", 8765)
    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, blocking)
    loop.run_until_complete(ws)
    loop.run_forever()
    '''
    wst = threading.Thread(target=wsThread, daemon=True)
    wst.start()

    # Trolley configuration
    t = Trolley()
    centreLimits = 200
    motorStatus = 0

    # load config file
    with open(Path(__file__).parent / 'cfg' / 'mot.json') as config_file:
        config = json.load(config_file, cls=fastmot.utils.ConfigDecoder)

    mot = None
    log = None
    elapsed_time = 0
    stream = fastmot.VideoIO(config['resize_to'], config['video_io'], args.input_uri, args.output_uri)

    if args.mot:
        draw = args.gui or args.output_uri is not None
        mot = fastmot.MOT(config['resize_to'], stream.cap_dt, config['mot'],
                          draw=draw, verbose=args.verbose)
        if args.log is not None:
            Path(args.log).parent.mkdir(parents=True, exist_ok=True)
            log = open(args.log, 'w')
    if args.gui:
        cv2.namedWindow("Video", cv2.WINDOW_AUTOSIZE)

    t.setStacklight(Trolley.LIGHT_GREEN, state=True)

    logger.info('Starting video capture...')
    stream.start_capture()
    try:
        tic = time.perf_counter()
        while not args.gui or cv2.getWindowProperty("Video", 0) >= 0:
            frame = stream.read()
            if frame is None:
                break

            # Get frame width, height & draw bounds on screen
            fw = frame.shape[1]
            fwh = int(fw/2)
            fh = frame.shape[0]
            fhh = int(fh/2)
            cv2.drawMarker(frame, (fwh, fhh), color=(0, 0, 255), markerType=cv2.MARKER_CROSS, thickness=2)
            cv2.line(frame, (fwh-centreLimits, 0), (fwh-centreLimits, fh), color=(0, 0, 255), thickness=2)
            cv2.line(frame, (fwh+centreLimits, 0), (fwh+centreLimits, fh), color=(0, 0, 255), thickness=2)


            if args.mot:
                mot.step(frame)

                websocketSendData['type'] = 'tracks'
                websocketSendData['data'] = {}

                if mot.visible_tracks:
                    t.setStacklight(Trolley.LIGHT_RED, state=True)
                    # Trolley command
                    for track in mot.visible_tracks:
                        tl, br = tuple(track.tlbr[:2]), tuple(track.tlbr[2:])
                        tx = tl[0]
                        ty = tl[1]
                        bx = br[0]
                        by = br[1]
                        mx = int(((bx - tx) / 2) + tx)
                        my = int(((by - ty) / 2) + ty)
                        print("ID: {}, middleX: {}, middleY: {}, motorStatus: {}".format(track.trk_id, mx, my, motorStatus))
                        cv2.drawMarker(frame, (mx, my), color=(0, 255, 0), markerType=cv2.MARKER_CROSS, thickness=2)

                        websocketSendData['data'].update({
                            'trk_id': track.trk_id,
                            'mx': mx,
                            'my': my,
                            })
                    
                        if mx > fwh+centreLimits and motorStatus == 0:
                            print("Right")
                            motorStatus = 1
                            t.turnRight(dutyCycle=7)

                        if mx < fwh-centreLimits and motorStatus == 0:
                            print("Left")
                            motorStatus = -1
                            t.turnLeft(dutyCycle=7)

                        if mx < fwh+(centreLimits/2) and motorStatus == 1 and not t.getFrontProx():
                            print("Forward after right")
                            motorStatus = 0
                            t.forward(dutyCycle=12)

                        if mx > fwh-(centreLimits/2) and motorStatus == -1 and not t.getFrontProx():
                            print("Forward after left")
                            motorStatus = 0
                            t.forward(dutyCycle=12)
                else:
                    t.setStacklight(Trolley.LIGHT_RED)
                    t.stop()

                websocketSendData['data'].update({
                    'motorStatus': motorStatus
                    })

                if log is not None:
                    for track in mot.visible_tracks:
                        tl = track.tlbr[:2] / config['resize_to'] * stream.resolution
                        br = track.tlbr[2:] / config['resize_to'] * stream.resolution
                        w, h = br - tl + 1
                        log.write(f'{mot.frame_count},{track.trk_id},{tl[0]:.6f},{tl[1]:.6f},'
                                  f'{w:.6f},{h:.6f},-1,-1,-1\n')

            if args.gui:
                cv2.imshow('Video', frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
            if args.output_uri is not None:
                stream.write(frame)

        toc = time.perf_counter()
        elapsed_time = toc - tic
    finally:
        # clean up resources
        if log is not None:
            log.close()
        stream.release()
        cv2.destroyAllWindows()
        t.setStacklight(Trolley.LIGHT_RED)
        t.setStacklight(Trolley.LIGHT_AMBER)
        t.setStacklight(Trolley.LIGHT_GREEN)
        t.cleanup()

    if args.mot:
        # timing results
        avg_fps = round(mot.frame_count / elapsed_time)
        avg_tracker_time = mot.tracker_time / (mot.frame_count - mot.detector_frame_count)
        avg_extractor_time = mot.extractor_time / mot.detector_frame_count
        avg_preproc_time = mot.preproc_time / mot.detector_frame_count
        avg_detector_time = mot.detector_time / mot.detector_frame_count
        avg_assoc_time = mot.association_time / mot.detector_frame_count

        logger.info('Average FPS: %d', avg_fps)
        logger.debug('Average tracker time: %f', avg_tracker_time)
        logger.debug('Average feature extractor time: %f', avg_extractor_time)
        logger.debug('Average preprocessing time: %f', avg_preproc_time)
        logger.debug('Average detector time: %f', avg_detector_time)
        logger.debug('Average association time: %f', avg_assoc_time)


if __name__ == '__main__':
    main()
