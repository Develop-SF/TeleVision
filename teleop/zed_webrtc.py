import numpy as np
import cv2
import pyzed.sl as sl
import asyncio
import json
import logging
import os
import platform
import ssl
import argparse
from av import VideoFrame

from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaStreamTrack
from aiortc.rtcrtpsender import RTCRtpSender

# Camera resolution settings
resolution = (720, 1280)
crop_size_w = 1
crop_size_h = 0
resolution_cropped = (resolution[0] - crop_size_h, resolution[1] - 2 * crop_size_w)

class ZEDTrack(MediaStreamTrack):
    kind = "video"

    def __init__(self):
        super().__init__()
        self.zed = sl.Camera()
        init_params = sl.InitParameters()
        init_params.camera_resolution = sl.RESOLUTION.HD720
        init_params.camera_fps = 60
        
        err = self.zed.open(init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            raise Exception(f"Failed to open ZED camera: {err}")
            
        self.image_left = sl.Mat()
        self.image_right = sl.Mat()
        self.runtime_parameters = sl.RuntimeParameters()
        self.timestamp = 0

    async def recv(self):
        if self.zed.grab(self.runtime_parameters) == sl.ERROR_CODE.SUCCESS:
            self.zed.retrieve_image(self.image_left, sl.VIEW.LEFT)
            self.zed.retrieve_image(self.image_right, sl.VIEW.RIGHT)
            
            # Combine left and right images
            rgba = np.hstack((self.image_left.numpy()[crop_size_h:, crop_size_w:-crop_size_w],
                            self.image_right.numpy()[crop_size_h:, crop_size_w:-crop_size_w]))
            rgb = cv2.cvtColor(rgba, cv2.COLOR_RGBA2RGB)
            
            # Create a VideoFrame
            frame = VideoFrame.from_ndarray(rgb, format='bgr24')
            frame.pts = self.timestamp
            frame.time_base = '1/90000'  # 90kHz timebase
            self.timestamp += 3000  # Increment timestamp by 1/30th of a second at 90kHz
            
            return frame
        return None

    def __del__(self):
        self.zed.close()

pcs = set()

async def index(request):
    content = open(os.path.join(os.path.dirname(__file__), "webrtc", "index.html"), "r").read()
    return web.Response(content_type="text/html", text=content)

async def javascript(request):
    content = open(os.path.join(os.path.dirname(__file__), "webrtc", "client.js"), "r").read()
    return web.Response(content_type="application/javascript", text=content)

async def offer(request):
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    pc = RTCPeerConnection()
    pcs.add(pc)

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        print("Connection state is %s" % pc.connectionState)
        if pc.connectionState == "failed":
            await pc.close()
            pcs.discard(pc)

    # Add ZED track
    pc.addTrack(ZEDTrack())

    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return web.Response(
        content_type="application/json",
        text=json.dumps(
            {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
        ),
    )

async def on_shutdown(app):
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ZED WebRTC streaming")
    parser.add_argument("--cert-file", help="SSL certificate file (for HTTPS)")
    parser.add_argument("--key-file", help="SSL key file (for HTTPS)")
    parser.add_argument("--host", default="0.0.0.0", help="Host for HTTP server (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8080, help="Port for HTTP server (default: 8080)")
    parser.add_argument("--verbose", "-v", action="count")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    if args.cert_file:
        ssl_context = ssl.SSLContext()
        ssl_context.load_cert_chain(args.cert_file, args.key_file)
    else:
        ssl_context = None

    app = web.Application()
    app.on_shutdown.append(on_shutdown)
    app.router.add_get("/", index)
    app.router.add_get("/client.js", javascript)
    app.router.add_post("/offer", offer)
    
    print("Starting ZED WebRTC server...")
    print(f"Open your browser and go to: https://[your-local-ip]:{args.port}")
    
    web.run_app(app, host=args.host, port=args.port, ssl_context=ssl_context) 