import asyncio
import json
import logging
import os
import platform
import ssl

import aiohttp_cors
from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaPlayer, MediaRelay
from aiortc.rtcrtpsender import RTCRtpSender
from multiprocessing import Process, Array, Value, shared_memory, Queue, Event

ROOT = os.path.dirname(__file__)

relay = None
webcam = None


from aiortc import MediaStreamTrack
from av import VideoFrame
import numpy as np
import time 
import pyzed.sl as sl
import cv2

class ZedVideoTrack(MediaStreamTrack):
    kind = "video"
    def __init__(self, queue, toggle_streaming, fps):
        super().__init__()  # Initialize base class
        
        # Disable direct ZED connection to avoid conflicts
        self.direct_zed = False
        print("Using queue-based approach for ZED streaming")
        
        # Setup for queue-based approach
        self.img_queue = queue
        self.toggle_streaming = toggle_streaming
        self.streaming_started = False
        self.timescale = 90000  # Standard video timebase (90kHz)
        self.timestamp = 0
        self.fps = fps
        self.frame_interval = int(self.timescale / fps)  # Frame interval in timebase units
        self.start_time = time.time()
        self.frame_counter = 0
        self.last_success_time = 0
        self.consecutive_errors = 0
        
        # Setup statistics for debugging
        self.stats = {
            "frames_received": 0,
            "frames_sent": 0,
            "errors": 0,
            "last_frame_time": 0,
            "avg_frame_interval": 0
        }
        
        # Default frame size for fallback frames
        self.default_height = 720
        self.default_width = 1280
    
    async def recv(self):
        """
        This method is called when a new frame is needed.
        """
        # Signal the main process to start streaming
        if not self.streaming_started:
            print("WebRTC client connected - starting ZED camera stream")
            self.toggle_streaming.set()
            self.streaming_started = True
        
        try:
            # Try to get a frame from the queue with timeout
            frame = None
            try:
                # Prioritize getting the latest frame
                if hasattr(self.img_queue, 'qsize') and self.img_queue.qsize() > 1:
                    # Multiple frames waiting, skip to the most recent
                    try:
                        while self.img_queue.qsize() > 1:
                            frame = self.img_queue.get_nowait()
                            self.stats["frames_received"] += 1
                    except:
                        pass
                else:
                    # Just get the next frame with a short timeout
                    frame = self.img_queue.get(timeout=0.1)
                    self.stats["frames_received"] += 1
                
                # Track timing for statistics
                now = time.time()
                if self.stats["last_frame_time"] > 0:
                    interval = now - self.stats["last_frame_time"]
                    # Update rolling average (with 90% previous value, 10% new value)
                    if self.stats["avg_frame_interval"] > 0:
                        self.stats["avg_frame_interval"] = 0.9 * self.stats["avg_frame_interval"] + 0.1 * interval
                    else:
                        self.stats["avg_frame_interval"] = interval
                self.stats["last_frame_time"] = now
                
            except Exception as qerr:
                if self.consecutive_errors == 0 or self.consecutive_errors % 100 == 0:
                    print(f"Queue error: {qerr}")
                frame = None
            
            # Check frame validity
            if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
                raise ValueError("Invalid or empty frame received")
                
            # Get frame info for debugging
            h, w = frame.shape[:2]
            if self.frame_counter % 300 == 0:  # Log every ~5 seconds at 60fps
                avg_fps = 1.0 / self.stats["avg_frame_interval"] if self.stats["avg_frame_interval"] > 0 else 0
                print(f"WebRTC stream stats: {w}x{h} frames, avg {avg_fps:.1f} fps")
            
            # Ensure frame is properly formatted for encoding
            if frame.dtype != np.uint8:
                frame = frame.astype(np.uint8)
            
            # Convert to RGB if needed
            if len(frame.shape) == 2:  # Grayscale
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            elif frame.shape[2] == 4:  # RGBA
                frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
            elif frame.shape[2] != 3:  # Not RGB
                raise ValueError(f"Unsupported frame format: {frame.shape}")
            
            # Check if frame needs resizing (H.264 encoder has size limitations)
            max_width = 2560  # Maximum width for H.264 encoding (increased for stereo view)
            max_height = 1440  # Maximum height for H.264 encoding
            
            # Only resize if absolutely necessary, preserving the stereo view aspect ratio
            if w > max_width or h > max_height:
                # Calculate scaling factor, preserving aspect ratio
                scale = min(max_width/w, max_height/h)
                new_w, new_h = int(w*scale), int(h*scale)
                
                # Ensure even dimensions for video encoding
                new_w = new_w - (new_w % 2)
                new_h = new_h - (new_h % 2)
                
                frame = cv2.resize(frame, (new_w, new_h))
                if self.frame_counter % 300 == 0:
                    print(f"Resized frame from {w}x{h} to {new_w}x{new_h} for encoding (scale={scale:.2f})")
            
            # Convert to VideoFrame
            av_frame = VideoFrame.from_ndarray(frame, format='rgb24')
            av_frame.pts = self.timestamp
            av_frame.time_base = '1/90000'
            self.timestamp += self.frame_interval
            
            # Reset error counters on success
            self.consecutive_errors = 0
            self.last_success_time = time.time()
            
            # Update statistics
            self.frame_counter += 1
            self.stats["frames_sent"] += 1
            
            # Log occasional status
            if self.frame_counter % 600 == 0:  # Every 10 seconds at 60fps
                print(f"WebRTC streaming: {self.stats['frames_sent']} frames sent, {self.stats['errors']} errors")
                
            return av_frame
            
        except Exception as e:
            # Track consecutive errors
            self.consecutive_errors += 1
            self.stats["errors"] += 1
            error_duration = time.time() - self.last_success_time if self.last_success_time > 0 else 0
            
            # Log errors (but not too frequently)
            if self.consecutive_errors == 1 or self.consecutive_errors % 300 == 0:
                print(f"WebRTC streaming error: {e} (after {error_duration:.1f}s without frames)")
            
            # Create a black frame as fallback with progress indicator
            black_frame = np.zeros((360, 640, 3), dtype=np.uint8)
            
            # Add error text
            font = cv2.FONT_HERSHEY_SIMPLEX
            if self.consecutive_errors > 50:
                # Alert about persistent errors
                cv2.putText(black_frame, f"Connection issues detected", 
                        (20, 50), font, 0.7, (50, 50, 255), 2)
                cv2.putText(black_frame, f"No frames for {error_duration:.1f}s", 
                        (20, 90), font, 0.6, (50, 50, 255), 1)
            else:
                cv2.putText(black_frame, f"Waiting for ZED frames...", 
                        (20, 70), font, 0.7, (255, 255, 255), 2)
            
            # Add queue status
            queue_status = "unknown"
            if hasattr(self.img_queue, 'qsize'):
                try:
                    queue_status = str(self.img_queue.qsize())
                except:
                    pass
            cv2.putText(black_frame, f"Frame queue: {queue_status}",
                    (20, 130), font, 0.5, (200, 200, 100), 1)
            
            # Add frame counter as progress indicator
            progress = int(time.time() * 2) % 20  # 0-19 moving indicator
            indicator = "[" + " " * progress + "●" + " " * (19-progress) + "]"
            cv2.putText(black_frame, indicator, 
                    (20, 200), font, 0.7, (100, 200, 100), 2)
            
            # Add statistics
            stats_text = f"Frames: {self.stats['frames_received']} rcvd, {self.stats['frames_sent']} sent, {self.stats['errors']} errors"
            cv2.putText(black_frame, stats_text,
                    (20, 240), font, 0.5, (180, 180, 180), 1)
            
            # If many consecutive errors, try to reset streaming
            if self.consecutive_errors % 100 == 50:
                print(f"⚠️ WebRTC stream experiencing issues: {self.consecutive_errors} consecutive errors")
                # Toggle streaming to attempt reset
                self.toggle_streaming.clear()
                await asyncio.sleep(0.1)
                self.toggle_streaming.set()
            
            av_frame = VideoFrame.from_ndarray(black_frame, format='rgb24')
            av_frame.pts = self.timestamp
            av_frame.time_base = '1/90000'
            self.timestamp += self.frame_interval
            return av_frame
    
    def __del__(self):
        print("ZedVideoTrack cleaned up")


def force_codec(pc, sender, forced_codec):
    kind = forced_codec.split("/")[0]
    codecs = RTCRtpSender.getCapabilities(kind).codecs
    transceiver = next(t for t in pc.getTransceivers() if t.sender == sender)
    transceiver.setCodecPreferences(
        [codec for codec in codecs if codec.mimeType == forced_codec]
    )


async def index(request):
    content = open(os.path.join(ROOT, "index.html"), "r").read()
    return web.Response(content_type="text/html", text=content)


async def javascript(request):
    content = open(os.path.join(ROOT, "client.js"), "r").read()
    return web.Response(content_type="application/javascript", text=content)

class RTC():
    def __init__(self, img_shape, img_queue, toggle_streaming, fps) -> None:
        self.img_shape = img_shape
        self.img_queue = img_queue
        self.fps = fps
        self.toggle_streaming = toggle_streaming

    async def offer(self, request):
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

        # open media source
        zed_track = ZedVideoTrack(self.img_queue, self.toggle_streaming, self.fps)
        video_sender = pc.addTrack(zed_track)
        # if Args.video_codec:
        force_codec(pc, video_sender, "video/H264")
        # elif Args.play_without_decoding:
            # raise Exception("You must specify the video codec using --video-codec")

        await pc.setRemoteDescription(offer)

        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)

        return web.Response(
            content_type="application/json",
            text=json.dumps(
                {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
            ),
        )


pcs = set()

async def on_shutdown(app):
    # close peer connections
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()


from params_proto import ParamsProto, Proto, Flag


class Args(ParamsProto):
    description = "WebRTC webcam demo"
    cert_file = Proto(help="SSL certificate file (for HTTPS)")
    key_file = Proto(help="SSL key file (for HTTPS)")

    host = Proto(default="0.0.0.0", help="Host for HTTP server (default: 0.0.0.0)")
    port = Proto(default=8080, dtype=int, help="Port for HTTP server (default: 8080)")

    play_from = Proto(help="Read the media from a file and send it.")
    play_without_decoding = Flag(
        "Read the media without decoding it (experimental). "
        "For now it only works with an MPEGTS container with only H.264 video."
    )

    audio_codec = Proto(help="Force a specific audio codec (e.g. audio/opus)")
    video_codec = Proto(help="Force a specific video codec (e.g. video/H264)")
    img_shape = Proto(help="")
    shm_name = Proto(help="")
    fps = Proto(help="")

    verbose = Flag()


if __name__ == '__main__':

    if Args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    if Args.cert_file:
        print("Using SSL certificate file: %s" % Args.cert_file)
        ssl_context = ssl.SSLContext()
        ssl_context.load_cert_chain(Args.cert_file, Args.key_file)
    else:
        ssl_context = None

    app = web.Application()
    cors = aiohttp_cors.setup(app, defaults={
        "*": aiohttp_cors.ResourceOptions(
            allow_credentials=True,
            expose_headers="*",
            allow_headers="*",
            allow_methods="*",
        )
    })
    queue = Queue()
    rtc = RTC((960, 640), queue, toggle_streaming=Event(), fps=60)
    app.on_shutdown.append(on_shutdown)
    cors.add(app.router.add_get("/", index))
    cors.add(app.router.add_get("/client.js", javascript))
    cors.add(app.router.add_post("/offer", rtc.offer))

    web.run_app(app, host=Args.host, port=Args.port, ssl_context=ssl_context)