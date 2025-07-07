var pc = null;
var connectionRetries = 0;
var autoReconnect = true;
var reconnectTimeout = null;

function negotiate() {
    if (!pc) {
        console.error("No peer connection available");
        return Promise.reject("No peer connection");
    }
    
    // Add status message
    updateStatus("Connecting to stereo camera...");
    
    pc.addTransceiver('video', { direction: 'recvonly' });
    pc.addTransceiver('audio', { direction: 'recvonly' });
    return pc.createOffer().then((offer) => {
        return pc.setLocalDescription(offer);
    }).then(() => {
        // wait for ICE gathering to complete
        return new Promise((resolve) => {
            if (pc.iceGatheringState === 'complete') {
                resolve();
            } else {
                const checkState = () => {
                    if (pc.iceGatheringState === 'complete') {
                        pc.removeEventListener('icegatheringstatechange', checkState);
                        resolve();
                    }
                };
                pc.addEventListener('icegatheringstatechange', checkState);
            }
        });
    }).then(() => {
        var offer = pc.localDescription;
        updateStatus("Sending connection request...");
        
        // Use relative URL if on same domain, but force HTTP
        const currentUrl = window.location.href;
        const baseUrl = currentUrl.replace(/^https:/, 'http:');
        const offerUrl = new URL('/offer', baseUrl).href;
        
        console.log("Connecting to WebRTC server at:", offerUrl);
        updateStatus(`Connecting to: ${offerUrl}`);
        
        return fetch(offerUrl, {
            body: JSON.stringify({
                sdp: offer.sdp,
                type: offer.type,
            }),
            headers: {
                'Content-Type': 'application/json'
            },
            method: 'POST'
        });
    }).then((response) => {
        if (!response.ok) {
            throw new Error(`Server error: ${response.status}`);
        }
        return response.json();
    }).then((answer) => {
        console.log("Received answer from server");
        updateStatus("Connection established, waiting for video...");
        return pc.setRemoteDescription(answer);
    }).catch((e) => {
        console.error("Connection error:", e);
        updateStatus(`Connection failed: ${e.message || e}`);
        
        // Add more debugging info
        if (e.message && e.message.includes("Failed to fetch")) {
            updateStatus("Network error: Make sure server is running on port 8080 and using HTTP");
            document.getElementById('connection-debug').innerHTML = `
                <p><strong>Debugging tips:</strong></p>
                <ul>
                    <li>Check that the WebRTC server is running</li>
                    <li>Make sure you're using <code>http://</code> not <code>https://</code></li>
                    <li>Try accessing <a href="http://${window.location.hostname}:8080" target="_blank">http://${window.location.hostname}:8080</a> directly</li>
                </ul>
            `;
        }
        
        // Increment retry counter
        connectionRetries++;
        
        if (autoReconnect && connectionRetries < 5) {
            updateStatus(`Retrying connection (${connectionRetries}/5)...`);
            // Try to reconnect after a delay
            if (reconnectTimeout) {
                clearTimeout(reconnectTimeout);
            }
            reconnectTimeout = setTimeout(() => {
                stop(true);
                start();
            }, 2000);
        } else if (connectionRetries >= 5) {
            updateStatus("Connection failed after multiple attempts. Please refresh page to try again.");
        }
    });
}

function updateStatus(message) {
    const statusEl = document.getElementById('status');
    if (statusEl) {
        statusEl.textContent = message;
    } else {
        console.log("Status:", message);
    }
}

function start() {
    var config = {
        sdpSemantics: 'unified-plan'
    };

    if (document.getElementById('use-stun').checked) {
        config.iceServers = [{ urls: ['stun:stun.l.google.com:19302'] }];
    }

    pc = new RTCPeerConnection(config);

    // Add connection state monitoring
    pc.addEventListener('connectionstatechange', () => {
        console.log("Connection state:", pc.connectionState);
        if (pc.connectionState === 'connected') {
            connectionRetries = 0; // Reset counter on successful connection
            updateStatus("Connected to stereo camera stream");
        } else if (pc.connectionState === 'failed' || pc.connectionState === 'disconnected') {
            updateStatus(`Connection ${pc.connectionState}. Will try to reconnect...`);
            if (autoReconnect) {
                if (reconnectTimeout) {
                    clearTimeout(reconnectTimeout);
                }
                reconnectTimeout = setTimeout(() => {
                    stop(true);
                    start();
                }, 2000);
            }
        }
    });

    // connect audio / video
    pc.addEventListener('track', (evt) => {
        if (evt.track.kind == 'video') {
            const videoEl = document.getElementById('video');
            videoEl.srcObject = evt.streams[0];
            
            // Add event listener for when video starts playing
            videoEl.onloadeddata = () => {
                updateStatus("Video stream active");
                videoEl.play().catch(e => {
                    console.error("Error starting video playback:", e);
                    updateStatus("Error starting video: " + e.message);
                });
            };
        } else {
            document.getElementById('audio').srcObject = evt.streams[0];
        }
    });

    document.getElementById('start').style.display = 'none';
    negotiate();
    document.getElementById('stop').style.display = 'inline-block';
}

function stop(silent = false) {
    if (reconnectTimeout) {
        clearTimeout(reconnectTimeout);
        reconnectTimeout = null;
    }
    
    if (!silent) {
        document.getElementById('stop').style.display = 'none';
        document.getElementById('start').style.display = 'inline-block';
        updateStatus("Connection stopped");
        autoReconnect = false;
    }

    // Close video tracks
    if (document.getElementById('video').srcObject) {
        document.getElementById('video').srcObject.getTracks().forEach(track => track.stop());
        document.getElementById('video').srcObject = null;
    }
    
    if (document.getElementById('audio').srcObject) {
        document.getElementById('audio').srcObject.getTracks().forEach(track => track.stop());
        document.getElementById('audio').srcObject = null;
    }

    // close peer connection
    if (pc) {
        pc.close();
        pc = null;
    }
    
    if (!silent) {
        connectionRetries = 0;
    }
}

// Initialize on page load
window.addEventListener('DOMContentLoaded', () => {
    // Add status element if it doesn't exist
    if (!document.getElementById('status')) {
        const statusDiv = document.createElement('div');
        statusDiv.id = 'status-container';
        statusDiv.innerHTML = '<h3>Connection Status</h3><p id="status">Ready to connect</p>';
        document.getElementById('media').before(statusDiv);
    }
    
    // Add connection info
    const infoEl = document.getElementById('connection-debug');
    if (infoEl) {
        infoEl.innerHTML = `
            <p><strong>Connection Information:</strong></p>
            <ul>
                <li>Current URL: ${window.location.href}</li>
                <li>WebRTC server should be at: http://${window.location.hostname}:8080</li>
                <li>Click "Connect to Camera" to start streaming</li>
            </ul>
        `;
    }
    
    // Auto-start option - if autostart=true in URL or on localhost:8080
    const urlParams = new URLSearchParams(window.location.search);
    const shouldAutoStart = urlParams.get('autostart') === 'true' || 
                            (window.location.hostname === 'localhost' && window.location.port === '8080') ||
                            (window.location.hostname === '127.0.0.1' && window.location.port === '8080');
    
    if (shouldAutoStart) {
        console.log("Auto-starting connection");
        setTimeout(start, 500);
    }
});