document.addEventListener('DOMContentLoaded', () => {
    const video = document.getElementById('video');
    const statusDiv = document.getElementById('status');
    const overlayCanvas = document.getElementById('overlay-canvas');
    const overlayContext = overlayCanvas.getContext('2d');
    const form = document.getElementById('login-form');
    const loginButton = document.getElementById('login-button');
    let modelsLoaded = false;
    let cameraStream = null;
    let detectionInterval = null;

    async function startCamera() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ video: true });
            video.srcObject = stream;
            cameraStream = stream;
            video.onloadedmetadata = () => {
                video.play();
                overlayCanvas.width = video.videoWidth;
                overlayCanvas.height = video.videoHeight;
            };
        } catch (err) {
            console.error("Error accessing camera: ", err);
            statusDiv.innerText = "Error: Could not access camera.";
            loginButton.disabled = true;
        }
    }

    async function loadModels() {
        const MODEL_URL = 'static/models';
        await faceapi.nets.tinyFaceDetector.loadFromUri(MODEL_URL);
        await faceapi.nets.faceLandmark68Net.loadFromUri(MODEL_URL);
        modelsLoaded = true;
        statusDiv.innerText = "Camera active. Enter your details and click login.";
    }

    video.addEventListener('play', () => {
        detectionInterval = setInterval(async () => {
            if (modelsLoaded) {
                const detections = await faceapi.detectSingleFace(video, new faceapi.TinyFaceDetectorOptions()).withFaceLandmarks();
                overlayContext.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
                if (detections) {
                    const resizedDetection = faceapi.resizeResults(detections, { width: overlayCanvas.width, height: overlayCanvas.height });
                    faceapi.draw.drawDetections(overlayCanvas, resizedDetection);
                    faceapi.draw.drawFaceLandmarks(overlayCanvas, resizedDetection);
                }
            }
        }, 100);
    });

    function stopCamera() {
        if (cameraStream) {
            cameraStream.getTracks().forEach(track => track.stop());
            video.srcObject = null;
        }
        if (detectionInterval) {
            clearInterval(detectionInterval);
        }
    }

    form.addEventListener('submit', (event) => {
        event.preventDefault();
        loginAndVerify();
    });

    function loginAndVerify() {
        const loginData = {
            voter_id: document.getElementById('voter_id').value,
            email: document.getElementById('email').value,
        };

        const canvas = document.createElement('canvas');
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        const context = canvas.getContext('2d');
        context.drawImage(video, 0, 0, canvas.width, canvas.height);
        const imageDataURL = canvas.toDataURL('image/png');
        loginData.image_data_url = imageDataURL;

        statusDiv.innerText = "Scanning face and verifying details...";
        loginButton.disabled = true;

        fetch('http://127.0.0.1:5000/login', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(loginData),
        })
        .then(response => response.json())
        .then(data => {
            console.log('Server response:', data);
            loginButton.disabled = false;
            if (data.message) {
                statusDiv.innerText = data.message;
                statusDiv.style.color = "green";
                stopCamera();
                // Redirect to the OTP verification page, passing the voter_id and email
                window.location.href = `/otp_verification?voter_id=${loginData.voter_id}&email=${loginData.email}`;
            } else if (data.error) {
                statusDidocument.addEventListener('DOMContentLoaded', () => {
                    const video = document.getElementById('video');
                    const statusDiv = document.getElementById('status');
                    const overlayCanvas = document.getElementById('overlay-canvas');
                    const overlayContext = overlayCanvas.getContext('2d');
                    const form = document.getElementById('login-form');
                    const loginButton = document.getElementById('login-button');
                    let modelsLoaded = false;
                    let cameraStream = null;
                    let detectionInterval = null;
                
                    async function startCamera() {
                        try {
                            const stream = await navigator.mediaDevices.getUserMedia({ video: true });
                            video.srcObject = stream;
                            cameraStream = stream;
                            video.onloadedmetadata = () => {
                                video.play();
                                overlayCanvas.width = video.videoWidth;
                                overlayCanvas.height = video.videoHeight;
                            };
                        } catch (err) {
                            console.error("Error accessing camera: ", err);
                            statusDiv.innerText = "Error: Could not access camera.";
                            loginButton.disabled = true;
                        }
                    }
                
                    async function loadModels() {
                        const MODEL_URL = 'static/models';
                        await faceapi.nets.tinyFaceDetector.loadFromUri(MODEL_URL);
                        await faceapi.nets.faceLandmark68Net.loadFromUri(MODEL_URL);
                        modelsLoaded = true;
                        statusDiv.innerText = "Camera active. Enter your details and click login.";
                    }
                
                    video.addEventListener('play', () => {
                        detectionInterval = setInterval(async () => {
                            if (modelsLoaded) {
                                const detections = await faceapi.detectSingleFace(video, new faceapi.TinyFaceDetectorOptions()).withFaceLandmarks();
                                overlayContext.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
                                if (detections) {
                                    const resizedDetection = faceapi.resizeResults(detections, { width: overlayCanvas.width, height: overlayCanvas.height });
                                    faceapi.draw.drawDetections(overlayCanvas, resizedDetection);
                                    faceapi.draw.drawFaceLandmarks(overlayCanvas, resizedDetection);
                                }
                            }
                        }, 100);
                    });
                
                    function stopCamera() {
                        if (cameraStream) {
                            cameraStream.getTracks().forEach(track => track.stop());
                            video.srcObject = null;
                        }
                        if (detectionInterval) {
                            clearInterval(detectionInterval);
                        }
                    }
                
                    form.addEventListener('submit', (event) => {
                        event.preventDefault();
                        loginAndVerify();
                    });
                
                    function loginAndVerify() {
                        const loginData = {
                            voter_id: document.getElementById('voter_id').value,
                            email: document.getElementById('email').value,
                        };
                
                        const canvas = document.createElement('canvas');
                        canvas.width = video.videoWidth;
                        canvas.height = video.videoHeight;
                        const context = canvas.getContext('2d');
                        context.drawImage(video, 0, 0, canvas.width, canvas.height);
                        const imageDataURL = canvas.toDataURL('image/png');
                        loginData.image_data_url = imageDataURL;
                
                        statusDiv.innerText = "Scanning face and verifying details...";
                        loginButton.disabled = true;
                
                        fetch('http://127.0.0.1:5000/login', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify(loginData),
                        })
                        .then(response => response.json())
                        .then(data => {
                            console.log('Server response:', data);
                            loginButton.disabled = false;
                            if (data.message) {
                                statusDiv.innerText = data.message;
                                statusDiv.style.color = "green";
                                stopCamera();
                                // Redirect to the OTP verification page, passing the voter_id and email
                                window.location.href = `/otp_verification?voter_id=${loginData.voter_id}&email=${loginData.email}`;
                            } else if (data.error) {
                                statusDiv.innerText = `Error: ${data.error}`;
                                statusDiv.style.color = "red";
                            }
                        })
                        .catch((error) => {
                            console.error('Error:', error);
                            statusDiv.innerText = `Network error. Is the server running?`;
                            statusDiv.style.color = "red";
                            loginButton.disabled = false;
                        });
                    }
                
                    loadModels();
                    startCamera();
                });
                v.innerText = `Error: ${data.error}`;
                statusDiv.style.color = "red";
            }
        })
        .catch((error) => {
            console.error('Error:', error);
            statusDiv.innerText = `Network error. Is the server running?`;
            statusDiv.style.color = "red";
            loginButton.disabled = false;
        });
    }

    loadModels();
    startCamera();
});




// document.addEventListener('DOMContentLoaded', () => {
//     const video = document.getElementById('video');
//     const statusDiv = document.getElementById('status');
//     const overlayCanvas = document.getElementById('overlay-canvas');
//     const overlayContext = overlayCanvas.getContext('2d');
//     const form = document.getElementById('login-form');
//     const loginButton = document.getElementById('login-button');
//     let modelsLoaded = false;
//     let cameraStream = null;
//     let detectionInterval = null;

//     async function startCamera() {
//         try {
//             const stream = await navigator.mediaDevices.getUserMedia({ video: true });
//             video.srcObject = stream;
//             cameraStream = stream;
//             video.onloadedmetadata = () => {
//                 video.play();
//                 overlayCanvas.width = video.videoWidth;
//                 overlayCanvas.height = video.videoHeight;
//             };
//         } catch (err) {
//             console.error("Error accessing camera: ", err);
//             statusDiv.innerText = "Error: Could not access camera.";
//             loginButton.disabled = true;
//         }
//     }

//     async function loadModels() {
//         const MODEL_URL = 'static/models';
//         await faceapi.nets.tinyFaceDetector.loadFromUri(MODEL_URL);
//         await faceapi.nets.faceLandmark68Net.loadFromUri(MODEL_URL);
//         modelsLoaded = true;
//         statusDiv.innerText = "Camera active. Enter your details and click login.";
//     }

//     video.addEventListener('play', () => {
//         detectionInterval = setInterval(async () => {
//             if (modelsLoaded) {
//                 const detections = await faceapi.detectSingleFace(video, new faceapi.TinyFaceDetectorOptions()).withFaceLandmarks();
//                 overlayContext.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
//                 if (detections) {
//                     const resizedDetection = faceapi.resizeResults(detections, { width: overlayCanvas.width, height: overlayCanvas.height });
//                     faceapi.draw.drawDetections(overlayCanvas, resizedDetection);
//                     faceapi.draw.drawFaceLandmarks(overlayCanvas, resizedDetection);
//                 }
//             }
//         }, 100);
//     });

//     function stopCamera() {
//         if (cameraStream) {
//             cameraStream.getTracks().forEach(track => track.stop());
//             video.srcObject = null;
//         }
//         if (detectionInterval) {
//             clearInterval(detectionInterval);
//         }
//     }

//     form.addEventListener('submit', (event) => {
//         event.preventDefault();
//         loginAndVerify();
//     });

//     function loginAndVerify() {
//         const loginData = {
//             voter_id: document.getElementById('voter_id').value,
//             email: document.getElementById('email').value,
//         };

//         const canvas = document.createElement('canvas');
//         canvas.width = video.videoWidth;
//         canvas.height = video.videoHeight;
//         const context = canvas.getContext('2d');
//         context.drawImage(video, 0, 0, canvas.width, canvas.height);
//         const imageDataURL = canvas.toDataURL('image/png');
//         loginData.image_data_url = imageDataURL;

//         statusDiv.innerText = "Scanning face and verifying details...";
//         loginButton.disabled = true;

//         fetch('http://127.0.0.1:5000/login', {
//             method: 'POST',
//             headers: { 'Content-Type': 'application/json' },
//             body: JSON.stringify(loginData),
//         })
//         .then(response => response.json())
//         .then(data => {
//             console.log('Server response:', data);
//             loginButton.disabled = false;
//             if (data.message) {
//                 statusDiv.innerText = data.message;
//                 statusDiv.style.color = "green";
//                 stopCamera();
//                 // Redirect to the OTP verification page, passing the voter_id and email
//                 window.location.href = `/otp_verification?voter_id=${loginData.voter_id}&email=${loginData.email}`;
//             } else if (data.error) {
//                 statusDiv.innerText = `Error: ${data.error}`;
//                 statusDiv.style.color = "red";
//             }
//         })
//         .catch((error) => {
//             console.error('Error:', error);
//             statusDiv.innerText = `Network error. Is the server running?`;
//             statusDiv.style.color = "red";
//             loginButton.disabled = false;
//         });
//     }

//     loadModels();
//     startCamera();
// });
