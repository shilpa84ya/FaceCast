document.addEventListener('DOMContentLoaded', () => {
    const video = document.getElementById('video');
    const statusDiv = document.getElementById('status');
    const overlayCanvas = document.getElementById('overlay-canvas');
    const overlayContext = overlayCanvas.getContext('2d');
    const form = document.getElementById('registration-form');
    const registerButton = document.getElementById('register-button');
    const cameraSection = document.getElementById('camera-section');
    const successContainer = document.getElementById('success-container');
    const ageDeniedContainer = document.getElementById('age-denied-container');
    const loginSuggestionContainer = document = document.getElementById('login-suggestion-container');
    const dobInput = document.getElementById('dob');
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
            registerButton.disabled = true;
        }
    }

    async function loadModels() {
        const MODEL_URL = 'static/models'; 
        await faceapi.nets.tinyFaceDetector.loadFromUri(MODEL_URL);
        await faceapi.nets.faceLandmark68Net.loadFromUri(MODEL_URL);
        modelsLoaded = true;
        statusDiv.innerText = "Camera active. Fill the form and click register.";
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

    function handleRegistrationSuccess() {
        stopCamera();
        form.style.display = 'none';
        cameraSection.style.display = 'none';
        successContainer.style.display = 'block';
    }

    form.addEventListener('submit', (event) => {
        event.preventDefault();
        captureAndRegister();
    });

    function captureAndRegister() {
        const dob = new Date(dobInput.value);
        const today = new Date();
        let age = today.getFullYear() - dob.getFullYear();
        const m = today.getMonth() - dob.getMonth();
        if (m < 0 || (m === 0 && today.getDate() < dob.getDate())) {
            age--;
        }

        if (age < 18) {
            stopCamera();
            form.style.display = 'none';
            cameraSection.style.display = 'none';
            ageDeniedContainer.style.display = 'block';
            return;
        }

        const voterData = {
            name: document.getElementById('name').value,
            address: document.getElementById('address').value,
            gender: document.getElementById('gender').value,
            aadhaar: document.getElementById('aadhaar').value,
            phone: document.getElementById('phone').value,
            email: document.getElementById('email').value,
            voter_id: document.getElementById('voter_id').value,
            dob: dobInput.value,
        };

        const canvas = document.createElement('canvas');
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        const context = canvas.getContext('2d');
        context.drawImage(video, 0, 0, canvas.width, canvas.height);
        const imageDataURL = canvas.toDataURL('image/png');
        voterData.image_data_url = imageDataURL;

        statusDiv.innerText = "Image captured! Sending to server...";
        registerButton.disabled = true;

        fetch('/register', { // <-- Updated to a relative path
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(voterData),
        })
        .then(response => response.json())
        .then(data => {
            console.log('Server response:', data);
            registerButton.disabled = false;
            if (data.message) {
                handleRegistrationSuccess();
            } else if (data.error) {
                if (data.error.includes("already exists") || data.error.includes("face is already registered")) {
                    stopCamera();
                    form.style.display = 'none';
                    cameraSection.style.display = 'none';
                    loginSuggestionContainer.style.display = 'block';
                } else {
                    statusDiv.innerText = `Error: ${data.error}`;
                    statusDiv.style.color = "red";
                }
            }
        })
        .catch((error) => {
            console.error('Error:', error);
            statusDiv.innerText = `Network error. Is the server running?`;
            statusDiv.style.color = "red";
            registerButton.disabled = false;
        });
    }
    
    function handleRegistrationSuccess() {
        stopCamera();
        form.style.display = 'none';
        cameraSection.style.display = 'none';
        successContainer.style.display = 'block';
    }

    loadModels();
    startCamera();
});
