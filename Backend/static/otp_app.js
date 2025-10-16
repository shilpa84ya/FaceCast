document.addEventListener('DOMContentLoaded', () => {
    const otpForm = document.getElementById('otp-form');
    const otpInput = document.getElementById('otp');
    const statusDiv = document.getElementById('status');
    const verifyButton = document.getElementById('verify-button');
    const timerSpan = document.getElementById('timer');
    const resendButton = document.getElementById('resend-button');
    const otpMessage = document.getElementById('otp-message');

    let countdown = 60;
    let timerInterval = null;

    // Get voter ID and email from URL parameters
    const urlParams = new URLSearchParams(window.location.search);
    const voterId = urlParams.get('voter_id');
    const email = urlParams.get('email');
    
    // Display the email where OTP was sent
    if (email) {
        otpMessage.innerText = `An OTP has been sent to ${email}.`;
    }

    function startTimer() {
        countdown = 60;
        timerSpan.innerText = `You can resend OTP in ${countdown}s`;
        resendButton.disabled = true;
        
        if (timerInterval) {
            clearInterval(timerInterval);
        }

        timerInterval = setInterval(() => {
            countdown--;
            timerSpan.innerText = `You can resend OTP in ${countdown}s`;
            if (countdown <= 0) {
                clearInterval(timerInterval);
                timerSpan.innerText = '';
                resendButton.disabled = false;
            }
        }, 1000);
    }
    
    async function resendOtp() {
        if (!voterId || !email) {
            statusDiv.innerText = "Error: Missing voter ID or email.";
            statusDiv.style.color = "red";
            return;
        }

        statusDiv.innerText = "Resending OTP...";
        resendButton.disabled = true;

        try {
            const response = await fetch('http://127.0.0.1:5000/resend_otp', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ voter_id: voterId, email: email }),
            });
            const data = await response.json();
            
            if (response.ok) {
                statusDiv.innerText = data.message;
                statusDiv.style.color = "green";
                startTimer(); // Restart the timer after a successful resend
            } else {
                statusDiv.innerText = `Error: ${data.error}`;
                statusDiv.style.color = "red";
            }
        } catch (error) {
            console.error('Error resending OTP:', error);
            statusDiv.innerText = `Network error. Could not resend OTP.`;
            statusDiv.style.color = "red";
        }
        resendButton.disabled = false;
    }

    otpForm.addEventListener('submit', (event) => {
        event.preventDefault();
        verifyButton.disabled = true;
        const otpCode = otpInput.value;
        statusDiv.innerText = "Verifying OTP...";

        fetch('http://127.0.0.1:5000/verify_otp', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ otp: otpCode }),
        })
        .then(response => response.json())
        .then(data => {
            verifyButton.disabled = false;
            if (data.message) {
                statusDiv.innerText = data.message;
                statusDiv.style.color = "green";
                clearInterval(timerInterval);
                // Redirect to a success page or main voting dashboard
                // window.location.href = '/voting_page'; 
                console.log("Frontend redirecting to:", data.redirect_url);
                window.location.href = data.redirect_url;
            } else if (data.error) {
                statusDiv.innerText = `Error: ${data.error}`;
                statusDiv.style.color = "red";
            }
        })
        .catch((error) => {
            console.error('Error:', error);
            statusDiv.innerText = `Network error. Is the server running?`;
            statusDiv.style.color = "red";
            verifyButton.disabled = false;
        });
    });

    resendButton.addEventListener('click', resendOtp);
    startTimer();
});