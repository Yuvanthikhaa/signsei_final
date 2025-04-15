document.addEventListener('DOMContentLoaded', function() {
    // Get reference to prediction display
    const predictionValue = document.getElementById('prediction-value');
    
    // Function to fetch current prediction from server
    function updatePrediction() {
        fetch('/get_prediction')
            .then(response => response.json())
            .then(data => {
                // Update prediction display
                predictionValue.textContent = data.prediction;
                
                // Add visual feedback - flash effect
                predictionValue.classList.add('updated');
                setTimeout(() => {
                    predictionValue.classList.remove('updated');
                }, 300);
            })
            .catch(error => console.error('Error fetching prediction:', error));
    }
    
    // Update prediction every second
    setInterval(updatePrediction, 1000);
    
    // Cleanup when user leaves page
    window.addEventListener('beforeunload', function() {
        // Send request to release camera
        fetch('/shutdown', {
            method: 'POST',
        }).catch(err => console.error('Error shutting down:', err));
    });
});