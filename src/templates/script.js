document.addEventListener("DOMContentLoaded", () => {
    // Get all the necessary elements from the page
    const wqiForm = document.getElementById("wqi-form");
    const predictBtn = document.getElementById("predict-btn");
    const resultContainer = document.getElementById("result-container");
    const resultText = document.getElementById("result-text");
    const loader = document.getElementById("loader");

    // Listen for the form to be submitted
    wqiForm.addEventListener("submit", async (event) => {
        // Prevent the form from refreshing the page
        event.preventDefault();

        // Show loading state and disable button
        resultContainer.classList.add("hidden");
        loader.classList.remove("hidden");
        predictBtn.disabled = true;
        predictBtn.innerText = "Predicting...";

        // 1. Get the form data
        const formData = new FormData(wqiForm);
        
        // 2. Convert form data to the exact JSON object the API expects
        const data = {};
        formData.forEach((value, key) => {
            // 'State Name' is the only required text field
            // For all other fields, if they are empty, send 'null'
            // Otherwise, send the number value
            if (key !== 'State Name') {
                data[key] = value === '' ? null : parseFloat(value);
            } else {
                data[key] = value;
            }
        });
        
        // 3. Call the FastAPI endpoint
        try {
            const response = await fetch("http://127.0.0.1:8000/predict", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                    "Accept": "application/json"
                },
                body: JSON.stringify(data)
            });

            // Handle bad responses (like 404, 500)
            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || `HTTP error! Status: ${response.status}`);
            }

            // Handle successful responses
            const result = await response.json();
            
            // 4. Display the result
            displayResult(result.prediction, "Success");

        } catch (error) {
            // 5. Display any errors
            console.error("Error during prediction:", error);
            displayResult(`Error: ${error.message}`, "Error");
        } finally {
            // 6. Restore the UI
            loader.classList.add("hidden");
            predictBtn.disabled = false;
            predictBtn.innerText = "Predict";
        }
    });

    function displayResult(message, type) {
        resultText.innerText = message;
        
        // Remove old result classes
        resultContainer.classList.remove("hidden", "result-Success", "result-Error");
        
        // Add the new one
        resultContainer.classList.add(`result-${type}`);
    }
});