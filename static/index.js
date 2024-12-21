const form = document.getElementById('uploadForm');
const imageInput = document.getElementById('imageInput');
const imagePreview = document.getElementById('imagePreview');
const result = document.getElementById('result');
const error = document.getElementById('error');

imageInput.addEventListener('change', function() {
    const file = this.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = function(e) {
            imagePreview.src = e.target.result;
            imagePreview.style.display = 'block';
        }
        reader.readAsDataURL(file);
    }
});

form.addEventListener('submit', async function(e) {
    e.preventDefault();
    
    const formData = new FormData();
    formData.append('file', imageInput.files[0]);
    
    try {
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (response.ok) {
            result.textContent = 'Caption: ' + data.caption;
            result.style.display = 'block';
            error.style.display = 'none';
        } else {
            error.textContent = data.error;
            error.style.display = 'block';
            result.style.display = 'none';
        }
    } catch (err) {
        error.textContent = 'An error occurred while processing your request.';
        error.style.display = 'block';
        result.style.display = 'none';
    }
});
