// Enhanced file upload with drag and drop
document.addEventListener("DOMContentLoaded", function() {
    const fileInput = document.getElementById("file-input");
    const dropZone = fileInput?.closest(".border-dashed");
    
    if (!dropZone) return;
    
    // Drag and drop functionality
    ["dragenter", "dragover", "dragleave", "drop"].forEach(eventName => {
        dropZone.addEventListener(eventName, preventDefaults, false);
    });
    
    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }
    
    ["dragenter", "dragover"].forEach(eventName => {
        dropZone.addEventListener(eventName, highlight, false);
    });
    
    ["dragleave", "drop"].forEach(eventName => {
        dropZone.addEventListener(eventName, unhighlight, false);
    });
    
    function highlight() {
        dropZone.classList.add("border-blue-500", "bg-blue-900", "bg-opacity-20");
    }
    
    function unhighlight() {
        dropZone.classList.remove("border-blue-500", "bg-blue-900", "bg-opacity-20");
    }
    
    dropZone.addEventListener("drop", handleDrop, false);
    
    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        fileInput.files = files;
        
        // Update label to show selected files
        const label = dropZone.querySelector("label p");
        if (label && files.length > 0) {
            label.textContent = `${files.length} file(s) selected`;
        }
    }
});
