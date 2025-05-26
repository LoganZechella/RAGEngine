// Search Progress Enhancement
document.addEventListener('DOMContentLoaded', function() {
    // Existing file upload code remains unchanged...
    const fileInput = document.getElementById("file-input");
    const dropZone = fileInput?.closest(".border-dashed");
    
    if (dropZone) {
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
    }

    // Enhanced search functionality
    const searchForm = document.getElementById('search-form');
    const searchButton = document.getElementById('search-button');
    
    if (searchForm) {
        searchForm.addEventListener('htmx:beforeRequest', function() {
            // Disable form during search
            const inputs = searchForm.querySelectorAll('input, select, button');
            inputs.forEach(input => input.disabled = true);
            
            // Update button state
            if (searchButton) {
                const icon = searchButton.querySelector('.search-icon');
                if (icon) icon.textContent = '⏳';
                searchButton.classList.add('opacity-75');
            }
        });

        searchForm.addEventListener('htmx:afterRequest', function() {
            // Re-enable form after request
            const inputs = searchForm.querySelectorAll('input, select, button');
            inputs.forEach(input => input.disabled = false);
            
            // Reset button state
            if (searchButton) {
                const icon = searchButton.querySelector('.search-icon');
                if (icon) icon.textContent = '🔍';
                searchButton.classList.remove('opacity-75');
            }
        });
    }

    // Keyboard shortcuts
    document.addEventListener('keydown', function(e) {
        // Ctrl/Cmd + K to focus search
        if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
            e.preventDefault();
            const searchInput = document.getElementById('search-query');
            if (searchInput) {
                searchInput.focus();
                searchInput.select();
            }
        }
        
        // Escape to cancel search
        if (e.key === 'Escape') {
            const cancelButton = document.getElementById('cancel-button');
            if (cancelButton && !cancelButton.classList.contains('hidden')) {
                cancelButton.click();
            }
        }
    });

    // Search result enhancement
    document.addEventListener('htmx:afterSettle', function(e) {
        if (e.detail.target.id === 'search-container') {
            // Smooth scroll to results
            e.detail.target.scrollIntoView({ behavior: 'smooth', block: 'start' });
            
            // Add fade-in animation to results
            const results = e.detail.target.querySelectorAll('.htmx-added');
            results.forEach((result, index) => {
                result.style.animationDelay = `${index * 0.1}s`;
                result.classList.add('animate-fadeIn');
            });
        }
    });
});

// Quick search functionality
function quickSearch(query) {
    const searchInput = document.getElementById('search-query');
    const searchForm = document.getElementById('search-form');
    
    if (searchInput && searchForm) {
        searchInput.value = query;
        
        // Trigger HTMX request
        htmx.trigger(searchForm, 'submit');
        
        // Focus and highlight the input
        setTimeout(() => {
            searchInput.focus();
            searchInput.select();
        }, 100);
    }
}

// Enhanced error handling
document.addEventListener('htmx:responseError', function(e) {
    console.error('HTMX Request Error:', e.detail);
    showToast(`Request failed: ${e.detail.xhr.statusText}`, 'error');
    
    // Re-enable any disabled forms
    document.querySelectorAll('form').forEach(form => {
        const inputs = form.querySelectorAll('input, select, button');
        inputs.forEach(input => input.disabled = false);
    });
});

// Progress bar animations
function animateProgressBar(element, targetWidth, duration = 500) {
    if (!element) return;
    
    const startWidth = parseFloat(element.style.width) || 0;
    const distance = targetWidth - startWidth;
    const startTime = performance.now();
    
    function animate(currentTime) {
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / duration, 1);
        
        // Easing function for smooth animation
        const easeOut = 1 - Math.pow(1 - progress, 3);
        const currentWidth = startWidth + distance * easeOut;
        
        element.style.width = currentWidth + '%';
        
        if (progress < 1) {
            requestAnimationFrame(animate);
        }
    }
    
    requestAnimationFrame(animate);
}

// Toast notification function (placeholder)
function showToast(message, type = 'info') {
    console.log(`Toast [${type}]: ${message}`);
    // This could be enhanced with actual toast UI in the future
}
