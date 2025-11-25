document.addEventListener('DOMContentLoaded', function() {
    const slides = document.querySelectorAll('.hero .slide');
    const progressBar = document.getElementById('hero-progress-bar');
    const controlBtn = document.getElementById('slide-control-btn');

    if (slides.length > 0 && progressBar && controlBtn) {
        let currentSlide = 0;
        let isPaused = false;
        const slideDuration = 7000; // 7 seconds per slide
        let intervalId;

        function showSlide(index) {
            // Hide all slides
            slides.forEach(slide => slide.classList.remove('active'));
            // Show the correct slide
            slides[index].classList.add('active');
        }

        function nextSlide() {
            // Move to the next slide, or loop back to the first
            currentSlide = (currentSlide + 1) % slides.length;
            showSlide(currentSlide);
        }

        function updateProgress() {
            // Reset the progress bar for the new slide
            progressBar.style.transition = 'none'; // Remove transition to reset instantly
            progressBar.style.width = '0%';
            
            // Force a DOM reflow to ensure the reset is applied before adding the transition back
            progressBar.getBoundingClientRect(); 

            // Add the transition back for the smooth progress animation
            progressBar.style.transition = `width ${slideDuration}ms linear`;
            progressBar.style.width = '100%';
        }
        
        function startSlideshow() {
            // Immediately start the progress for the current slide
            updateProgress();
            // Set an interval to switch to the next slide after the duration
            intervalId = setInterval(function() {
                nextSlide();
                updateProgress();
            }, slideDuration);
        }

        function pauseSlideshow() {
            isPaused = true;
            clearInterval(intervalId); // Stop the slide switching
            // Get the current width and freeze it
            const currentWidth = window.getComputedStyle(progressBar).getPropertyValue('width');
            progressBar.style.transition = 'none';
            progressBar.style.width = currentWidth;
            // Update button UI
            controlBtn.innerHTML = '►'; // Play icon
            controlBtn.setAttribute('aria-label', 'Play slideshow');
        }
        
        function resumeSlideshow() {
            isPaused = false;
            // Calculate remaining time
            const currentWidthPercent = parseFloat(progressBar.style.width) / progressBar.parentElement.offsetWidth * 100;
            const remainingTime = (1 - (currentWidthPercent / 100)) * slideDuration;

            // Animate the rest of the progress
            progressBar.style.transition = `width ${remainingTime}ms linear`;
            progressBar.style.width = '100%';
            
            // After the remaining time, switch to the next slide and restart the full loop
            setTimeout(() => {
                if (!isPaused) { // Check if it hasn't been paused again
                    nextSlide();
                    startSlideshow();
                }
            }, remainingTime);
            // Update button UI
            controlBtn.innerHTML = '❚❚'; // Pause icon
            controlBtn.setAttribute('aria-label', 'Pause slideshow');
        }

        // Add click event to the control button
        controlBtn.addEventListener('click', function() {
            if (isPaused) {
                resumeSlideshow();
            } else {
                pauseSlideshow();
            }
        });

        // Initialize the slideshow
        showSlide(currentSlide);
        startSlideshow();
    }
});
