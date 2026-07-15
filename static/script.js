       console.log("SCRIPT LOADED");
        const dragZone = document.getElementById('drag-zone');
        const fileInput = document.getElementById('file-input');
        const previewWrapper = document.getElementById('preview-wrapper');
        const previewImage = document.getElementById('preview-image');
        const removeFileBtn = document.getElementById('remove-file-btn');
        const submitBtn = document.getElementById('submit-btn');
        const detectForm = document.getElementById('detect-form');
        const loadingOverlay = document.getElementById('loading-overlay');
        const resultCard = document.getElementById('result-card');
        const orSeparator = document.getElementById('or-separator');
        const cameraActionContainer = document.getElementById('camera-action-container');
        const cameraTriggerBtn = document.getElementById('camera-trigger-btn');
        const cameraInput = document.getElementById('camera-input');
        const modeButtons = document.querySelectorAll('.mode-btn');
        const imageTypeInput = document.getElementById('image_type');
        const modeLabel = document.getElementById('mode-label');
        const resultModal = document.getElementById('result-modal');
        const modalCloseBtn = document.getElementById('modal-close-btn');

        // Drag events handling
        if (dragZone) {
            ['dragenter', 'dragover'].forEach(eventName => {
                dragZone.addEventListener(eventName, (e) => {
                    e.preventDefault();
                    dragZone.classList.add('dragover');
                }, false);
            });

            ['dragleave', 'drop'].forEach(eventName => {
                dragZone.addEventListener(eventName, (e) => {
                    e.preventDefault();
                    dragZone.classList.remove('dragover');
                }, false);
            });
        }

        // Drop file handling
        if (dragZone) {
            dragZone.addEventListener('drop', (e) => {
                const dt = e.dataTransfer;
                const files = dt.files;
                if (files.length) {
                    fileInput.files = files;
                    handleFiles(files[0]);
                }
            });
        }

        // Click on dragzone opens standard picker
        if (fileInput) {
            fileInput.addEventListener('change', (e) => {
                if (fileInput.files.length) {
                    handleFiles(fileInput.files[0]);
                }
            });
        }

        function handleFiles(file) {
            if (!file.type.startsWith('image/')) {
                alert('Unsupported file type. Please upload an image file.');
                return;
            }
            const reader = new FileReader();
            reader.readAsDataURL(file);
            reader.onload = (e) => {
                const imgDataUrl = e.target.result;
                previewImage.src = imgDataUrl;
                previewWrapper.style.display = 'flex';
                dragZone.style.display = 'none';
                if (orSeparator) orSeparator.style.display = 'none';
                if (cameraActionContainer) cameraActionContainer.style.display = 'none';
                submitBtn.style.display = 'inline-flex';

                // Real-time Quality Checks
                const img = new Image();
                img.src = imgDataUrl;
                img.onload = () => {
                    analyzeImageQuality(img);
                };
            };
        }

        function analyzeImageQuality(img) {
            const canvas = document.createElement('canvas');
            const ctx = canvas.getContext('2d');
            const width = 120;
            const height = 120;
            canvas.width = width;
            canvas.height = height;
            ctx.drawImage(img, 0, 0, width, height);

            try {
                const imgData = ctx.getImageData(0, 0, width, height);
                const data = imgData.data;

                let totalBrightness = 0;
                const grayscale = new Float32Array(width * height);

                for (let i = 0; i < data.length; i += 4) {
                    const r = data[i];
                    const g = data[i + 1];
                    const b = data[i + 2];
                    const lum = 0.299 * r + 0.587 * g + 0.114 * b;
                    totalBrightness += lum;
                    grayscale[i / 4] = lum;
                }

                const avgBrightness = totalBrightness / (width * height);

                let diffSum = 0;
                let diffSqSum = 0;
                let count = 0;

                for (let y = 1; y < height - 1; y++) {
                    for (let x = 1; x < width - 1; x++) {
                        const idx = y * width + x;
                        const center = grayscale[idx];
                        const diffX = center - grayscale[idx + 1];
                        const diffY = center - grayscale[idx + width];
                        const grad = Math.sqrt(diffX * diffX + diffY * diffY);
                        diffSum += grad;
                        diffSqSum += grad * grad;
                        count++;
                    }
                }

                const meanGrad = diffSum / count;
                const varianceGrad = (diffSqSum / count) - (meanGrad * meanGrad);

                const warningDiv = document.getElementById('quality-warning');
                const warningMsg = document.getElementById('quality-msg');

                let warnings = [];

                if (avgBrightness < 35) {
                    warnings.push("image is too dark (black/underexposed). please clear image uploaded.");
                } else if (avgBrightness > 225) {
                    warnings.push("image is over-lit (extremely bright/overexposed). please clear image uploaded.");
                }

                if (varianceGrad < 12) {
                    warnings.push("image is blurry. please clear image uploaded.");
                }

                if (warnings.length > 0) {
                    if (warningDiv && warningMsg) {
                        warningMsg.innerHTML = warnings.join("<br>");
                        warningDiv.style.display = 'block';
                    }
                } else {
                    if (warningDiv) {
                        warningDiv.style.display = 'none';
                    }
                }
            } catch (err) {
                console.error("Error checking image quality:", err);
            }
        }

        // Remove Selected File
        if (removeFileBtn) {
            removeFileBtn.addEventListener('click', (e) => {
                e.stopPropagation();
                if (fileInput) fileInput.value = '';
                if (cameraInput) cameraInput.value = '';
                if (previewImage) previewImage.src = '#';
                if (previewWrapper) previewWrapper.style.display = 'none';
                if (dragZone) dragZone.style.display = 'flex';
                if (orSeparator) orSeparator.style.display = 'flex';
                if (cameraActionContainer) cameraActionContainer.style.display = 'flex';
                if (submitBtn) submitBtn.style.display = 'none';
                const warningDiv = document.getElementById('quality-warning');
                if (warningDiv) warningDiv.style.display = 'none';
            });
        }

        // Form Submit triggers premium Loading Overlay and validates image quality
        if (detectForm) {
            detectForm.addEventListener('submit', (e) => {
                const warningDiv = document.getElementById('quality-warning');
                if (warningDiv && warningDiv.style.display === 'block') {
                    e.preventDefault();
                    alert("Error: Please upload a clear image (not blurry, not too dark, and not over-lit).");
                    return;
                }
                if (loadingOverlay) loadingOverlay.classList.add('active');
            });
        }

        // Automatically animate confidence bar when results are loaded
        if (modeButtons.length && imageTypeInput && modeLabel) {
            modeButtons.forEach((button) => {
                button.addEventListener('click', () => {
                    modeButtons.forEach((btn) => btn.classList.remove('active'));
                    button.classList.add('active');
                    const selectedMode = button.getAttribute('data-mode');
                    imageTypeInput.value = selectedMode;
                    modeLabel.textContent = button.textContent;
                });
            });
        }

        if (modalCloseBtn && resultModal) {
            modalCloseBtn.addEventListener('click', () => {
                resultModal.classList.remove('open');
            });

            resultModal.addEventListener('click', (event) => {
                if (event.target === resultModal) {
                    resultModal.classList.remove('open');
                }
            });
        }

        window.addEventListener('DOMContentLoaded', () => {
            if (resultCard) {
                resultCard.scrollIntoView({ behavior: 'smooth' });
            }

            const confBars = document.querySelectorAll('.confidence-bar');
            if (confBars.length) {
                confBars.forEach((confBar) => {
                    const targetWidth = parseFloat(confBar.getAttribute('data-target-width')) || 0;
                    const label = confBar.querySelector('.confidence-text');
                    let currentVal = 0;
                    const duration = 1000;
                    const increment = targetWidth / (duration / 16);

                    setTimeout(() => {
                        confBar.style.width = targetWidth + '%';
                        const interval = setInterval(() => {
                            currentVal += increment;
                            if (currentVal >= targetWidth) {
                                label.innerText = targetWidth.toFixed(2) + '%';
                                clearInterval(interval);
                            } else {
                                label.innerText = currentVal.toFixed(1) + '%';
                            }
                        }, 16);
                    }, 300);
                });
            }

            // Modal popup disabled per request
            /*
            if (resultModal && resultModal.dataset.open === 'true') {
                resultModal.classList.add('open');
            }
            */
        });
   