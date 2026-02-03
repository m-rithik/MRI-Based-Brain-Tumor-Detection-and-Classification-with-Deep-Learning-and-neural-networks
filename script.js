/**
 * Brain Tumor Detection System - Frontend JavaScript
 * Handles image upload, API communication, and results display
 */

// Configuration
const API_URL = 'http://localhost:8000';
const MAX_FILE_SIZE = 16 * 1024 * 1024; // 16MB
const ALLOWED_TYPES = ['image/png', 'image/jpeg', 'image/jpg'];

// DOM Elements
let uploadArea, fileInput, uploadContent, imagePreview, previewImg, removeBtn;
let analyzeBtn, resultsSection, loadingOverlay, newAnalysisBtn;
let classificationResult, confidenceValue, progressFill, statusIndicator;
let noTumorProb, tumorProb, loadingText;

// State
let selectedFile = null;

/**
 * Initialize the application
 */
document.addEventListener('DOMContentLoaded', () => {
    initializeElements();
    attachEventListeners();
    console.log('Brain Tumor Detection System initialized');
});

/**
 * Initialize DOM element references
 */
function initializeElements() {
    // Upload elements
    uploadArea = document.getElementById('uploadArea');
    fileInput = document.getElementById('fileInput');
    uploadContent = document.getElementById('uploadContent');
    imagePreview = document.getElementById('imagePreview');
    previewImg = document.getElementById('previewImg');
    removeBtn = document.getElementById('removeBtn');
    analyzeBtn = document.getElementById('analyzeBtn');

    // Results elements
    resultsSection = document.getElementById('resultsSection');
    classificationResult = document.getElementById('classificationResult');
    confidenceValue = document.getElementById('confidenceValue');
    progressFill = document.getElementById('progressFill');
    statusIndicator = document.getElementById('statusIndicator');
    noTumorProb = document.getElementById('noTumorProb');
    tumorProb = document.getElementById('tumorProb');
    newAnalysisBtn = document.getElementById('newAnalysisBtn');

    // Loading elements
    loadingOverlay = document.getElementById('loadingOverlay');
    loadingText = document.getElementById('loadingText');
}

/**
 * Attach event listeners to interactive elements
 */
function attachEventListeners() {
    // Upload area click
    uploadArea.addEventListener('click', () => {
        if (!selectedFile) {
            fileInput.click();
        }
    });

    // File input change
    fileInput.addEventListener('change', handleFileSelect);

    // Drag and drop
    uploadArea.addEventListener('dragover', handleDragOver);
    uploadArea.addEventListener('dragleave', handleDragLeave);
    uploadArea.addEventListener('drop', handleDrop);

    // Remove button
    removeBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        resetUpload();
    });

    // Analyze button
    analyzeBtn.addEventListener('click', analyzeScan);

    // New analysis button
    newAnalysisBtn.addEventListener('click', startNewAnalysis);
}

/**
 * Handle file selection from input
 */
function handleFileSelect(e) {
    const file = e.target.files[0];
    if (file) {
        validateAndPreviewFile(file);
    }
}

/**
 * Handle drag over event
 */
function handleDragOver(e) {
    e.preventDefault();
    e.stopPropagation();
    if (!selectedFile) {
        uploadArea.classList.add('drag-over');
    }
}

/**
 * Handle drag leave event
 */
function handleDragLeave(e) {
    e.preventDefault();
    e.stopPropagation();
    uploadArea.classList.remove('drag-over');
}

/**
 * Handle file drop event
 */
function handleDrop(e) {
    e.preventDefault();
    e.stopPropagation();
    uploadArea.classList.remove('drag-over');

    if (!selectedFile) {
        const file = e.dataTransfer.files[0];
        if (file) {
            validateAndPreviewFile(file);
        }
    }
}

/**
 * Validate file and show preview
 */
function validateAndPreviewFile(file) {
    // Check file type
    if (!ALLOWED_TYPES.includes(file.type)) {
        showError('Invalid file type. Please upload PNG, JPG, or JPEG images.');
        return;
    }

    // Check file size
    if (file.size > MAX_FILE_SIZE) {
        showError('File size exceeds 16MB. Please upload a smaller image.');
        return;
    }

    // Store file and show preview
    selectedFile = file;
    displayImagePreview(file);
}

/**
 * Display image preview
 */
function displayImagePreview(file) {
    const reader = new FileReader();

    reader.onload = (e) => {
        previewImg.src = e.target.result;
        uploadContent.style.display = 'none';
        imagePreview.style.display = 'block';
        analyzeBtn.disabled = false;
    };

    reader.readAsDataURL(file);
}

/**
 * Reset upload state
 */
function resetUpload() {
    selectedFile = null;
    fileInput.value = '';
    previewImg.src = '';
    uploadContent.style.display = 'flex';
    imagePreview.style.display = 'none';
    analyzeBtn.disabled = true;
}

/**
 * Analyze the uploaded scan
 */
async function analyzeScan() {
    if (!selectedFile) {
        showError('Please upload an image first.');
        return;
    }

    // Show loading overlay
    showLoading();

    // Simulate loading steps
    simulateLoadingSteps();

    try {
        // Create form data
        const formData = new FormData();
        formData.append('image', selectedFile);

        // Send to API
        const response = await fetch(`${API_URL}/api/predict`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();

        if (data.success) {
            // Hide loading and show results
            hideLoading();
            displayResults(data.result);
        } else {
            throw new Error(data.error || 'Prediction failed');
        }
    } catch (error) {
        hideLoading();
        showError(`Analysis failed: ${error.message}`);
        console.error('Error:', error);
    }
}

/**
 * Simulate loading steps animation
 */
function simulateLoadingSteps() {
    const steps = document.querySelectorAll('.step');
    const messages = [
        'Preprocessing image...',
        'Extracting features...',
        'Running classification...'
    ];

    steps.forEach((step, index) => {
        setTimeout(() => {
            // Remove active from all steps
            steps.forEach(s => s.classList.remove('active'));
            // Add active to current step
            step.classList.add('active');
            // Update loading text
            loadingText.textContent = messages[index];
        }, index * 1000);
    });
}

/**
 * Display analysis results
 */
function displayResults(result) {
    // Update classification
    classificationResult.textContent = result.classification;
    classificationResult.className = 'result-value ' +
        (result.tumor_detected ? 'tumor' : 'no-tumor');

    // Update status indicator
    statusIndicator.className = 'status-indicator ' +
        (result.tumor_detected ? 'negative' : 'positive');

    // Update confidence
    confidenceValue.textContent = `${result.confidence}%`;
    progressFill.style.width = `${result.confidence}%`;

    // Update probabilities
    noTumorProb.textContent = `${result.probabilities.no_tumor}%`;
    tumorProb.textContent = `${result.probabilities.tumor}%`;

    // Show results section with animation
    resultsSection.style.display = 'block';
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

/**
 * Start a new analysis
 */
function startNewAnalysis() {
    resultsSection.style.display = 'none';
    resetUpload();
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

/**
 * Show loading overlay
 */
function showLoading() {
    loadingOverlay.style.display = 'flex';
    loadingText.textContent = 'Analyzing image with deep neural network...';

    // Reset steps
    const steps = document.querySelectorAll('.step');
    steps.forEach(s => s.classList.remove('active'));
    steps[0].classList.add('active');
}

/**
 * Hide loading overlay
 */
function hideLoading() {
    loadingOverlay.style.display = 'none';
}

/**
 * Show error message
 */
function showError(message) {
    alert(message);
}

/**
 * Check API health on page load
 */
async function checkAPIHealth() {
    try {
        const response = await fetch(`${API_URL}/api/health`);
        const data = await response.json();
        console.log('API Health:', data);
    } catch (error) {
        console.warn('API connection failed. Make sure backend is running.');
    }
}

// Check API health when page loads
checkAPIHealth();
