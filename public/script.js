// Global variables
let currentTab = 'prediction';
let modelStats = null;
let charts = {};

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    initializeApp();
    setupEventListeners();
    loadDataSummary();
    loadModelStats();
});

// Initialize application
function initializeApp() {
    console.log('🚗 Auto Price Prediction App Initialized');
    showTab('prediction');
    
    // Populate dropdowns with default data immediately
    populateDefaultDropdowns();
}

// Populate dropdowns with default data
function populateDefaultDropdowns() {
    // Default car makes
    const defaultMakes = [
        'alfa-romero', 'audi', 'bmw', 'chevrolet', 'dodge', 'honda',
        'isuzu', 'jaguar', 'mazda', 'mercedes-benz', 'mercury',
        'mitsubishi', 'nissan', 'peugot', 'plymouth', 'porsche',
        'renault', 'saab', 'subaru', 'toyota', 'volkswagen', 'volvo'
    ];
    
    populateSelect('make', defaultMakes);
}

// Setup event listeners
function setupEventListeners() {
    // Form submission
    const form = document.getElementById('predictionForm');
    if (form) {
        form.addEventListener('submit', handlePrediction);
    }

    // Tab switching
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            const tabName = e.target.closest('.tab-btn').onclick.toString().match(/showTab\('(.+?)'\)/)[1];
            showTab(tabName);
        });
    });
}

// Tab switching functionality
function showTab(tabName) {
    // Hide all tab contents
    document.querySelectorAll('.tab-content').forEach(content => {
        content.classList.remove('active');
    });

    // Remove active class from all tab buttons
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.classList.remove('active');
    });

    // Show selected tab content
    const selectedTab = document.getElementById(tabName);
    if (selectedTab) {
        selectedTab.classList.add('active');
    }

    // Add active class to selected tab button
    const selectedBtn = document.querySelector(`[onclick="showTab('${tabName}')"]`);
    if (selectedBtn) {
        selectedBtn.classList.add('active');
    }

    currentTab = tabName;

    // Initialize charts when analytics tab is shown
    if (tabName === 'analytics') {
        setTimeout(() => {
            initializeCharts();
        }, 100);
    }
}

// Load data summary for dropdowns
async function loadDataSummary() {
    try {
        const response = await fetch('/api/data-summary');
        const data = await response.json();

        if (data.makes && data.makes.length > 0) {
            populateSelect('make', data.makes);
        }

        // Update total records display
        if (data.totalRecords) {
            const totalRecordsElement = document.getElementById('totalRecords');
            if (totalRecordsElement) {
                totalRecordsElement.textContent = data.totalRecords;
            }
        }
    } catch (error) {
        console.error('Error loading data summary:', error);
        // Keep default data if API fails
    }
}

// Load model statistics
async function loadModelStats() {
    try {
        const response = await fetch('/api/stats');
        const data = await response.json();
        modelStats = data;
        
        if (data.models) {
            populateMetricsTable(data.models);
        }
    } catch (error) {
        console.error('Error loading model stats:', error);
        // Use default stats if API fails
        modelStats = {
            totalRecords: 205,
            models: {
                'XGBoost': { r2: 0.8937, adjustedR2: 0.6723, mae: 2847.32, rmse: 4521.18 },
                'Random Forest': { r2: 0.8773, adjustedR2: 0.6216, mae: 3124.56, rmse: 4892.34 },
                'KNN': { r2: 0.8730, adjustedR2: 0.6084, mae: 3287.91, rmse: 5012.67 },
                'Linear Regression': { r2: 0.7845, adjustedR2: 0.5234, mae: 4123.78, rmse: 6234.89 }
            }
        };
        populateMetricsTable(modelStats.models);
    }
}

// Populate select dropdown
function populateSelect(selectId, options) {
    const select = document.getElementById(selectId);
    if (!select) {
        console.error(`Select element with id '${selectId}' not found`);
        return;
    }

    // Clear existing options except the first one (placeholder)
    const firstOption = select.firstElementChild;
    select.innerHTML = '';
    if (firstOption && firstOption.value === '') {
        select.appendChild(firstOption);
    } else {
        // Create default placeholder option
        const placeholderOption = document.createElement('option');
        placeholderOption.value = '';
        placeholderOption.textContent = 'Select Brand';
        select.appendChild(placeholderOption);
    }

    // Add new options
    options.forEach(option => {
        const optionElement = document.createElement('option');
        optionElement.value = option;
        optionElement.textContent = option.charAt(0).toUpperCase() + option.slice(1).replace('-', ' ');
        select.appendChild(optionElement);
    });

    console.log(`Populated ${selectId} with ${options.length} options`);
}

// Handle prediction form submission
async function handlePrediction(event) {
    event.preventDefault();
    
    const form = event.target;
    const formData = new FormData(form);
    const features = {};

    // Convert form data to object
    for (let [key, value] of formData.entries()) {
        // Convert numeric fields
        const numericFields = [
            'symboling', 'engine_size', 'horsepower', 'curb_weight',
            'length', 'width', 'city_mpg', 'highway_mpg'
        ];
        
        if (numericFields.includes(key)) {
            features[key] = parseFloat(value);
        } else {
            features[key] = value;
        }
    }

    // Validate required fields
    const requiredFields = ['make', 'fuel_type', 'aspiration', 'body_style', 'drive_wheels', 'num_of_doors'];
    const missingFields = requiredFields.filter(field => !features[field]);
    
    if (missingFields.length > 0) {
        showError(`Please fill in all required fields: ${missingFields.join(', ')}`);
        return;
    }

    // Show loading state
    const predictBtn = form.querySelector('.predict-btn');
    const originalText = predictBtn.innerHTML;
    predictBtn.innerHTML = '<div class="loading"></div> Predicting...';
    predictBtn.disabled = true;

    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(features)
        });

        const result = await response.json();

        if (result.success) {
            displayPredictionResult(result);
        } else {
            showError('Prediction failed: ' + result.error);
        }
    } catch (error) {
        console.error('Prediction error:', error);
        showError('Network error occurred. Please try again.');
    } finally {
        // Reset button state
        predictBtn.innerHTML = originalText;
        predictBtn.disabled = false;
    }
}

// Display prediction result
function displayPredictionResult(result) {
    const resultDiv = document.getElementById('predictionResult');
    const priceElement = document.getElementById('predictedPrice');
    const confidenceElement = document.getElementById('confidence');

    if (resultDiv && priceElement && confidenceElement) {
        priceElement.textContent = result.prediction.toLocaleString();
        confidenceElement.textContent = Math.round(result.confidence * 100) + '%';
        
        resultDiv.classList.remove('hidden');
        
        // Smooth scroll to result
        resultDiv.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        
        // Store result for download
        window.lastPrediction = result;
    }
}

// Show error message
function showError(message) {
    // Create a better error display
    const errorDiv = document.createElement('div');
    errorDiv.className = 'error-message';
    errorDiv.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        background: #e74c3c;
        color: white;
        padding: 15px 20px;
        border-radius: 5px;
        z-index: 1000;
        box-shadow: 0 2px 10px rgba(0,0,0,0.2);
    `;
    errorDiv.textContent = message;
    
    document.body.appendChild(errorDiv);
    
    // Remove after 5 seconds
    setTimeout(() => {
        if (errorDiv.parentNode) {
            errorDiv.parentNode.removeChild(errorDiv);
        }
    }, 5000);
}

// Reset form
function resetForm() {
    const form = document.getElementById('predictionForm');
    if (form) {
        form.reset();
        
        // Hide prediction result
        const resultDiv = document.getElementById('predictionResult');
        if (resultDiv) {
            resultDiv.classList.add('hidden');
        }
    }
}

// Download prediction result
function downloadResult() {
    if (!window.lastPrediction) {
        showError('No prediction result to download');
        return;
    }

    const result = window.lastPrediction;
    const csvContent = [
        'Feature,Value',
        ...Object.entries(result.features).map(([key, value]) => `${key},${value}`),
        '',
        'Prediction Results',
        `Predicted Price,$${result.prediction.toLocaleString()}`,
        `Confidence,${Math.round(result.confidence * 100)}%`,
        `Model,${result.model}`,
        `Timestamp,${new Date().toISOString()}`
    ].join('\n');

    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `car_price_prediction_${Date.now()}.csv`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    window.URL.revokeObjectURL(url);
}

// Initialize charts
function initializeCharts() {
    if (!modelStats || !modelStats.models) return;

    // Model Performance Chart
    initializeModelChart();
    
    // Feature Importance Chart
    initializeFeatureChart();
}

// Initialize model performance chart
function initializeModelChart() {
    const ctx = document.getElementById('modelChart');
    if (!ctx) return;

    // Destroy existing chart if it exists
    if (charts.modelChart) {
        charts.modelChart.destroy();
    }

    const models = Object.keys(modelStats.models);
    const r2Scores = models.map(model => modelStats.models[model].r2);
    const adjustedR2Scores = models.map(model => modelStats.models[model].adjustedR2);

    charts.modelChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: models,
            datasets: [
                {
                    label: 'R² Score',
                    data: r2Scores,
                    backgroundColor: 'rgba(52, 152, 219, 0.8)',
                    borderColor: 'rgba(52, 152, 219, 1)',
                    borderWidth: 2
                },
                {
                    label: 'Adjusted R²',
                    data: adjustedR2Scores,
                    backgroundColor: 'rgba(46, 204, 113, 0.8)',
                    borderColor: 'rgba(46, 204, 113, 1)',
                    borderWidth: 2
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true,
                    max: 1,
                    ticks: {
                        callback: function(value) {
                            return (value * 100).toFixed(1) + '%';
                        }
                    }
                }
            },
            plugins: {
                legend: {
                    position: 'top',
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return context.dataset.label + ': ' + (context.parsed.y * 100).toFixed(2) + '%';
                        }
                    }
                }
            }
        }
    });
}

// Initialize feature importance chart
function initializeFeatureChart() {
    const ctx = document.getElementById('featureChart');
    if (!ctx) return;

    // Destroy existing chart if it exists
    if (charts.featureChart) {
        charts.featureChart.destroy();
    }

    // Mock feature importance data (in a real app, this would come from the model)
    const features = [
        'Engine Size',
        'Horsepower',
        'Curb Weight',
        'Length',
        'Brand',
        'Fuel Type',
        'Highway MPG',
        'City MPG',
        'Width',
        'Aspiration'
    ];

    const importance = [0.25, 0.20, 0.15, 0.10, 0.08, 0.07, 0.06, 0.05, 0.03, 0.01];

    charts.featureChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: features,
            datasets: [{
                label: 'Importance',
                data: importance,
                backgroundColor: 'rgba(155, 89, 182, 0.8)',
                borderColor: 'rgba(155, 89, 182, 1)',
                borderWidth: 2
            }]
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    beginAtZero: true,
                    max: 0.3,
                    ticks: {
                        callback: function(value) {
                            return (value * 100).toFixed(1) + '%';
                        }
                    }
                }
            },
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return 'Importance: ' + (context.parsed.x * 100).toFixed(2) + '%';
                        }
                    }
                }
            }
        }
    });
}

// Populate metrics table
function populateMetricsTable(models) {
    const tbody = document.getElementById('metricsTableBody');
    if (!tbody) return;

    tbody.innerHTML = '';

    Object.entries(models).forEach(([modelName, metrics]) => {
        const row = document.createElement('tr');
        row.innerHTML = `
            <td><strong>${modelName}</strong></td>
            <td>${(metrics.r2 * 100).toFixed(2)}%</td>
            <td>${(metrics.adjustedR2 * 100).toFixed(2)}%</td>
            <td>$${metrics.mae.toLocaleString()}</td>
            <td>$${metrics.rmse.toLocaleString()}</td>
        `;
        
        // Highlight best model
        if (modelName === 'XGBoost') {
            row.style.backgroundColor = '#e8f5e8';
            row.style.fontWeight = 'bold';
        }
        
        tbody.appendChild(row);
    });
}

// Utility functions
function formatNumber(num, decimals = 2) {
    return num.toFixed(decimals).replace(/\B(?=(\d{3})+(?!\d))/g, ",");
}

function formatCurrency(amount) {
    return new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: 'USD',
        minimumFractionDigits: 0,
        maximumFractionDigits: 0,
    }).format(amount);
}

// Error handling
window.addEventListener('error', function(event) {
    console.error('JavaScript Error:', event.error);
});

// Handle unhandled promise rejections
window.addEventListener('unhandledrejection', function(event) {
    console.error('Unhandled Promise Rejection:', event.reason);
});