// static/js/script.js - BITCOIN THEME ENHANCED VERSION

// Global variables for charts
let priceChart, sentimentChart, performanceChart, featureChart, confidenceChart;
let predictionHistory = [];

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    initializeCharts();
    checkStatus();
    loadDashboardData();
    updateDataFreshness();
});

// Tab switching function
function switchTab(tabName) {
    // Hide all tab contents
    document.querySelectorAll('.tab-content').forEach(tab => {
        tab.classList.remove('active');
    });
    
    // Remove active class from all tabs
    document.querySelectorAll('.tab').forEach(tab => {
        tab.classList.remove('active');
    });
    
    // Show selected tab content
    document.getElementById(tabName).classList.add('active');
    
    // Add active class to clicked tab
    event.currentTarget.classList.add('active');
    
    // Load data for specific tabs
    if (tabName === 'analytics') {
        loadAnalyticsData();
    } else if (tabName === 'history') {
        loadHistoryData();
    }
}

// Initialize all charts with Bitcoin theme
function initializeCharts() {
    // Price History Chart
    const priceCtx = document.getElementById('priceChart').getContext('2d');
    priceChart = new Chart(priceCtx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Bitcoin Price (USD)',
                data: [],
                borderColor: '#F7931A',
                backgroundColor: 'rgba(247, 147, 26, 0.1)',
                borderWidth: 3,
                fill: true,
                tension: 0.4,
                pointBackgroundColor: '#F7931A',
                pointBorderColor: '#FFFFFF',
                pointBorderWidth: 2,
                pointRadius: 4
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: {
                    position: 'top',
                },
                title: {
                    display: true,
                    color: '#F7931A'
                }
            },
            scales: {
                y: {
                    beginAtZero: false,
                    grid: {
                        color: 'rgba(0,0,0,0.1)'
                    }
                },
                x: {
                    grid: {
                        color: 'rgba(0,0,0,0.1)'
                    }
                }
            }
        }
    });
    
    // Sentiment Chart
    const sentimentCtx = document.getElementById('sentimentChart').getContext('2d');
    sentimentChart = new Chart(sentimentCtx, {
        type: 'bar',
        data: {
            labels: ['Positive', 'Neutral', 'Negative'],
            datasets: [{
                label: 'Wikipedia Sentiment',
                data: [0, 0, 0],
                backgroundColor: [
                    'rgba(40, 167, 69, 0.8)',
                    'rgba(108, 117, 125, 0.8)',
                    'rgba(247, 147, 26, 0.8)'
                ],
                borderColor: [
                    'rgb(40, 167, 69)',
                    'rgb(108, 117, 125)',
                    'rgb(247, 147, 26)'
                ],
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: {
                    position: 'top',
                }
            }
        }
    });
    
    // Performance Chart
    const performanceCtx = document.getElementById('performanceChart').getContext('2d');
    performanceChart = new Chart(performanceCtx, {
        type: 'doughnut',
        data: {
            labels: ['Correct Predictions', 'Incorrect Predictions'],
            datasets: [{
                data: [0, 0],
                backgroundColor: [
                    'rgba(40, 167, 69, 0.8)',
                    'rgba(247, 147, 26, 0.8)'
                ],
                borderColor: [
                    'rgb(40, 167, 69)',
                    'rgb(247, 147, 26)'
                ],
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: {
                    position: 'top',
                }
            }
        }
    });
    
    // Feature Importance Chart
    const featureCtx = document.getElementById('featureChart').getContext('2d');
    featureChart = new Chart(featureCtx, {
        type: 'bar',
        data: {
            labels: [],
            datasets: [{
                label: 'Feature Importance',
                data: [],
                backgroundColor: 'rgba(247, 147, 26, 0.8)',
                borderColor: 'rgb(247, 147, 26)',
                borderWidth: 1
            }]
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            plugins: {
                legend: {
                    position: 'top',
                }
            }
        }
    });
    
    // Confidence Distribution Chart
    const confidenceCtx = document.getElementById('confidenceChart').getContext('2d');
    confidenceChart = new Chart(confidenceCtx, {
        type: 'polarArea',
        data: {
            labels: ['High (70-100%)', 'Medium (50-70%)', 'Low (0-50%)'],
            datasets: [{
                data: [0, 0, 0],
                backgroundColor: [
                    'rgba(40, 167, 69, 0.8)',
                    'rgba(247, 147, 26, 0.8)',
                    'rgba(220, 53, 69, 0.8)'
                ],
                borderColor: [
                    'rgb(40, 167, 69)',
                    'rgb(247, 147, 26)',
                    'rgb(220, 53, 69)'
                ],
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: {
                    position: 'top',
                }
            }
        }
    });
}

// Load all dashboard data
async function loadDashboardData() {
    await loadPriceHistory();
    await loadSentimentData();
    await loadPerformanceData();
    updateDataFreshness();
}

// Load price history from API
async function loadPriceHistory() {
    try {
        const response = await fetch('/api/price_history?days=30');
        const data = await response.json();
        
        if (data.status === 'success') {
            const prices = data.data;
            
            // Update price chart
            priceChart.data.labels = prices.map(item => {
                const date = new Date(item.date);
                return `${date.getMonth()+1}/${date.getDate()}`;
            });
            priceChart.data.datasets[0].data = prices.map(item => item.price);
            priceChart.update();
        }
    } catch (error) {
        console.error('Error loading price history:', error);
        loadSamplePriceData();
    }
}

// Load sentiment data from API
async function loadSentimentData() {
    try {
        const response = await fetch('/api/sentiment_data');
        const data = await response.json();
        
        if (data.status === 'success') {
            const sentiment = data.data;
            sentimentChart.data.datasets[0].data = [
                sentiment.positive,
                sentiment.neutral,
                sentiment.negative
            ];
            sentimentChart.update();
        }
    } catch (error) {
        console.error('Error loading sentiment data:', error);
        // Fallback to sample data
        sentimentChart.data.datasets[0].data = [12, 8, 5];
        sentimentChart.update();
    }
}

// Load performance data from API
async function loadPerformanceData() {
    try {
        const response = await fetch('/api/model_performance');
        const data = await response.json();
        
        if (data.status === 'success') {
            const performance = data.data;
            
            // Update performance chart
            performanceChart.data.datasets[0].data = [
                performance.correct_predictions,
                performance.total_predictions - performance.correct_predictions
            ];
            performanceChart.update();
            
            // Update stats cards with proper rounding
            document.getElementById('accuracyStat').textContent = `${Math.round(performance.accuracy)}%`;
            document.getElementById('upAccuracy').textContent = `${Math.round(performance.up_accuracy)}%`;
            document.getElementById('downAccuracy').textContent = `${Math.round(performance.down_accuracy)}%`;
            document.getElementById('avgConfidence').textContent = `${Math.round(performance.avg_confidence)}%`;
        }
    } catch (error) {
        console.error('Error loading performance data:', error);
        // Fallback to sample data
        document.getElementById('accuracyStat').textContent = '65%';
        document.getElementById('upAccuracy').textContent = '68%';
        document.getElementById('downAccuracy').textContent = '62%';
        document.getElementById('avgConfidence').textContent = '71%';
    }
}

// Load analytics data
async function loadAnalyticsData() {
    await loadFeatureImportance();
    await loadConfidenceDistribution();
}

// Load feature importance from API
async function loadFeatureImportance() {
    try {
        const response = await fetch('/api/feature_importance');
        const data = await response.json();
        
        if (data.status === 'success') {
            const features = data.data;
            
            featureChart.data.labels = features.features;
            featureChart.data.datasets[0].data = features.importance;
            featureChart.update();
        }
    } catch (error) {
        console.error('Error loading feature importance:', error);
    }
}

// Load confidence distribution (calculated from history)
async function loadConfidenceDistribution() {
    try {
        const response = await fetch('/api/prediction_history?limit=50');
        const data = await response.json();
        
        if (data.status === 'success') {
            const history = data.data;
            
            let highConfidence = 0, mediumConfidence = 0, lowConfidence = 0;
            
            history.forEach(prediction => {
                if (prediction.confidence >= 70) highConfidence++;
                else if (prediction.confidence >= 50) mediumConfidence++;
                else lowConfidence++;
            });
            
            confidenceChart.data.datasets[0].data = [highConfidence, mediumConfidence, lowConfidence];
            confidenceChart.update();
        }
    } catch (error) {
        console.error('Error loading confidence distribution:', error);
    }
}

// Load history data
async function loadHistoryData() {
    await loadPredictionHistory();
    updateCalendarView();
}

// Load prediction history from API
async function loadPredictionHistory() {
    try {
        const response = await fetch('/api/prediction_history?limit=20');
        const data = await response.json();
        
        if (data.status === 'success') {
            predictionHistory = data.data;
            updateHistoryDisplay();
        }
    } catch (error) {
        console.error('Error loading prediction history:', error);
        loadSampleHistory();
    }
}

// Update history display
function updateHistoryDisplay() {
    const historyList = document.getElementById('historyList');
    historyList.innerHTML = '';
    
    predictionHistory.forEach(item => {
        const historyItem = document.createElement('div');
        historyItem.className = 'history-item';
        
        const date = new Date(item.timestamp);
        const formattedDate = `${date.getMonth()+1}/${date.getDate()}/${date.getFullYear()}`;
        
        // Round probabilities for display
        const upProb = Math.round(item.up_probability * 100) / 100;
        const downProb = Math.round(item.down_probability * 100) / 100;
        
        historyItem.innerHTML = `
            <div>
                <strong>${formattedDate}</strong>
                <div style="font-size: 0.8em; color: #6c757d;">
                    Price: $${item.current_price?.toLocaleString() || 'N/A'}
                </div>
            </div>
            <div style="text-align: right;">
                <span class="prediction-badge ${item.prediction === 'UP' ? 'badge-up' : 'badge-down'}">
                    ${item.prediction} (${item.confidence}%)
                </span>
                <div style="margin-top: 5px; font-size: 0.8em; color: #6c757d;">
                    UP: ${upProb}% | DOWN: ${downProb}%
                </div>
            </div>
        `;
        
        historyList.appendChild(historyItem);
    });
    
    // Update performance summary
    updatePerformanceSummary();
}

// Update performance summary
function updatePerformanceSummary() {
    if (predictionHistory.length === 0) {
        document.getElementById('performanceSummary').innerHTML = `
            <p>No prediction history available</p>
        `;
        return;
    }
    
    const total = predictionHistory.length;
    const avgConfidence = predictionHistory.reduce((sum, item) => sum + item.confidence, 0) / total;
    
    document.getElementById('performanceSummary').innerHTML = `
        <p><strong>Total Predictions:</strong> ${total}</p>
        <p><strong>Average Confidence:</strong> ${Math.round(avgConfidence)}%</p>
        <p><strong>Recent Activity:</strong> Active</p>
        <p><strong>Last Prediction:</strong> ${new Date(predictionHistory[0].timestamp).toLocaleDateString()}</p>
    `;
}

// Update calendar view
function updateCalendarView() {
    const calendarGrid = document.getElementById('calendarGrid');
    if (!calendarGrid) return;
    
    calendarGrid.innerHTML = '';
    
    // Get last 7 days of predictions
    const lastWeek = predictionHistory.slice(0, 7);
    
    for (let i = 6; i >= 0; i--) {
        const date = new Date();
        date.setDate(date.getDate() - i);
        const dateStr = `${date.getMonth()+1}/${date.getDate()}`;
        
        const prediction = lastWeek.find(p => {
            const predDate = new Date(p.timestamp);
            return predDate.toDateString() === date.toDateString();
        });
        
        const dayElement = document.createElement('div');
        dayElement.className = `calendar-day ${prediction ? (prediction.prediction === 'UP' ? 'up' : 'down') : ''}`;
        dayElement.innerHTML = `
            <div>${dateStr}</div>
            <div style="font-size: 0.7em; margin-top: 2px;">
                ${prediction ? prediction.prediction : '-'}
            </div>
        `;
        
        calendarGrid.appendChild(dayElement);
    }
}

// Update data freshness indicator
function updateDataFreshness() {
    const freshnessElement = document.getElementById('dataFreshness');
    if (!freshnessElement) return;
    
    // This would ideally come from the backend status
    const now = new Date();
    const lastUpdate = new Date(); // This should come from backend status
    const hoursSinceUpdate = Math.floor((now - lastUpdate) / (1000 * 60 * 60));
    
    if (hoursSinceUpdate < 1) {
        freshnessElement.innerHTML = '<span class="fresh">🟢 Data Updated < 1h ago</span>';
    } else if (hoursSinceUpdate < 24) {
        freshnessElement.innerHTML = `<span class="stale">🟡 Data Updated ${hoursSinceUpdate}h ago</span>`;
    } else {
        freshnessElement.innerHTML = '<span class="outdated">🔴 Data Outdated - Run Update</span>';
    }
}

// Fallback to sample data if APIs fail
function loadSamplePriceData() {
    const samplePrices = [];
    let basePrice = 90000;
    for (let i = 30; i >= 0; i--) {
        const price = basePrice + (Math.random() - 0.5) * 10000;
        samplePrices.push(price);
        basePrice = price;
    }
    
    priceChart.data.labels = Array.from({length: 31}, (_, i) => `${i+1}d`);
    priceChart.data.datasets[0].data = samplePrices;
    priceChart.update();
}

function loadSampleHistory() {
    const sampleHistory = [
        { timestamp: new Date().toISOString(), prediction: 'UP', confidence: 68, current_price: 95123, up_probability: 68, down_probability: 32 },
        { timestamp: new Date(Date.now() - 86400000).toISOString(), prediction: 'DOWN', confidence: 72, current_price: 94876, up_probability: 28, down_probability: 72 },
        { timestamp: new Date(Date.now() - 172800000).toISOString(), prediction: 'UP', confidence: 54, current_price: 94567, up_probability: 54, down_probability: 46 },
    ];
    
    predictionHistory = sampleHistory;
    updateHistoryDisplay();
    updateCalendarView();
}

// Enhanced getPrediction function with better loading states
async function getPrediction() {
    showLoading(
        'Analyzing current market data...',
        'Processing Wikipedia sentiment and technical indicators...'
    );
    disableButtons(true);
    
    try {
        console.log('Fetching prediction from server...');
        const response = await fetch('/predict');
        const data = await response.json();
        
        console.log('Prediction API response:', data);
        
        hideLoading();
        disableButtons(false);
        
        if (data.error) {
            console.error('Prediction error from server:', data.error);
            showError(data.error);
            return;
        }
        
        // Ensure we have all required fields
        if (!data.prediction || data.confidence === undefined || !data.current_price) {
            console.error('Invalid prediction data received:', data);
            showError('Invalid prediction data received from server');
            return;
        }
        
        console.log('Updating display with valid prediction data');
        updatePredictionDisplay(data);
        checkStatus();
        updateDataFreshness();
        
        // Refresh dashboard data
        loadDashboardData();
        
    } catch (error) {
        console.error('Network error fetching prediction:', error);
        hideLoading();
        disableButtons(false);
        showError('Failed to get prediction: ' + error.message);
    }
}

async function updateData() {
    showLoading(
        'Updating Wikipedia and price data...',
        'This may take a few minutes as we retrain the AI model...'
    );
    disableButtons(true);
    
    try {
        const response = await fetch('/update', { method: 'POST' });
        const data = await response.json();
        
        hideLoading();
        disableButtons(false);
        
        if (data.status === 'success') {
            // Show success message with animation
            const updateBtn = document.getElementById('updateBtn');
            updateBtn.innerHTML = '<i class="fas fa-check"></i> Updated!';
            updateBtn.style.background = 'linear-gradient(135deg, #28a745, #20c997)';
            
            setTimeout(() => {
                updateBtn.innerHTML = '<i class="fas fa-sync-alt"></i> Update Data';
                updateBtn.style.background = '';
            }, 2000);
            
            checkStatus();
            updateDataFreshness();
            loadDashboardData();
        } else {
            showError('Update failed: ' + data.message);
        }
        
    } catch (error) {
        hideLoading();
        disableButtons(false);
        showError('Update failed: ' + error.message);
    }
}

async function checkStatus() {
    try {
        const response = await fetch('/status');
        const data = await response.json();
        
        document.getElementById('statusInfo').innerHTML = `
            Model: ${data.model_loaded ? '✅ Loaded' : '❌ Missing'}<br>
            Data: ${data.data_loaded ? '✅ Loaded' : '❌ Missing'}<br>
            Last Update: ${data.last_update}<br>
            Current Time: ${data.current_time}
        `;
    } catch (error) {
        console.error('Status check failed:', error);
        document.getElementById('statusInfo').textContent = 'Unable to check status';
    }
}

// Enhanced prediction display function
function updatePredictionDisplay(data) {
    console.log('Starting prediction display update with:', data);
    
    // Get all required elements
    const card = document.getElementById('predictionCard');
    const predictionText = document.getElementById('predictionText');
    const confidenceValue = document.getElementById('confidenceValue');
    const confidenceFill = document.getElementById('confidenceFill');
    const priceText = document.getElementById('priceText');
    const dateText = document.getElementById('predictionDate');
    
    // Verify all elements exist
    if (!card || !predictionText || !confidenceValue || !confidenceFill || !priceText || !dateText) {
        console.error('Missing required DOM elements:', {
            card: !!card,
            predictionText: !!predictionText,
            confidenceValue: !!confidenceValue,
            confidenceFill: !!confidenceFill,
            priceText: !!priceText,
            dateText: !!dateText
        });
        showError('UI elements not found');
        return;
    }
    
    try {
        // Reset card state
        card.className = 'prediction-card';
        card.classList.remove('pulse');
        
        // Set prediction style
        const isUp = data.prediction === 'UP';
        card.classList.add(isUp ? 'prediction-up' : 'prediction-down');
        
        // Update prediction text
        predictionText.innerHTML = isUp ? 
            '<i class="fas fa-arrow-up"></i> PRICE WILL GO UP' : 
            '<i class="fas fa-arrow-down"></i> PRICE WILL GO DOWN';
        predictionText.style.color = isUp ? '#28a745' : '#F7931A';
        
        // Update confidence
        const confidence = parseFloat(data.confidence);
        confidenceValue.textContent = `${confidence}%`;
        
        // Update confidence meter with animation
        setTimeout(() => {
            confidenceFill.style.width = `${confidence}%`;
            
            // Set confidence color
            if (confidence >= 70) {
                confidenceFill.className = 'confidence-fill confidence-high';
            } else if (confidence >= 50) {
                confidenceFill.className = 'confidence-fill confidence-medium';
            } else {
                confidenceFill.className = 'confidence-fill confidence-low';
            }
        }, 100);
        
        // Update price with animation
        priceText.textContent = `$${parseFloat(data.current_price).toLocaleString()}`;
        priceText.classList.add('price-update');
        setTimeout(() => {
            priceText.classList.remove('price-update');
        }, 1500);
        
        // Update date
        dateText.textContent = `Prediction for: ${data.prediction_date}`;
        
        // Add pulse for high confidence
        if (confidence >= 80) {
            card.classList.add('pulse');
        }
        
        console.log('Prediction display updated successfully');
        
    } catch (error) {
        console.error('Error in prediction display:', error);
        showError('Display error: ' + error.message);
    }
}

function showLoading(message, submessage = '') {
    const loading = document.getElementById('loading');
    const loadingText = document.getElementById('loadingText');
    const loadingSubtext = document.getElementById('loadingSubtext');
    
    if (loading && loadingText) {
        loading.style.display = 'block';
        loadingText.textContent = message;
        if (loadingSubtext && submessage) {
            loadingSubtext.textContent = submessage;
        }
    }
}

function hideLoading() {
    const loading = document.getElementById('loading');
    if (loading) {
        loading.style.display = 'none';
    }
}

function disableButtons(disabled) {
    const predictBtn = document.getElementById('predictBtn');
    const updateBtn = document.getElementById('updateBtn');
    if (predictBtn) predictBtn.disabled = disabled;
    if (updateBtn) updateBtn.disabled = disabled;
}

// Enhanced error display
function showError(message) {
    console.error('Showing error:', message);
    
    const predictionText = document.getElementById('predictionText');
    const confidenceValue = document.getElementById('confidenceValue');
    const confidenceFill = document.getElementById('confidenceFill');
    const priceText = document.getElementById('priceText');
    const dateText = document.getElementById('predictionDate');
    const card = document.getElementById('predictionCard');
    
    // Reset all elements to error state
    if (card) {
        card.className = 'prediction-card';
        card.classList.remove('pulse', 'prediction-up', 'prediction-down');
    }
    
    if (predictionText) {
        predictionText.innerHTML = '<i class="fas fa-exclamation-triangle"></i> Error';
        predictionText.style.color = '#dc3545';
    }
    
    if (confidenceValue) {
        confidenceValue.textContent = '0%';
    }
    
    if (confidenceFill) {
        confidenceFill.style.width = '0%';
        confidenceFill.className = 'confidence-fill confidence-low';
    }
    
    if (priceText) {
        priceText.textContent = '$--';
    }
    
    if (dateText) {
        dateText.textContent = '';
    }
}