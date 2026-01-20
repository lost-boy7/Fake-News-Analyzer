// Fake News Detection System - Main JavaScript

document.addEventListener('DOMContentLoaded', () => {
    // Tab switching
    const tabBtns = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');
    
    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const tabName = btn.dataset.tab;
            
            // Update active tab button
            tabBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            
            // Update active tab content
            tabContents.forEach(content => {
                content.classList.remove('active');
                if (content.id === `${tabName}-tab`) {
                    content.classList.add('active');
                }
            });
        });
    });
    
    // Analyze text button
    const analyzeTextBtn = document.getElementById('analyzeTextBtn');
    const analyzeUrlBtn = document.getElementById('analyzeUrlBtn');
    
    analyzeTextBtn.addEventListener('click', () => {
        const text = document.getElementById('newsText').value.trim();
        if (text) {
            analyzeNews('text', text);
        } else {
            showError('Please enter some text to analyze.');
        }
    });
    
    analyzeUrlBtn.addEventListener('click', () => {
        const url = document.getElementById('newsUrl').value.trim();
        if (url) {
            if (isValidUrl(url)) {
                analyzeNews('url', url);
            } else {
                showError('Please enter a valid URL.');
            }
        } else {
            showError('Please enter a URL to analyze.');
        }
    });
    
    // Enter key support for URL input
    document.getElementById('newsUrl').addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            analyzeUrlBtn.click();
        }
    });
});

function isValidUrl(string) {
    try {
        new URL(string);
        return true;
    } catch (_) {
        return false;
    }
}

async function analyzeNews(type, content) {
    const resultCard = document.getElementById('resultCard');
    const errorCard = document.getElementById('errorCard');
    const analyzeBtn = type === 'text' ? 
        document.getElementById('analyzeTextBtn') : 
        document.getElementById('analyzeUrlBtn');
    
    // Hide previous results
    resultCard.classList.add('hidden');
    errorCard.classList.add('hidden');
    
    // Show loading state
    const btnText = analyzeBtn.querySelector('.btn-text');
    const loader = analyzeBtn.querySelector('.loader');
    btnText.classList.add('hidden');
    loader.classList.remove('hidden');
    analyzeBtn.disabled = true;
    
    try {
        const response = await fetch('/analyze', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                type: type,
                content: content
            })
        });
        
        const data = await response.json();
        
        if (data.error) {
            showError(data.error);
        } else {
            displayResults(data);
        }
    } catch (error) {
        showError('Network error. Please check your connection and try again.');
        console.error('Error:', error);
    } finally {
        // Reset button state
        btnText.classList.remove('hidden');
        loader.classList.add('hidden');
        analyzeBtn.disabled = false;
    }
}

function displayResults(data) {
    const resultCard = document.getElementById('resultCard');
    const predictionBadge = document.getElementById('predictionBadge');
    const predictionText = document.getElementById('predictionText');
    const confidenceScore = document.getElementById('confidenceScore');
    
    // Update prediction badge
    predictionText.textContent = data.prediction;
    predictionBadge.className = 'prediction-badge ' + data.prediction.toLowerCase();
    
    // Update confidence score
    confidenceScore.textContent = data.confidence + '%';
    
    // Update probability bars
    updateProbabilityBar('fake', data.probabilities.fake);
    updateProbabilityBar('real', data.probabilities.real);
    
    // Update features
    if (data.features) {
        document.getElementById('wordCount').textContent = data.features.word_count || '-';
        document.getElementById('sensationalCount').textContent = data.features.sensational_count || '0';
        document.getElementById('emotionalCount').textContent = data.features.emotional_count || '0';
    }
    
    // Show result card with animation
    resultCard.classList.remove('hidden');
    resultCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function updateProbabilityBar(type, percentage) {
    const probElement = document.getElementById(`${type}Prob`);
    const fillElement = document.getElementById(`${type}Fill`);
    
    probElement.textContent = percentage + '%';
    
    // Animate the bar
    setTimeout(() => {
        fillElement.style.width = percentage + '%';
    }, 100);
}

function showError(message) {
    const errorCard = document.getElementById('errorCard');
    const errorMessage = document.getElementById('errorMessage');
    const resultCard = document.getElementById('resultCard');
    
    resultCard.classList.add('hidden');
    errorMessage.textContent = message;
    errorCard.classList.remove('hidden');
    errorCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// Sample news articles for testing
const sampleArticles = {
    fake: "SHOCKING: Scientists discover miracle cure that Big Pharma doesn't want you to know! This amazing breakthrough will change everything! URGENT - share before it gets deleted!",
    real: "According to a study published in Nature Medicine, researchers at Johns Hopkins University have made incremental progress in cancer treatment methods. The peer-reviewed research suggests potential applications pending further clinical trials."
};

// Add sample buttons (optional - for demo purposes)
function addSampleButtons() {
    const textTab = document.getElementById('text-tab');
    
    const sampleContainer = document.createElement('div');
    sampleContainer.style.cssText = 'display: flex; gap: 10px; margin-bottom: 15px;';
    
    const fakeBtn = createSampleButton('Try Fake News', sampleArticles.fake);
    const realBtn = createSampleButton('Try Real News', sampleArticles.real);
    
    sampleContainer.appendChild(fakeBtn);
    sampleContainer.appendChild(realBtn);
    
    textTab.insertBefore(sampleContainer, textTab.firstChild);
}

function createSampleButton(text, content) {
    const btn = document.createElement('button');
    btn.textContent = text;
    btn.className = 'btn';
    btn.style.cssText = 'width: auto; padding: 8px 16px; font-size: 0.9em; background: rgba(99, 102, 241, 0.2); color: #a5b4fc;';
    btn.onclick = (e) => {
        e.preventDefault();
        document.getElementById('newsText').value = content;
    };
    return btn;
}

// Uncomment to add sample buttons
// addSampleButtons();
