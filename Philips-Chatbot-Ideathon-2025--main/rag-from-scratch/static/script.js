document.addEventListener('DOMContentLoaded', () => {
    const messageContainer = document.querySelector('.message-container');
    const inputField = document.querySelector('.input-field');
    const sendButton = document.querySelector('.send-button');
    const typingIndicator = document.querySelector('.typing-indicator');

    function formatTimestamp() {
        const now = new Date();
        const hours = now.getHours().toString().padStart(2, '0');
        const minutes = now.getMinutes().toString().padStart(2, '0');
        return `${hours}:${minutes}`;
    }

    function createSourcesSection(sources) {
        if (!sources || sources.length === 0) return '';
        
        const sourcesHtml = sources.map(source => 
            `<div class="source-item">${source}</div>`
        ).join('');
        
        return `
            <div class="sources-container">
                <div>Sources:</div>
                ${sourcesHtml}
            </div>
        `;
    }

    function createFollowUpSection(followUps) {
        if (!followUps || followUps.length === 0) return '';
        
        const followUpsHtml = followUps.map(question => 
            `<button class="follow-up-button" onclick="this.closest('.message-wrapper').querySelector('.follow-up-container').remove(); document.querySelector('.input-field').value = this.textContent; document.querySelector('.send-button').click();">${question}</button>`
        ).join('');
        
        return `
            <div class="follow-up-container">
                <div>Related questions:</div>
                ${followUpsHtml}
            </div>
        `;
    }

    function createConfidenceSection(confidence) {
        if (!confidence) return '';
        
        return `
            <div class="confidence-indicator">
                <div class="confidence-score">
                    <span class="confidence-label">Confidence: ${confidence.level}</span>
                    <div class="confidence-bar">
                        <div class="confidence-fill confidence-${confidence.level.toLowerCase()}" 
                             style="width: ${confidence.score}%"></div>
                    </div>
                </div>
                <div class="confidence-reason">${confidence.reason}</div>
            </div>
        `;
    }

    function addMessage(text, isUser = false, sources = [], followUps = [], confidence = null) {
        const messageWrapper = document.createElement('div');
        messageWrapper.classList.add('message-wrapper');
        if (isUser) {
            messageWrapper.classList.add('user-message-wrapper');
        }

        const avatar = document.createElement('div');
        avatar.classList.add('avatar');
        avatar.classList.add(isUser ? 'user-avatar' : 'bot-avatar');

        const messageDiv = document.createElement('div');
        messageDiv.classList.add('message');
        messageDiv.classList.add(isUser ? 'user-message' : 'bot-message');

        // Add message content
        const messageContent = document.createElement('div');
        messageContent.classList.add('message-content');
        messageContent.innerHTML = text; // Changed from textContent to innerHTML to support formatting
        messageDiv.appendChild(messageContent);

        // Add confidence, sources and follow-ups for bot messages
        if (!isUser) {
            messageDiv.insertAdjacentHTML('beforeend', createConfidenceSection(confidence));
            messageDiv.insertAdjacentHTML('beforeend', createSourcesSection(sources));
            messageDiv.insertAdjacentHTML('beforeend', createFollowUpSection(followUps));
        }

        const timestamp = document.createElement('div');
        timestamp.classList.add('timestamp');
        timestamp.textContent = formatTimestamp();
        messageDiv.appendChild(timestamp);

        if (isUser) {
            messageWrapper.appendChild(messageDiv);
            messageWrapper.appendChild(avatar);
        } else {
            messageWrapper.appendChild(avatar);
            messageWrapper.appendChild(messageDiv);
        }

        messageContainer.appendChild(messageWrapper);
        messageContainer.scrollTop = messageContainer.scrollHeight;
    }

    async function sendMessage() {
        const message = inputField.value.trim();
        if (!message) return;

        // Add user message
        addMessage(message, true);
        inputField.value = '';

        // Show typing indicator
        typingIndicator.classList.add('visible');

        try {
            const response = await fetch('/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ query: message }),
            });

            const data = await response.json();
            
            // Hide typing indicator
            typingIndicator.classList.remove('visible');

            // Add bot response with confidence, sources and follow-ups
            addMessage(data.response, false, data.sources, data.follow_up_questions, data.confidence);
        } catch (error) {
            console.error('Error:', error);
            typingIndicator.classList.remove('visible');
            addMessage('Sorry, there was an error processing your request.', false);
        }
    }

    // Event listeners
    sendButton.addEventListener('click', sendMessage);
    inputField.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            sendMessage();
        }
    });
}); 