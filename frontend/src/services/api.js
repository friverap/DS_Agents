import axios from 'axios';

// Create axios instance with base URL
const api = axios.create({
  baseURL: '/api',
  headers: {
    'Content-Type': 'application/json',
  },
});

// API endpoints for model providers
export const modelsApi = {
  // Get available model providers and their models
  getProviders: () => api.get('/models/providers'),
  
  // Get current model configuration
  getCurrentModel: () => api.get('/models/current'),
  
  // Configure model provider and specific model
  configureModel: (provider, model) => api.post('/models/configure', { provider, model }),
  
  // Get the status of API keys for all providers
  getApiKeyStatus: () => api.get('/models/api-keys/status'),
  
  // Update API key for a specific provider
  updateApiKey: (provider, apiKey) => api.post('/models/api-keys/update', { provider, apiKey }),
};

// API endpoints for chat
export const chatApi = {
  // Send a chat message to the backend
  sendChatMessage: (message, conversationId = null, modelProvider = null, modelName = null) => 
    api.post('/chat', { message, conversation_id: conversationId, model_provider: modelProvider, model_name: modelName }),
};

// API endpoints for web search
export const searchApi = {
  // Search the web using Brave Search API
  webSearch: (query, count = 10) => api.get('/search', { params: { q: query, count } }),
};

// API endpoints for file upload
export const uploadApi = {
  // Upload a file
  uploadFile: (formData) => api.post('/upload', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  }),
};

// API endpoints for feedback
export const feedbackApi = {
  // Submit feedback
  submitFeedback: (feedback) => api.post('/feedback', feedback),
};

export default api; 