import React, { useState, useRef, useEffect } from 'react';
import { Input, Button, Card, Typography, Spin, Divider, notification } from 'antd';
import { SendOutlined, RobotOutlined, UserOutlined } from '@ant-design/icons';
import { chatApi } from '../services/api';
import axios from 'axios';

const { Text, Paragraph } = Typography;

const ChatInterface = ({ modelProvider, modelName, initialQuestion = null, onChatStart = () => {} }) => {
  const [messages, setMessages] = useState([]);
  const [inputMessage, setInputMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [conversationId, setConversationId] = useState(null);
  const [apiStatus, setApiStatus] = useState(null);
  const messagesEndRef = useRef(null);

  // Handle example question click (for any remaining buttons in the chat)
  const handleExampleClick = (question) => {
    setInputMessage(question);
    // Send the message directly instead of using setTimeout
    handleSendMessage(question);
    if (onChatStart) onChatStart();
  };

  // Process initialQuestion if provided
  useEffect(() => {
    if (initialQuestion && !messages.some(msg => msg.type === 'user')) {
      handleSendMessage(initialQuestion);
    }
  }, [initialQuestion]);

  // Log when component mounts and check API status
  useEffect(() => {
    console.log('ChatInterface mounted with provider:', modelProvider, 'model:', modelName);
    
    // Just set API status without triggering a chat
    setApiStatus('Chat API Ready');
  }, []);

  // Auto-scroll to the bottom when messages change
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSendMessage = async (messageToSend = null) => {
    // Use provided message or input field value
    const message = messageToSend || inputMessage;
    
    if (!message.trim()) return;
    
    // Notify parent that chat has started
    if (onChatStart) onChatStart();
    
    // Add user message to chat
    const userMessage = {
      type: 'user',
      content: message,
      timestamp: new Date().toISOString()
    };
    
    setMessages(prevMessages => [...prevMessages, userMessage]);
    setInputMessage('');
    setIsLoading(true);
    
    try {
      console.log('Sending message to API:', message);
      console.log('Using model provider:', modelProvider);
      console.log('Using model name:', modelName);
      console.log('Using conversation ID:', conversationId);
      
      // Send message to API
      const response = await chatApi.sendChatMessage(
        message, 
        conversationId,
        modelProvider,
        modelName
      );
      
      console.log('Received response from API:', response);
      
      // Store conversation ID for future messages
      if (response.data.conversation_id) {
        console.log('Setting conversation ID:', response.data.conversation_id);
        setConversationId(response.data.conversation_id);
      }
      
      // Get the response content, with fallback to extracting from agent_results
      let responseContent = response.data.response;
      
      // Filter out unwanted standard summary messages
      if (responseContent && 
          responseContent.includes("Data Analysis Summary") && 
          (responseContent.includes("initial consultation") || 
           responseContent.includes("standard data science workflow"))) {
        responseContent = "I'm ready to help with your data science needs. Please provide more details or a specific dataset for analysis.";
      }
      
      // If response is empty but we have agent_results, try to extract a meaningful response
      if (!responseContent && response.data.agent_results) {
        console.log('Response is empty, extracting from agent_results:', response.data.agent_results);
        
        const agentResults = response.data.agent_results;
        
        // Try to extract a meaningful response from various agent results
        if (agentResults.goalrefiner && agentResults.goalrefiner.refined_goal) {
          responseContent = `I understand you want to ${agentResults.goalrefiner.refined_goal}\n\n`;
        }
        
        if (agentResults.analyticalplanner && agentResults.analyticalplanner.plan_rationale) {
          responseContent += `Here's my plan to help you:\n${agentResults.analyticalplanner.plan_rationale}\n\n`;
        }
        
        // If we still don't have a response, provide a generic one
        if (!responseContent.trim()) {
          responseContent = "I've analyzed your request and formulated a plan to help you. However, I need more specific information to proceed. Could you provide more details about your data science task?";
        }
      }
      
      // Add AI response to chat
      const aiMessage = {
        type: 'ai',
        content: responseContent,
        timestamp: new Date().toISOString(),
        agentResults: response.data.agent_results
      };
      
      console.log('Adding AI message to chat:', aiMessage);
      setMessages(prevMessages => [...prevMessages, aiMessage]);
    } catch (error) {
      console.error('Error sending message:', error);
      console.error('Error details:', {
        name: error.name,
        message: error.message,
        stack: error.stack,
        response: error.response ? {
          status: error.response.status,
          statusText: error.response.statusText,
          data: error.response.data
        } : 'No response data'
      });
      
      // Add error message to chat
      const errorMessage = {
        type: 'system',
        content: "I'm having trouble generating a response. Please try again.",
        timestamp: new Date().toISOString()
      };
      
      setMessages(prevMessages => [...prevMessages, errorMessage]);
      
      notification.error({
        message: 'Chat Error',
        description: error.response?.data?.detail || error.message || 'Failed to send message',
      });
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  return (
    <Card title="Chat Assistant" style={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      {apiStatus && (
        <div style={{ marginBottom: '10px', padding: '5px', backgroundColor: '#f0f9ff', borderRadius: '4px', fontSize: '12px' }}>
          {apiStatus}
        </div>
      )}
      <div 
        style={{ 
          flexGrow: 1, 
          overflowY: 'auto', 
          maxHeight: '400px',
          marginBottom: '10px',
          padding: '10px',
          backgroundColor: '#f5f5f5',
          borderRadius: '4px'
        }}
      >
        {messages.length === 0 ? (
          <div style={{ textAlign: 'center', color: '#555', marginTop: '20px' }}>
            <RobotOutlined style={{ fontSize: '24px' }} />
            <Paragraph>Ask me about your data science project</Paragraph>
          </div>
        ) : (
          <>
            {messages.map((msg, index) => (
              <div 
                key={index} 
                style={{ 
                  marginBottom: '12px', 
                  textAlign: msg.type === 'user' ? 'right' : 'left' 
                }}
              >
                <div
                  style={{
                    display: 'inline-block',
                    maxWidth: '80%',
                    padding: '10px',
                    borderRadius: '8px',
                    backgroundColor: msg.type === 'user' ? '#1890ff' : (msg.type === 'system' ? '#ffccc7' : 'white'),
                    color: msg.type === 'user' ? 'white' : 'rgba(0, 0, 0, 0.85)',
                    boxShadow: '0 1px 2px rgba(0, 0, 0, 0.1)'
                  }}
                >
                  <div style={{ fontWeight: 'bold', marginBottom: '4px' }}>
                    {msg.type === 'user' ? <UserOutlined /> : <RobotOutlined />}
                    {' '}
                    {msg.type === 'user' ? 'You' : (msg.type === 'system' ? 'System' : 'AI Assistant')}
                  </div>
                  
                  {/* Regular content display */}
                  <div style={{ whiteSpace: 'pre-line' }}>{msg.content}</div>
                </div>
              </div>
            ))}
          </>
        )}
        {isLoading && (
          <div style={{ textAlign: 'center', margin: '10px 0' }}>
            <Spin size="small" /> <Text type="secondary">Thinking...</Text>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>
      
      <div style={{ display: 'flex', marginTop: 'auto' }}>
        <Input.TextArea
          value={inputMessage}
          onChange={(e) => setInputMessage(e.target.value)}
          onKeyPress={handleKeyPress}
          placeholder="Type your message here..."
          autoSize={{ minRows: 1, maxRows: 3 }}
          disabled={isLoading}
          style={{ flexGrow: 1 }}
        />
        <Button 
          type="primary" 
          icon={<SendOutlined />} 
          onClick={handleSendMessage} 
          disabled={isLoading || !inputMessage.trim()}
          style={{ marginLeft: '8px', height: '100%' }}
        />
      </div>
    </Card>
  );
};

export default ChatInterface; 