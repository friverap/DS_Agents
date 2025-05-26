import React, { useState, useEffect } from 'react';
import { Button, Typography, Spin, notification } from 'antd';
import ChatInterface from './components/ChatInterface';

function App() {
  const [showQuestions, setShowQuestions] = useState(true);
  const [selectedQuestion, setSelectedQuestion] = useState(null);
  const [welcomeData, setWelcomeData] = useState({
    welcome_message: "DSAgency",
    description: "AI-Powered Assistant Platform",
    example_questions: [
      "What can you help me with?",
      "Tell me about the available agents in this system",
      "How does this platform work?",
      "What are your capabilities?",
      "Can you help me with general questions?"
    ]
  });
  const [loading, setLoading] = useState(false);
  
  // Handle question click
  const handleQuestionClick = (question) => {
    setSelectedQuestion(question);
    setShowQuestions(false);
  };

  return (
    <div className="app-container">
      <h1>DSAgency - AI Assistant Platform</h1>
      <div>
        <h2>{welcomeData.welcome_message}</h2>
        {welcomeData.description && <p>{welcomeData.description}</p>}
        
        {loading ? (
          <div style={{ margin: '20px 0', textAlign: 'center' }}>
            <Spin /> <span style={{ marginLeft: '10px' }}>Loading...</span>
          </div>
        ) : showQuestions ? (
          <div style={{ 
            marginTop: '20px', 
            marginBottom: '20px',
            maxWidth: '600px'
          }}>
            <Typography.Text strong style={{ display: 'block', marginBottom: '10px', fontSize: '16px' }}>
              Choose a question to get started:
            </Typography.Text>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
              {welcomeData.example_questions.map((question, idx) => (
                <Button 
                  key={idx}
                  onClick={() => handleQuestionClick(question)}
                  style={{ textAlign: 'left', height: 'auto', padding: '8px 12px' }}
                >
                  {question}
                </Button>
              ))}
            </div>
          </div>
        ) : null}
      </div>
      
      <ChatInterface 
        initialQuestion={selectedQuestion} 
        onChatStart={() => setShowQuestions(false)}
      />
    </div>
  );
}

export default App;
