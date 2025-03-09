import { useState, useEffect, useRef } from 'react';
import SearchBar from './SearchBar';
import ChatMessage from './ChatMessage';
import { sendQuery } from '../services/api';

const ChatInterface = () => {
  const [messages, setMessages] = useState([
    { text: "Hello! I'm the Tap Bonds AI assistant. How can I help you with bond information today?", isUser: false }
  ]);
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef(null);

  const handleSubmit = async (query) => {
    // Add user message to chat
    setMessages(prev => [...prev, { text: query, isUser: true }]);
    
    // Add a temporary "Thinking..." message
    setMessages(prev => [...prev, { text: "Thinking...", isUser: false, isLoading: true }]);
    setIsLoading(true);

    try {
      // Send query to backend
      const { response } = await sendQuery(query);
      
      // Replace the "Thinking..." message with the actual response
      setMessages(prev => prev.map((msg, idx) => 
        idx === prev.length - 1 && msg.isLoading 
          ? { text: response, isUser: false } 
          : msg
      ));
    } catch (error) {
      // Replace the "Thinking..." message with an error message
      setMessages(prev => prev.map((msg, idx) => 
        idx === prev.length - 1 && msg.isLoading 
          ? { text: "Sorry, I encountered an error processing your request. Please try again.", isUser: false } 
          : msg
      ));
    } finally {
      setIsLoading(false);
    }
  };

  // Auto-scroll to bottom when messages change
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  return (
    <div className="chat-container flex flex-col h-screen">
      <div className="bg-white p-4 shadow-md rounded-t-lg">
        <h1 className="text-2xl font-bold text-primary text-center">Tap Bonds AI Assistant</h1>
      </div>
      
      <div className="flex-grow overflow-auto p-4">
        <div className="space-y-4">
          {messages.map((msg, index) => (
            <ChatMessage 
              key={index} 
              message={msg.text} 
              isUser={msg.isUser} 
              isLoading={msg.isLoading} 
            />
          ))}
          <div ref={messagesEndRef} />
        </div>
      </div>
      
      <div className="p-4 bg-white border-t">
        <SearchBar onSubmit={handleSubmit} isLoading={isLoading} />
      </div>
    </div>
  );
};

export default ChatInterface;
