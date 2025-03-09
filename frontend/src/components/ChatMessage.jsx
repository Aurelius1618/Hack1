const ChatMessage = ({ message, isUser }) => {
    return (
      <div className={`message ${isUser ? 'user-message' : 'bot-message'}`}>
        <div className="font-medium mb-1">{isUser ? 'You' : 'Tap Bonds AI'}</div>
        <div className="whitespace-pre-line">{message}</div>
      </div>
    );
  };
  
  export default ChatMessage;
  