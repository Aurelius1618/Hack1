import axios from 'axios';

const API_URL = 'http://localhost:5000/api';

export const sendQuery = async (query) => {
  try {
    const response = await axios.post(`${API_URL}/query`, { query });
    return response.data;
  } catch (error) {
    console.error('Error sending query:', error);
    throw error;
  }
};
