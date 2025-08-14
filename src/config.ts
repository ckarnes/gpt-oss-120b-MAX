import { readFileSync } from 'node:fs';

export const loadEnvFile = () => {
  try {
    const envContent = readFileSync('.env', 'utf-8');
    const envVars = envContent.split('\n').reduce((acc, line) => {
      const [key, value] = line.split('=');
      if (key && value) {
        acc[key.trim()] = value.trim();
      }
      return acc;
    }, {} as Record<string, string>);

    for (const [key, value] of Object.entries(envVars)) {
      process.env[key] = value;
    }
  } catch (error) {
    console.warn('Could not load .env file:', (error as Error).message);
  }
};

export const OLLAMA_API_KEY = process.env.OLLAMA_API_KEY;
export const PORT = 3304;
export const OLLAMA_HOST = 'https://ollama.com';

export const KNOWN_ENDPOINTS = [
  { method: 'POST', path: '/v1/chat/completions' },
  { method: 'GET', path: '/v1/models' },
  { method: 'POST', path: '/v1/completions' },
  { method: 'POST', path: '/v1/embeddings' },
  { method: 'POST', path: '/api/chat' },
  { method: 'GET', path: '/api/tags' }
];
