"""
Data Collection Script for Web Development LLM
Collects training data from various web development documentation sources
"""

import requests
from bs4 import BeautifulSoup
from pathlib import Path
import json
import time
from typing import List, Dict
from tqdm import tqdm
import re


class WebDevDataCollector:
    """Collects web development documentation and tutorials"""
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Data sources
        self.sources = {
            'mdn': {
                'base_url': 'https://developer.mozilla.org',
                'topics': [
                    '/en-US/docs/Web/HTML',
                    '/en-US/docs/Web/CSS',
                    '/en-US/docs/Web/JavaScript',
                    '/en-US/docs/Web/API',
                ]
            }
        }
        
        self.collected_data = []
        
    def fetch_page(self, url: str, retries: int = 3) -> str:
        """Fetch HTML content from URL with retries"""
        for attempt in range(retries):
            try:
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                return response.text
            except Exception as e:
                if attempt == retries - 1:
                    print(f"Failed to fetch {url}: {e}")
                    return ""
                time.sleep(2 ** attempt)  # Exponential backoff
        return ""
    
    def extract_text_from_html(self, html: str) -> str:
        """Extract clean text from HTML"""
        soup = BeautifulSoup(html, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style", "nav", "footer", "header"]):
            script.decompose()
        
        # Get text
        text = soup.get_text()
        
        # Clean up whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        return text
    
    def extract_code_examples(self, html: str) -> List[str]:
        """Extract code examples from HTML"""
        soup = BeautifulSoup(html, 'html.parser')
        code_blocks = []
        
        # Find code blocks
        for code in soup.find_all(['code', 'pre']):
            code_text = code.get_text().strip()
            if len(code_text) > 10:  # Minimum code length
                code_blocks.append(code_text)
        
        return code_blocks
    
    def collect_from_mdn(self, max_pages: int = 100):
        """Collect data from MDN Web Docs"""
        print("Collecting data from MDN Web Docs...")
        
        collected_urls = set()
        
        for topic_url in self.sources['mdn']['topics']:
            full_url = self.sources['mdn']['base_url'] + topic_url
            
            print(f"\nCollecting from: {full_url}")
            html = self.fetch_page(full_url)
            
            if not html:
                continue
            
            # Extract content
            text = self.extract_text_from_html(html)
            code_examples = self.extract_code_examples(html)
            
            if text:
                self.collected_data.append({
                    'source': 'MDN',
                    'url': full_url,
                    'topic': topic_url.split('/')[-1],
                    'text': text,
                    'code_examples': code_examples,
                    'type': 'documentation'
                })
                collected_urls.add(full_url)
            
            # Find and collect linked pages
            soup = BeautifulSoup(html, 'html.parser')
            links = soup.find_all('a', href=True)
            
            for link in links:
                href = link['href']
                
                # Only collect MDN docs links
                if href.startswith('/en-US/docs/Web/'):
                    link_url = self.sources['mdn']['base_url'] + href
                    
                    if link_url not in collected_urls and len(collected_urls) < max_pages:
                        time.sleep(1)  # Be respectful to the server
                        
                        page_html = self.fetch_page(link_url)
                        if page_html:
                            page_text = self.extract_text_from_html(page_html)
                            page_code = self.extract_code_examples(page_html)
                            
                            if page_text:
                                self.collected_data.append({
                                    'source': 'MDN',
                                    'url': link_url,
                                    'topic': href.split('/')[-1],
                                    'text': page_text,
                                    'code_examples': page_code,
                                    'type': 'documentation'
                                })
                                collected_urls.add(link_url)
                                
                                if len(collected_urls) % 10 == 0:
                                    print(f"Collected {len(collected_urls)} pages...")
    
    def add_synthetic_examples(self):
        """Add synthetic web development examples"""
        print("\nAdding synthetic examples...")
        
        synthetic_examples = [
            # HTML
            {
                'source': 'WebDevDocs',
                'topic': 'HTML',
                'text': 'HTML (HyperText Markup Language) is the standard markup language for documents designed to be displayed in a web browser. It defines the structure and semantic meaning of web content using elements, tags, and attributes.',
                'code_examples': [
                    '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Accessible Semantic Web</title>
</head>
<body>
    <header>
        <nav aria-label="Main Navigation">
            <ul>
                <li><a href="#home">Home</a></li>
                <li><a href="#articles">Articles</a></li>
            </ul>
        </nav>
    </header>
    <main>
        <article>
            <h1>Semantic HTML5 Architecture</h1>
            <section>
                <p>Semantic tags improve accessibility, SEO, and developer readability.</p>
            </section>
        </article>
    </main>
</body>
</html>''',
                ],
                'type': 'documentation'
            },
            # React examples
            {
                'source': 'Synthetic',
                'topic': 'React',
                'text': 'React is a component-based JavaScript library for building interactive user interfaces. State management and lifecycle are handled elegantly via React Hooks such as useState, useEffect, useMemo, and useCallback.',
                'code_examples': [
                    '''import React, { useState, useEffect } from 'react';

export function CounterWidget({ initialCount = 0 }) {
  const [count, setCount] = useState(initialCount);

  useEffect(() => {
    document.title = `Current Count: ${count}`;
  }, [count]);

  return (
    <div className="counter-card">
      <h2>Interactive Counter</h2>
      <p className="value">Count: {count}</p>
      <div className="btn-group">
        <button onClick={() => setCount(c => c + 1)}>Increment</button>
        <button onClick={() => setCount(c => c - 1)}>Decrement</button>
        <button onClick={() => setCount(0)}>Reset</button>
      </div>
    </div>
  );
}''',
                    '''import React, { createContext, useContext, useState } from 'react';

const ThemeContext = createContext();

export function ThemeProvider({ children }) {
  const [theme, setTheme] = useState('dark');
  const toggleTheme = () => setTheme(prev => prev === 'dark' ? 'light' : 'dark');

  return (
    <ThemeContext.Provider value={{ theme, toggleTheme }}>
      {children}
    </ThemeContext.Provider>
  );
}

export const useTheme = () => useContext(ThemeContext);'''
                ],
                'type': 'example'
            },
            # CSS examples
            {
                'source': 'Synthetic',
                'topic': 'CSS',
                'text': 'Modern CSS provides Flexbox for 1D layouts, CSS Grid for 2D layouts, and CSS custom properties (variables) for consistent and reactive design systems.',
                'code_examples': [
                    ''':root {
  --primary-color: #6366f1;
  --bg-dark: #0f172a;
  --card-bg: rgba(30, 41, 59, 0.7);
  --border-glow: rgba(99, 102, 241, 0.3);
}

.dashboard-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
  gap: 1.5rem;
  padding: 2rem;
}

.glass-card {
  background: var(--card-bg);
  backdrop-filter: blur(12px);
  border: 1px solid var(--border-glow);
  border-radius: 16px;
  padding: 1.5rem;
  transition: transform 0.3s ease, box-shadow 0.3s ease;
}

.glass-card:hover {
  transform: translateY(-4px);
  box-shadow: 0 12px 30px rgba(99, 102, 241, 0.25);
}''',
                    '''.flex-center {
  display: flex;
  justify-content: center;
  align-items: center;
  gap: 1rem;
}'''
                ],
                'type': 'example'
            },
            # JavaScript examples
            {
                'source': 'Synthetic',
                'topic': 'JavaScript',
                'text': 'JavaScript is an asynchronous, event-driven, single-threaded language with non-blocking I/O. Promises and async/await simplify asynchronous workflows, error handling, and parallel operations using Promise.all.',
                'code_examples': [
                    '''async function fetchUserData(userId) {
  try {
    const response = await fetch(`https://api.example.com/users/${userId}`);
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    const user = await response.json();
    return { success: true, data: user };
  } catch (error) {
    console.error('Failed to fetch user:', error);
    return { success: false, error: error.message };
  }
}''',
                    '''const debounce = (fn, delay = 300) => {
  let timeoutId;
  return (...args) => {
    clearTimeout(timeoutId);
    timeoutId = setTimeout(() => fn(...args), delay);
  };
};'''
                ],
                'type': 'example'
            },
            # TypeScript
            {
                'source': 'Synthetic',
                'topic': 'TypeScript',
                'text': 'TypeScript adds static type definitions to JavaScript, enabling compile-time type checking, refactoring confidence, and self-documenting codebases with Generics, Interfaces, and Union types.',
                'code_examples': [
                    '''interface ApiResponse<T> {
  data: T;
  status: number;
  message: string;
}

interface UserProfile {
  id: string;
  username: string;
  email: string;
  roles: Array<'admin' | 'user' | 'editor'>;
}

async function fetchProfile(id: string): Promise<ApiResponse<UserProfile>> {
  const res = await fetch(`/api/users/${id}`);
  return res.json();
}'''
                ],
                'type': 'documentation'
            },
            # Next.js
            {
                'source': 'Synthetic',
                'topic': 'Next.js',
                'text': 'Next.js is a full-stack React framework featuring App Router, Server Components (RSC), Server Actions, automatic image optimization, and hybrid static & dynamic rendering.',
                'code_examples': [
                    '''// app/api/cards/route.ts
import { NextResponse } from 'next/server';

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const topic = searchParams.get('topic') || 'general';
  
  return NextResponse.json({
    topic,
    title: `Guide to ${topic}`,
    createdAt: new Date().toISOString(),
  });
}'''
                ],
                'type': 'documentation'
            },
            # Node.js & Express
            {
                'source': 'Synthetic',
                'topic': 'Node.js',
                'text': 'Express is a fast, unopinionated, minimalist web framework for Node.js. It facilitates building robust REST APIs with middleware pipelines and modular routing.',
                'code_examples': [
                    '''const express = require('express');
const cors = require('cors');

const app = express();
app.use(cors());
app.use(express.json());

app.get('/api/health', (req, res) => {
  res.json({ status: 'ok', uptime: process.uptime() });
});

app.post('/api/generate', (req, res) => {
  const { topic } = req.body;
  if (!topic) {
    return res.status(400).json({ error: 'Topic parameter is required' });
  }
  res.json({ topic, message: `Generated card for ${topic}` });
});

const PORT = process.env.PORT || 5000;
app.listen(PORT, () => console.log(`Server running on port ${PORT}`));'''
                ],
                'type': 'example'
            },
            # Python & FastAPI
            {
                'source': 'Synthetic',
                'topic': 'FastAPI',
                'text': 'FastAPI is a modern, high-performance web framework for building APIs with Python 3.8+ based on standard Python type hints, Pydantic validation, and OpenAPI documentation.',
                'code_examples': [
                    '''from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(title="WebDev LLM API")

class CardRequest(BaseModel):
    topic: str = Field(..., description="Subject to explain")
    card_type: str = "concept"

@app.post("/generate-card")
async def generate_card(req: CardRequest):
    return {
        "topic": req.topic,
        "type": req.card_type,
        "title": f"{req.topic} Knowledge Card",
        "content": f"Essential overview and guide to {req.topic}."
    }'''
                ],
                'type': 'documentation'
            },
            # Databases & SQL
            {
                'source': 'Synthetic',
                'topic': 'SQL',
                'text': 'Relational databases store structured data in tables with relationships. SQL queries allow efficient querying, filtering, aggregation, and indexing for fast lookups.',
                'code_examples': [
                    '''-- Users and Projects Schema with Foreign Keys and Indexes
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE projects (
    id SERIAL PRIMARY KEY,
    user_id INT REFERENCES users(id) ON DELETE CASCADE,
    title VARCHAR(100) NOT NULL,
    status VARCHAR(20) DEFAULT 'active'
);

CREATE INDEX idx_projects_user ON projects(user_id);

SELECT u.username, COUNT(p.id) AS total_projects
FROM users u
LEFT JOIN projects p ON u.id = p.user_id
GROUP BY u.id, u.username
ORDER BY total_projects DESC;'''
                ],
                'type': 'example'
            },
            # MongoDB
            {
                'source': 'Synthetic',
                'topic': 'MongoDB',
                'text': 'MongoDB is a document-oriented NoSQL database that stores data in flexible, JSON-like BSON documents. It supports rich indexing and aggregation pipelines.',
                'code_examples': [
                    '''const mongoose = require('mongoose');

const CardSchema = new mongoose.Schema({
  topic: { type: String, required: true, index: true },
  title: { type: String, required: true },
  content: { type: String, required: true },
  codeExamples: [String],
  createdAt: { type: Date, default: Date.now }
});

const Card = mongoose.model('Card', CardSchema);
module.exports = Card;'''
                ],
                'type': 'documentation'
            },
            # Docker & DevOps
            {
                'source': 'Synthetic',
                'topic': 'Docker',
                'text': 'Docker packages applications and their dependencies into lightweight, portable containers, ensuring consistent runtime behavior across development, testing, and production environments.',
                'code_examples': [
                    '''# Production Multi-Stage Dockerfile
FROM node:20-alpine AS builder
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
RUN npm run build

FROM nginx:alpine
COPY --from=builder /app/dist /usr/share/nginx/html
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]'''
                ],
                'type': 'documentation'
            },
            # Git
            {
                'source': 'Synthetic',
                'topic': 'Git',
                'text': 'Git is a distributed version control system tracking changes in source code during software development, enabling branching, merging, and distributed collaboration.',
                'code_examples': [
                    '''# Feature Branch Workflow
git checkout -b feature/card-generator
git add .
git commit -m "feat: implement knowledge card synthesis module"
git push -u origin feature/card-generator
# Rebase on main
git checkout main
git pull origin main
git checkout feature/card-generator
git rebase main'''
                ],
                'type': 'documentation'
            },
            # REST & APIs
            {
                'source': 'Synthetic',
                'topic': 'REST API',
                'text': 'REST (Representational State Transfer) is an architectural style for networked hypermedia applications using standard HTTP methods (GET, POST, PUT, PATCH, DELETE) and status codes.',
                'code_examples': [
                    '''// HTTP Status Conventions
// 200 OK - Successful retrieval
// 201 Created - Resource created
// 400 Bad Request - Validation error
// 401 Unauthorized - Authentication missing
// 404 Not Found - Resource does not exist
// 500 Internal Server Error - Server failure'''
                ],
                'type': 'documentation'
            }
        ]
        
        self.collected_data.extend(synthetic_examples)
    
    def save_data(self, filename: str = 'raw_data.json'):
        """Save collected data to JSON file"""
        output_path = self.output_dir / filename
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.collected_data, f, ensure_ascii=False, indent=2)
        
        print(f"\nSaved {len(self.collected_data)} items to {output_path}")
    
    def get_statistics(self) -> Dict:
        """Get statistics about collected data"""
        total_items = len(self.collected_data)
        total_text_length = sum(len(item['text']) for item in self.collected_data)
        total_code_examples = sum(len(item.get('code_examples', [])) for item in self.collected_data)
        
        topics = {}
        for item in self.collected_data:
            topic = item.get('topic', 'Unknown')
            topics[topic] = topics.get(topic, 0) + 1
        
        return {
            'total_items': total_items,
            'total_text_length': total_text_length,
            'total_code_examples': total_code_examples,
            'topics': topics,
        }


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent / "src"))
    from config import DataConfig
    
    # Create collector
    collector = WebDevDataCollector(DataConfig.raw_data_dir)
    
    # Add synthetic examples (guaranteed high quality webdev dataset)
    collector.add_synthetic_examples()
    
    # Collect additional data from MDN if network is available
    print("Attempting MDN documentation fetch...")
    try:
        collector.collect_from_mdn(max_pages=5)
    except Exception as e:
        print(f"MDN fetch skipped or timed out ({e}). Continuing with rich synthetic dataset.")
    
    # Save data
    collector.save_data()
    
    # Print statistics
    stats = collector.get_statistics()
    print("\n=== Data Collection Statistics ===")
    print(f"Total items: {stats['total_items']}")
    print(f"Total text length: {stats['total_text_length']:,} characters")
    print(f"Total code examples: {stats['total_code_examples']}")
    print(f"\nTopics:")
    for topic, count in stats['topics'].items():
        print(f"  {topic}: {count}")
