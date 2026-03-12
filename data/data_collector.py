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
            # React examples
            {
                'source': 'Synthetic',
                'topic': 'React',
                'text': 'React is a JavaScript library for building user interfaces. Components are the building blocks of React applications.',
                'code_examples': [
                    '''function Welcome(props) {
  return <h1>Hello, {props.name}</h1>;
}''',
                    '''const App = () => {
  const [count, setCount] = useState(0);
  return (
    <div>
      <p>Count: {count}</p>
      <button onClick={() => setCount(count + 1)}>Increment</button>
    </div>
  );
};'''
                ],
                'type': 'example'
            },
            
            # CSS examples
            {
                'source': 'Synthetic',
                'topic': 'CSS',
                'text': 'CSS Flexbox is a layout model that allows responsive elements within a container to be automatically arranged.',
                'code_examples': [
                    '''.container {
  display: flex;
  justify-content: center;
  align-items: center;
  gap: 1rem;
}''',
                    '''.grid-container {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 20px;
}'''
                ],
                'type': 'example'
            },
            
            # JavaScript examples
            {
                'source': 'Synthetic',
                'topic': 'JavaScript',
                'text': 'JavaScript async/await makes asynchronous code look and behave like synchronous code.',
                'code_examples': [
                    '''async function fetchData() {
  try {
    const response = await fetch('https://api.example.com/data');
    const data = await response.json();
    return data;
  } catch (error) {
    console.error('Error:', error);
  }
}''',
                    '''const getData = async () => {
  const data = await fetchData();
  console.log(data);
};'''
                ],
                'type': 'example'
            },
            
            # Node.js examples
            {
                'source': 'Synthetic',
                'topic': 'Node.js',
                'text': 'Express.js is a minimal and flexible Node.js web application framework.',
                'code_examples': [
                    '''const express = require('express');
const app = express();

app.get('/', (req, res) => {
  res.send('Hello World!');
});

app.listen(3000, () => {
  console.log('Server running on port 3000');
});''',
                    '''app.post('/api/users', async (req, res) => {
  try {
    const user = await User.create(req.body);
    res.status(201).json(user);
  } catch (error) {
    res.status(400).json({ error: error.message });
  }
});'''
                ],
                'type': 'example'
            },
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
    from config import DataConfig
    
    # Create collector
    collector = WebDevDataCollector(DataConfig.raw_data_dir)
    
    # Collect data (extensive scrape)
    print("Starting extensive data collection... This may take several minutes.")
    collector.collect_from_mdn(max_pages=1000)
    
    # Add synthetic examples for testing
    collector.add_synthetic_examples()
    
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
