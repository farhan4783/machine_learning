const { useState, useEffect, useRef, useMemo } = React;

// API Base URL - Uses current window origin if served by FastAPI or defaults to http://localhost:8000
const API_BASE = window.location.origin.includes(':8000') || window.location.origin.includes(':3000') || window.location.origin.includes(':5173')
  ? (window.location.port === '8000' ? '' : 'http://localhost:8000')
  : 'http://localhost:8000';

// ==========================================
// Reusable Markdown & Syntax Highlighting Component
// ==========================================
function MarkdownRenderer({ content }) {
  const containerRef = useRef(null);

  useEffect(() => {
    if (window.hljs && containerRef.current) {
      containerRef.current.querySelectorAll('pre code').forEach((block) => {
        window.hljs.highlightElement(block);
      });
    }
  }, [content]);

  const rawHtml = useMemo(() => {
    if (!content) return '';
    if (window.marked) {
      try {
        return window.marked.parse(content);
      } catch (e) {
        console.error("Markdown parse error:", e);
      }
    }
    return content.replace(/\n/g, '<br/>');
  }, [content]);

  return (
    <div 
      ref={containerRef}
      className="markdown-body-custom"
      dangerouslySetInnerHTML={{ __html: rawHtml }}
    />
  );
}

// ==========================================
// Root Application
// ==========================================
function App() {
  const [activeTab, setActiveTab] = useState('generator');
  const [backendStatus, setBackendStatus] = useState({ online: false, device: 'unknown', modelLoaded: false });
  const [modelInfo, setModelInfo] = useState(null);
  const [selectedTopicFromExplorer, setSelectedTopicFromExplorer] = useState(null);

  // Check backend health
  const checkHealth = async () => {
    try {
      const res = await fetch(`${API_BASE}/health`);
      if (res.ok) {
        const data = await res.json();
        setBackendStatus({ online: true, device: data.device, modelLoaded: data.model_loaded });
        
        // Fetch model info if model is loaded
        if (data.model_loaded) {
          const infoRes = await fetch(`${API_BASE}/model-info`);
          if (infoRes.ok) {
            const infoData = await infoRes.json();
            setModelInfo(infoData);
          }
        }
      } else {
        setBackendStatus({ online: false, device: 'offline', modelLoaded: false });
      }
    } catch (e) {
      setBackendStatus({ online: false, device: 'offline', modelLoaded: false });
    }
  };

  useEffect(() => {
    checkHealth();
    const interval = setInterval(checkHealth, 4000);
    return () => clearInterval(interval);
  }, []);

  const handleSelectTopic = (topicName) => {
    setSelectedTopicFromExplorer(topicName);
    setActiveTab('generator');
  };

  return (
    <div className="app-container">
      {/* Header Bar */}
      <header className="header">
        <div className="brand-group">
          <div className="logo-icon">⚡</div>
          <div>
            <div className="brand-title">WebDev LLM</div>
          </div>
          <span className="brand-badge">Transformer v1.0</span>
        </div>

        {/* Navigation Tabs */}
        <nav className="nav-tabs">
          <button 
            className={`nav-btn ${activeTab === 'generator' ? 'active' : ''}`}
            onClick={() => setActiveTab('generator')}
          >
            <span>🎴</span> Card Generator
          </button>
          <button 
            className={`nav-btn ${activeTab === 'topics' ? 'active' : ''}`}
            onClick={() => setActiveTab('topics')}
          >
            <span>📚</span> Topic Explorer
          </button>
          <button 
            className={`nav-btn ${activeTab === 'playground' ? 'active' : ''}`}
            onClick={() => setActiveTab('playground')}
          >
            <span>💻</span> AI Playground
          </button>
          <button 
            className={`nav-btn ${activeTab === 'diagnostics' ? 'active' : ''}`}
            onClick={() => setActiveTab('diagnostics')}
          >
            <span>⚙️</span> Diagnostics
          </button>
        </nav>

        {/* Backend Status Indicator */}
        <div className="status-badge" title={backendStatus.online ? `Connected to ${API_BASE || 'FastAPI Server'}` : 'FastAPI Server Not Reachable'}>
          <span className={`status-dot ${backendStatus.online ? 'online' : 'offline'}`}></span>
          <span>{backendStatus.online ? `API Online (${backendStatus.device.toUpperCase()})` : 'API Disconnected'}</span>
        </div>
      </header>

      {/* Main Content */}
      <main className="main-content">
        {activeTab === 'generator' && (
          <CardGeneratorView 
            initialTopic={selectedTopicFromExplorer} 
            isOnline={backendStatus.online}
          />
        )}
        {activeTab === 'topics' && (
          <TopicExplorerView onSelectTopic={handleSelectTopic} />
        )}
        {activeTab === 'playground' && (
          <PlaygroundView isOnline={backendStatus.online} />
        )}
        {activeTab === 'diagnostics' && (
          <DiagnosticsView backendStatus={backendStatus} modelInfo={modelInfo} />
        )}
      </main>
    </div>
  );
}

// ==========================================
// 1. Card Generator View
// ==========================================
function CardGeneratorView({ initialTopic, isOnline }) {
  const [topic, setTopic] = useState(initialTopic || 'React Hooks');
  const [cardType, setCardType] = useState('concept');
  const [temperature, setTemperature] = useState(0.7);
  const [maxLength, setMaxLength] = useState(384);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [generatedCard, setGeneratedCard] = useState(null);
  const [history, setHistory] = useState(() => {
    try {
      return JSON.parse(localStorage.getItem('webdev_cards_history') || '[]');
    } catch {
      return [];
    }
  });

  useEffect(() => {
    if (initialTopic) {
      setTopic(initialTopic);
    }
  }, [initialTopic]);

  const presetTopics = [
    "React Hooks", "CSS Flexbox", "JavaScript Promises", "FastAPI Dependency Injection",
    "Docker Multi-Stage", "SQL Indexing", "Next.js App Router", "TypeScript Generics"
  ];

  const handleGenerate = async (e) => {
    if (e) e.preventDefault();
    if (!topic.trim()) return;

    setLoading(true);
    setError(null);

    try {
      const response = await fetch(`${API_BASE}/generate-card`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          topic: topic.trim(),
          card_type: cardType,
          max_length: parseInt(maxLength),
          temperature: parseFloat(temperature),
        }),
      });

      if (!response.ok) {
        const errData = await response.json().catch(() => ({}));
        throw new Error(errData.detail || `Server error: ${response.status}`);
      }

      const card = await response.json();
      setGeneratedCard(card);

      // Save to history
      const newHistory = [card, ...history.filter(h => h.topic !== card.topic || h.type !== card.type)].slice(0, 10);
      setHistory(newHistory);
      localStorage.setItem('webdev_cards_history', JSON.stringify(newHistory));
    } catch (err) {
      setError(err.message || "Failed to generate knowledge card.");
    } finally {
      setLoading(false);
    }
  };

  const copyToClipboard = (text) => {
    navigator.clipboard.writeText(text);
    alert("Copied to clipboard!");
  };

  const downloadMarkdown = () => {
    if (!generatedCard) return;
    const md = `# ${generatedCard.title}\n\n**Topic:** ${generatedCard.topic}  \n**Type:** ${generatedCard.type}\n\n${generatedCard.content}`;
    const blob = new Blob([md], { type: 'text/markdown' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${generatedCard.topic.toLowerCase().replace(/\s+/g, '_')}_card.md`;
    a.click();
  };

  return (
    <div>
      <div className="hero">
        <h1 className="hero-title">Synthesize Web Dev Knowledge Cards</h1>
        <p className="hero-subtitle">
          Generate structured, educational cards powered by a custom Transformer LLM specialized in modern full-stack web development.
        </p>
      </div>

      <div className="generator-layout">
        {/* Controls Panel */}
        <div className="glass-panel">
          <h2 className="panel-title"><span>⚙️</span> Card Configuration</h2>
          <form onSubmit={handleGenerate}>
            <div className="form-group">
              <label className="form-label">Subject / Topic</label>
              <input 
                type="text" 
                className="form-input" 
                placeholder="e.g. React Hooks, CSS Grid, Express"
                value={topic}
                onChange={(e) => setTopic(e.target.value)}
                required
              />
              <div className="topic-pills">
                {presetTopics.map((p) => (
                  <button 
                    type="button" 
                    key={p} 
                    className="topic-pill"
                    onClick={() => setTopic(p)}
                  >
                    {p}
                  </button>
                ))}
              </div>
            </div>

            <div className="form-group">
              <label className="form-label">Card Type</label>
              <select 
                className="form-select"
                value={cardType}
                onChange={(e) => setCardType(e.target.value)}
              >
                <option value="concept">Core Concept & Explanation</option>
                <option value="code_example">Detailed Code Implementation</option>
                <option value="tutorial">Step-by-Step Guide</option>
                <option value="best_practices">Best Practices & Pitfalls</option>
                <option value="use_cases">Real-World Use Cases</option>
              </select>
            </div>

            <div className="form-group">
              <div className="slider-group">
                <label className="form-label" style={{margin:0}}>Sampling Temperature</label>
                <span>{temperature}</span>
              </div>
              <input 
                type="range" 
                min="0.1" 
                max="1.0" 
                step="0.05"
                value={temperature}
                onChange={(e) => setTemperature(e.target.value)}
                className="form-range"
              />
            </div>

            <div className="form-group">
              <div className="slider-group">
                <label className="form-label" style={{margin:0}}>Max Generation Tokens</label>
                <span>{maxLength}</span>
              </div>
              <input 
                type="range" 
                min="64" 
                max="512" 
                step="32"
                value={maxLength}
                onChange={(e) => setMaxLength(e.target.value)}
                className="form-range"
              />
            </div>

            <button 
              type="submit" 
              className="btn-primary"
              disabled={loading || !topic.trim()}
            >
              {loading ? (
                <>
                  <div className="spinner"></div>
                  <span>Generating Knowledge Card...</span>
                </>
              ) : (
                <>
                  <span>✨</span>
                  <span>Generate Knowledge Card</span>
                </>
              )}
            </button>
          </form>

          {/* History Sidebar */}
          {history.length > 0 && (
            <div style={{ marginTop: '2rem' }}>
              <div className="form-label" style={{ marginBottom: '0.75rem' }}>Recent Generations</div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.4rem' }}>
                {history.map((h, idx) => (
                  <button
                    key={idx}
                    className="btn-secondary"
                    style={{ justifyContent: 'space-between', width: '100%', fontSize: '0.8rem' }}
                    onClick={() => {
                      setTopic(h.topic);
                      setCardType(h.type);
                      setGeneratedCard(h);
                    }}
                  >
                    <span>{h.topic}</span>
                    <span style={{ opacity: 0.6, fontSize: '0.7rem' }}>{h.type}</span>
                  </button>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Card Display Panel */}
        <div className="glass-panel">
          {error && (
            <div style={{ padding: '1rem', background: 'rgba(239, 68, 68, 0.15)', border: '1px solid rgba(239, 68, 68, 0.4)', borderRadius: '12px', color: '#fca5a5', marginBottom: '1.5rem' }}>
              <strong>Error:</strong> {error}
              {!isOnline && (
                <div style={{ marginTop: '0.5rem', fontSize: '0.85rem' }}>
                  Make sure the FastAPI server is running with <code>uvicorn api.main:app --reload</code>.
                </div>
              )}
            </div>
          )}

          {loading ? (
            <div className="empty-state">
              <div className="spinner" style={{ width: '40px', height: '40px' }}></div>
              <div style={{ fontSize: '1.2rem', fontWeight: 600, color: '#f8fafc' }}>
                Synthesizing {topic} Card...
              </div>
              <p style={{ maxWidth: '400px', fontSize: '0.9rem' }}>
                Extracting domain concepts, synthesizing production-ready code examples and best practices.
              </p>
            </div>
          ) : generatedCard ? (
            <div className="card-display">
              <div className="card-header-bar">
                <div>
                  <h2 className="card-title-lg">{generatedCard.title}</h2>
                  <div className="card-meta">
                    <span className="card-badge">{generatedCard.type ? generatedCard.type.replace('_', ' ') : 'Concept'}</span>
                    <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
                      Topic: <strong>{generatedCard.topic}</strong>
                    </span>
                  </div>
                </div>
                <div className="card-actions">
                  <button className="btn-secondary" onClick={() => copyToClipboard(generatedCard.content)}>
                    <span>📋</span> Copy
                  </button>
                  <button className="btn-secondary" onClick={downloadMarkdown}>
                    <span>⬇️</span> Markdown
                  </button>
                </div>
              </div>

              <div className="card-body-content">
                <MarkdownRenderer content={generatedCard.content} />
              </div>
            </div>
          ) : (
            <div className="empty-state">
              <div className="empty-icon">🎴</div>
              <div style={{ fontSize: '1.25rem', fontWeight: 600, color: '#f8fafc' }}>
                No Card Generated Yet
              </div>
              <p style={{ maxWidth: '450px', fontSize: '0.9rem' }}>
                Select a topic on the left or enter any web development subject (HTML, CSS, React, TypeScript, APIs) and click <strong>Generate Knowledge Card</strong>.
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

// ==========================================
// 2. Topic Explorer View
// ==========================================
function TopicExplorerView({ onSelectTopic }) {
  const [topicsData, setTopicsData] = useState(null);
  const [search, setSearch] = useState('');
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch(`${API_BASE}/topics`)
      .then(res => res.json())
      .then(data => {
        setTopicsData(data);
        setLoading(false);
      })
      .catch(() => {
        // Fallback default topics if API offline
        setTopicsData({
          "Frontend": [
            { name: "HTML", description: "Semantic markup, accessibility, and modern DOM elements." },
            { name: "CSS", description: "Responsive layouts with Flexbox, CSS Grid, and custom variables." },
            { name: "JavaScript", description: "Asynchronous workflows, Event Loop, closures, and ES6+ features." },
            { name: "React", description: "Component-driven UI, state management, and modern React Hooks." },
            { name: "Vue", description: "Progressive JavaScript framework with reactive Composition API." },
            { name: "TypeScript", description: "Static typing, generics, interfaces, and union types." }
          ],
          "Backend": [
            { name: "Node.js", description: "V8-powered asynchronous JavaScript runtime for server applications." },
            { name: "Express", description: "Minimalist, robust REST API routing and middleware pipelines." },
            { name: "FastAPI", description: "Modern, high-performance Python web framework with Pydantic typing." },
            { name: "Django", description: "Batteries-included full-stack Python web framework with built-in ORM." }
          ],
          "Database": [
            { name: "SQL", description: "Relational database querying, joins, transactions, and indexing." },
            { name: "PostgreSQL", description: "Advanced open-source relational database with JSONB support." },
            { name: "MongoDB", description: "Scalable document-oriented NoSQL database with flexible BSON schemas." },
            { name: "Redis", description: "In-memory key-value data store for high-throughput caching & queues." }
          ],
          "DevOps & Architecture": [
            { name: "Docker", description: "Containerization platform for reproducible builds and multi-stage workflows." },
            { name: "Git", description: "Distributed version control system for collaborative software engineering." },
            { name: "REST API", description: "Representational State Transfer architectural principles and status codes." },
            { name: "CI/CD", description: "Automated continuous integration and deployment pipelines." }
          ]
        });
        setLoading(false);
      });
  }, []);

  if (loading) {
    return (
      <div className="empty-state">
        <div className="spinner"></div>
        <div>Loading topic directory...</div>
      </div>
    );
  }

  return (
    <div>
      <div className="hero">
        <h1 className="hero-title">Interactive Topic Directory</h1>
        <p className="hero-subtitle">
          Explore curated full-stack web development categories. Click on any topic to immediately synthesize knowledge cards.
        </p>
      </div>

      <div style={{ maxWidth: '600px', margin: '0 auto 2.5rem auto' }}>
        <input 
          type="text" 
          className="form-input" 
          placeholder="🔍 Search topics (e.g. React, SQL, Docker, FastAPI)..."
          value={search}
          onChange={(e) => setSearch(e.target.value)}
        />
      </div>

      <div className="topic-categories">
        {Object.entries(topicsData || {}).map(([category, topics]) => {
          const filteredTopics = topics.filter(t => 
            t.name.toLowerCase().includes(search.toLowerCase()) || 
            (t.description && t.description.toLowerCase().includes(search.toLowerCase()))
          );

          if (filteredTopics.length === 0) return null;

          return (
            <div key={category}>
              <h2 className="category-title">
                <span>{category === 'Frontend' ? '🎨' : category === 'Backend' ? '⚙️' : category === 'Database' ? '🗄️' : '🚀'}</span>
                {category}
              </h2>
              <div className="topics-grid">
                {filteredTopics.map((t) => (
                  <div 
                    key={t.name} 
                    className="topic-card"
                    onClick={() => onSelectTopic(t.name)}
                  >
                    <div>
                      <div className="topic-card-header">
                        <span className="topic-card-name">{t.name}</span>
                        <span className="card-badge" style={{ fontSize: '0.65rem' }}>{category}</span>
                      </div>
                      <p className="topic-card-desc">{t.description}</p>
                    </div>
                    <div className="topic-card-footer">
                      <span>Generate Card</span>
                      <span>→</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

// ==========================================
// 3. AI Playground View
// ==========================================
function PlaygroundView({ isOnline }) {
  const [prompt, setPrompt] = useState('Explain the difference between synchronous and asynchronous code in JavaScript:');
  const [maxLength, setMaxLength] = useState(384);
  const [temperature, setTemperature] = useState(0.7);
  const [loading, setLoading] = useState(false);
  const [generatedText, setGeneratedText] = useState('');
  const [error, setError] = useState(null);

  const presets = [
    { label: "Code Completion", text: "Complete this React function:\n\nfunction useLocalStorage(key, initialValue) {" },
    { label: "Code Explanation", text: "Explain how this Express middleware works:\n\nconst auth = (req, res, next) => { ... }" },
    { label: "Q&A Assistant", text: "User: What are the main benefits of using Docker in web development?\n\nAssistant:" },
    { label: "CSS Layout", text: "How do I center a div both horizontally and vertically with modern CSS Grid?" }
  ];

  const handleGenerateText = async () => {
    if (!prompt.trim()) return;
    setLoading(true);
    setError(null);

    try {
      const response = await fetch(`${API_BASE}/generate-text`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          prompt: prompt.trim(),
          max_length: parseInt(maxLength),
          temperature: parseFloat(temperature),
          top_k: 50,
          top_p: 0.95,
        }),
      });

      if (!response.ok) {
        const err = await response.json().catch(() => ({}));
        throw new Error(err.detail || `Server error: ${response.status}`);
      }

      const data = await response.json();
      setGeneratedText(data.generated_text);
    } catch (err) {
      setError(err.message || "Failed to generate text.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <div className="hero">
        <h1 className="hero-title">AI WebDev Playground & Assistant</h1>
        <p className="hero-subtitle">
          Test raw prompt completions, code explanations, and conversational Q&A directly with the specialized WebDev AI engine.
        </p>
      </div>

      <div className="generator-layout">
        {/* Input Panel */}
        <div className="glass-panel">
          <h2 className="panel-title"><span>✍️</span> Prompt Studio</h2>
          
          <div className="form-group">
            <label className="form-label">Presets</label>
            <div className="topic-pills">
              {presets.map((p, idx) => (
                <button 
                  key={idx} 
                  className="topic-pill"
                  onClick={() => setPrompt(p.text)}
                >
                  {p.label}
                </button>
              ))}
            </div>
          </div>

          <div className="form-group">
            <label className="form-label">Prompt / Code Snippet</label>
            <textarea 
              className="form-textarea" 
              rows="8"
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              placeholder="Enter code or question..."
            />
          </div>

          <div className="form-group">
            <div className="slider-group">
              <label className="form-label" style={{margin:0}}>Sampling Temperature</label>
              <span>{temperature}</span>
            </div>
            <input 
              type="range" 
              min="0.1" 
              max="1.0" 
              step="0.05"
              value={temperature}
              onChange={(e) => setTemperature(e.target.value)}
              className="form-range"
            />
          </div>

          <div className="form-group">
            <div className="slider-group">
              <label className="form-label" style={{margin:0}}>Max Output Tokens</label>
              <span>{maxLength}</span>
            </div>
            <input 
              type="range" 
              min="32" 
              max="512" 
              step="32"
              value={maxLength}
              onChange={(e) => setMaxLength(e.target.value)}
              className="form-range"
            />
          </div>

          <button 
            className="btn-primary"
            onClick={handleGenerateText}
            disabled={loading || !prompt.trim()}
          >
            {loading ? <div className="spinner"></div> : <span>⚡ Run Inference</span>}
          </button>
        </div>

        {/* Output Panel */}
        <div className="glass-panel">
          <h2 className="panel-title"><span>💬</span> Output Response</h2>
          
          {error && (
            <div style={{ padding: '1rem', background: 'rgba(239, 68, 68, 0.15)', border: '1px solid rgba(239, 68, 68, 0.4)', borderRadius: '12px', color: '#fca5a5', marginBottom: '1.5rem' }}>
              <strong>Error:</strong> {error}
            </div>
          )}

          {loading ? (
            <div className="empty-state">
              <div className="spinner" style={{ width: '40px', height: '40px' }}></div>
              <div>Computing next-token probabilities & synthesizing...</div>
            </div>
          ) : generatedText ? (
            <div className="code-block" style={{ margin: 0, minHeight: '300px' }}>
              <div className="code-header">
                <span>Model Output</span>
                <button 
                  className="btn-secondary" 
                  style={{ padding: '0.2rem 0.6rem', fontSize: '0.75rem' }}
                  onClick={() => {
                    navigator.clipboard.writeText(generatedText);
                    alert("Copied to clipboard!");
                  }}
                >
                  Copy Text
                </button>
              </div>
              <div style={{ padding: '1.25rem' }}>
                <MarkdownRenderer content={generatedText} />
              </div>
            </div>
          ) : (
            <div className="empty-state">
              <div className="empty-icon">💻</div>
              <div>Ready for inference</div>
              <p style={{ fontSize: '0.875rem' }}>Click <strong>Run Inference</strong> to test text generation.</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

// ==========================================
// 4. Model Diagnostics View
// ==========================================
function DiagnosticsView({ backendStatus, modelInfo }) {
  return (
    <div>
      <div className="hero">
        <h1 className="hero-title">Model Architecture & Diagnostics</h1>
        <p className="hero-subtitle">
          Technical specifications, neural network architecture, and runtime status of the WebDev LLM engine.
        </p>
      </div>

      <div className="diag-grid">
        <div className="diag-card">
          <div className="diag-label">Model Architecture</div>
          <div className="diag-value" style={{ fontSize: '1.6rem' }}>Decoder LLM</div>
          <p style={{ color: 'var(--text-secondary)', fontSize: '0.8rem' }}>Transformer with RoPE + SwiGLU</p>
        </div>

        <div className="diag-card">
          <div className="diag-label">Trainable Parameters</div>
          <div className="diag-value">
            {modelInfo ? `${(modelInfo.parameters / 1000000).toFixed(2)}M` : '5.72M'}
          </div>
          <p style={{ color: 'var(--text-secondary)', fontSize: '0.8rem' }}>Configurable scaling</p>
        </div>

        <div className="diag-card">
          <div className="diag-label">Vocabulary Size</div>
          <div className="diag-value">
            {modelInfo ? modelInfo.vocab_size : '5,000'}
          </div>
          <p style={{ color: 'var(--text-secondary)', fontSize: '0.8rem' }}>Custom BPE Tokenizer</p>
        </div>

        <div className="diag-card">
          <div className="diag-label">Compute Device</div>
          <div className="diag-value" style={{ fontSize: '1.8rem', color: '#10b981' }}>
            {backendStatus.device ? backendStatus.device.toUpperCase() : 'CPU'}
          </div>
          <p style={{ color: 'var(--text-secondary)', fontSize: '0.8rem' }}>PyTorch Execution Backend</p>
        </div>
      </div>

      <div className="glass-panel" style={{ marginBottom: '2rem' }}>
        <h2 className="panel-title"><span>🧬</span> Modern Transformer Innovations</h2>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '1.5rem', marginTop: '1rem' }}>
          <div style={{ padding: '1rem', background: 'rgba(255,255,255,0.02)', borderRadius: '12px', border: '1px solid var(--border-subtle)' }}>
            <h3 style={{ color: '#818cf8', marginBottom: '0.4rem' }}>Rotary Position Embeddings (RoPE)</h3>
            <p style={{ fontSize: '0.875rem', color: 'var(--text-secondary)', lineHeight: 1.5 }}>
              Encodes relative token distances using complex rotation matrices instead of fixed absolute embeddings, enhancing long context extrapolation.
            </p>
          </div>

          <div style={{ padding: '1rem', background: 'rgba(255,255,255,0.02)', borderRadius: '12px', border: '1px solid var(--border-subtle)' }}>
            <h3 style={{ color: '#ec4899', marginBottom: '0.4rem' }}>RMSNorm & SwiGLU</h3>
            <p style={{ fontSize: '0.875rem', color: 'var(--text-secondary)', lineHeight: 1.5 }}>
              Implements Root Mean Square normalization for training stability alongside Swish-Gated linear units for expressive non-linear representations.
            </p>
          </div>

          <div style={{ padding: '1rem', background: 'rgba(255,255,255,0.02)', borderRadius: '12px', border: '1px solid var(--border-subtle)' }}>
            <h3 style={{ color: '#06b6d4', marginBottom: '0.4rem' }}>KV Caching</h3>
            <p style={{ fontSize: '0.875rem', color: 'var(--text-secondary)', lineHeight: 1.5 }}>
              Maintains Key-Value state in memory across generation steps, eliminating redundant computations and enabling real-time token streaming.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}

// Render React App
ReactDOM.render(<App />, document.getElementById('root'));
