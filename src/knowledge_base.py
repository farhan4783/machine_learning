"""
Comprehensive Web Development Knowledge Base
Contains curated, production-grade knowledge, syntax guides, code examples, best practices, and architecture patterns.
"""

from typing import Dict, List, Optional, Any
import re

WEBDEV_KNOWLEDGE: Dict[str, Dict[str, Any]] = {
    # -------------------------------------------------------------
    # FRONTEND TOPICS
    # -------------------------------------------------------------
    "react": {
        "title": "React.js Component Architecture & Hooks",
        "category": "Frontend",
        "description": "React is a declarative, efficient component-based JavaScript library for building interactive user interfaces.",
        "concept": """React structures web applications into reusable, encapsulated components that manage their own state. It utilizes a Virtual DOM to perform minimal reconciliation operations, ensuring optimal rendering performance.

Key architectural concepts include:
- **Unidirectional Data Flow**: Data flows down via props; events flow up via callbacks.
- **Component Lifecycle & Hooks**: Hooks allow functional components to hook into state and lifecycle features without writing classes.
- **Virtual DOM Reconciliation**: React computes diffs in a lightweight virtual tree before batching updates to the real browser DOM.
- **JSX**: A declarative syntax extension combining HTML-like markup with the full expressive power of JavaScript.""",
        "code_example": """import React, { useState, useEffect, useMemo, useCallback } from 'react';

// Custom Hook for Window Resizing
function useWindowWidth() {
  const [width, setWidth] = useState(window.innerWidth);

  useEffect(() => {
    const handleResize = () => setWidth(window.innerWidth);
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  return width;
}

// Interactive Task List Component
export function TaskManager({ initialTasks = [] }) {
  const [tasks, setTasks] = useState(initialTasks);
  const [input, setInput] = useState('');
  const [filter, setFilter] = useState('all');
  const width = useWindowWidth();

  const addTask = useCallback(() => {
    if (!input.trim()) return;
    const newTask = { id: Date.now(), text: input.trim(), completed: false };
    setTasks(prev => [newTask, ...prev]);
    setInput('');
  }, [input]);

  const toggleTask = (id) => {
    setTasks(prev => prev.map(t => t.id === id ? { ...t, completed: !t.completed } : t));
  };

  const filteredTasks = useMemo(() => {
    if (filter === 'completed') return tasks.filter(t => t.completed);
    if (filter === 'active') return tasks.filter(t => !t.completed);
    return tasks;
  }, [tasks, filter]);

  return (
    <div className="task-container" style={{ padding: width < 600 ? '1rem' : '2rem' }}>
      <h2>React Task Manager ({tasks.length})</h2>
      <div className="input-group">
        <input 
          value={input} 
          onChange={(e) => setInput(e.target.value)}
          placeholder="Enter new task..."
          onKeyDown={(e) => e.key === 'Enter' && addTask()}
        />
        <button onClick={addTask}>Add Task</button>
      </div>

      <div className="filters">
        <button onClick={() => setFilter('all')}>All</button>
        <button onClick={() => setFilter('active')}>Active</button>
        <button onClick={() => setFilter('completed')}>Completed</button>
      </div>

      <ul className="task-list">
        {filteredTasks.map(task => (
          <li 
            key={task.id} 
            onClick={() => toggleTask(task.id)}
            style={{ textDecoration: task.completed ? 'line-through' : 'none' }}
          >
            {task.text}
          </li>
        ))}
      </ul>
    </div>
  );
}""",
        "best_practices": [
            "Keep state as local as possible; lift state only when shared across siblings.",
            "Always include all reactive dependencies in useEffect and useCallback dependency arrays.",
            "Use memoization (useMemo, useCallback, React.memo) deliberately, not preemptively.",
            "Avoid mutating state directly; always use immutable update patterns (e.g. spread operator or immer).",
            "Use unique, stable IDs as list keys rather than array indexes."
        ],
        "common_pitfalls": [
            "Missing dependency array in useEffect causing infinite re-render loops.",
            "Mutating state in-place (e.g. state.push()) which prevents React from detecting state changes.",
            "Calling Hooks inside conditional statements or nested loops instead of top-level.",
            "Overusing Context for rapidly changing state, causing widespread re-renders."
        ],
        "use_cases": [
            "Single Page Applications (SPAs) with complex dynamic user workflows.",
            "Component design systems and enterprise dashboards.",
            "Real-time collaborative tools and social media feeds."
        ]
    },

    "react hooks": {
        "title": "Mastering React Hooks (useState, useEffect, useMemo, useCallback)",
        "category": "Frontend",
        "description": "Deep dive into React Hooks for state management, side effects, memoization, and custom logic encapsulation.",
        "concept": """React Hooks were introduced in React 16.8 to enable state and lifecycle management in functional components without class syntax.

Core Hooks:
1. **useState**: Manages local component state.
2. **useEffect**: Handles side-effects (data fetching, DOM subscriptions, timers, event listeners).
3. **useContext**: Consumes values from React Context without wrapping render props.
4. **useRef**: Persists mutable values across renders without triggering a re-render.
5. **useMemo & useCallback**: Memoizes expensive computations and function references across renders.
6. **useReducer**: Alternative to useState for complex state transitions following the Redux action-reducer pattern.""",
        "code_example": """import React, { useState, useEffect, useRef, useReducer } from 'react';

// Reducer for complex state logic
function cartReducer(state, action) {
  switch (action.type) {
    case 'ADD_ITEM': {
      const existing = state.find(i => i.id === action.payload.id);
      if (existing) {
        return state.map(i => i.id === action.payload.id ? { ...i, qty: i.qty + 1 } : i);
      }
      return [...state, { ...action.payload, qty: 1 }];
    }
    case 'REMOVE_ITEM':
      return state.filter(i => i.id !== action.payload.id);
    case 'CLEAR':
      return [];
    default:
      return state;
  }
}

export function ShoppingCart() {
  const [items, dispatch] = useReducer(cartReducer, []);
  const renderCount = useRef(0);
  renderCount.current += 1;

  const total = items.reduce((sum, item) => sum + item.price * item.qty, 0);

  return (
    <div className="cart-card">
      <h3>Shopping Cart (Renders: {renderCount.current})</h3>
      <button onClick={() => dispatch({ type: 'ADD_ITEM', payload: { id: 1, name: 'Pro Keyboard', price: 99 } })}>
        Add Keyboard ($99)
      </button>
      <button onClick={() => dispatch({ type: 'ADD_ITEM', payload: { id: 2, name: 'Wireless Mouse', price: 49 } })}>
        Add Mouse ($49)
      </button>

      <ul>
        {items.map(item => (
          <li key={item.id}>
            {item.name} x {item.qty} = ${item.price * item.qty}
            <button onClick={() => dispatch({ type: 'REMOVE_ITEM', payload: { id: item.id } })}>Remove</button>
          </li>
        ))}
      </ul>
      <p><strong>Total: ${total}</strong></p>
    </div>
  );
}""",
        "best_practices": [
            "Follow the Rules of Hooks: Call Hooks only at the top level, never inside conditions or loops.",
            "Create custom hooks to extract and reuse stateful logic across multiple components.",
            "Always return a cleanup function in useEffect to abort fetch requests or remove listeners.",
            "Use useReducer when subsequent state updates depend on multiple sub-values or complex logic."
        ],
        "common_pitfalls": [
            "Stale closures in callbacks when dependencies are missing from dependency arrays.",
            "Triggering state updates in unmounted components causing memory leak warnings.",
            "Over-optimizing with useMemo/useCallback on trivial computations, increasing overhead."
        ],
        "use_cases": [
            "Form state handling, debounced inputs, data polling, and websocket connections.",
            "Custom audio/video players, infinite scroll pagination, and theme toggling."
        ]
    },

    "css flexbox": {
        "title": "Modern CSS Flexbox Layout Architecture",
        "category": "Frontend",
        "description": "A comprehensive guide to 1-dimensional layout distribution, alignment, and responsive flow using CSS Flexbox.",
        "concept": """CSS Flexible Box Layout (Flexbox) provides a powerful, predictable way to align and distribute space among items in a container, even when their sizes are dynamic or unknown.

Core Axes:
- **Main Axis**: Defined by `flex-direction` (`row`, `row-reverse`, `column`, `column-reverse`).
- **Cross Axis**: Perpendicular to the main axis.

Key Container Properties:
- `display: flex | inline-flex`
- `justify-content`: `flex-start | center | flex-end | space-between | space-around | space-evenly`
- `align-items`: `stretch | center | flex-start | flex-end | baseline`
- `flex-wrap`: `nowrap | wrap | wrap-reverse`
- `gap`: Sets gutter spacing along both row and column axes without margin hacks.""",
        "code_example": """/* Responsive Navigation & Modern Card Grid with Flexbox */

.navbar {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 1rem 2rem;
  background: rgba(15, 23, 42, 0.9);
  backdrop-filter: blur(10px);
}

.nav-links {
  display: flex;
  align-items: center;
  gap: 1.5rem;
  list-style: none;
}

/* Card Container with Auto-wrapping and Equal Height Columns */
.card-deck {
  display: flex;
  flex-wrap: wrap;
  gap: 1.5rem;
  padding: 2rem;
}

.card-item {
  display: flex;
  flex-direction: column;
  flex: 1 1 300px; /* flex-grow: 1, flex-shrink: 1, flex-basis: 300px */
  background: #1e293b;
  border-radius: 12px;
  padding: 1.5rem;
  border: 1px solid rgba(255, 255, 255, 0.1);
}

.card-body {
  flex-grow: 1; /* Pushes card footer to the bottom */
  margin-bottom: 1rem;
}

.card-footer {
  display: flex;
  justify-content: space-between;
  align-items: center;
}""",
        "best_practices": [
            "Use the `gap` property instead of margin calculations for spacing child items.",
            "Use Flexbox for 1D layouts (rows OR columns) and CSS Grid for 2D layouts (rows AND columns).",
            "Use `margin-left: auto` or `margin-top: auto` on child elements for flexible spacing shortcuts.",
            "Always specify `flex-wrap: wrap` when building responsive layouts that must accommodate smaller viewports."
        ],
        "common_pitfalls": [
            "Setting explicit widths on flex items instead of using `flex-basis` and `min-width`.",
            "Confusing `justify-content` (main axis) with `align-items` (cross axis).",
            "Assuming flex items respect standard box-sizing without setting `min-width: 0` on shrinking items with long text."
        ],
        "use_cases": [
            "Navigation bars, button toolbars, hero headers, modal action footers, and responsive card decks."
        ]
    },

    "javascript promises": {
        "title": "JavaScript Asynchronous Programming: Promises & Async/Await",
        "category": "Frontend",
        "description": "Mastering asynchronous workflows, concurrency management with Promise.all, error handling, and the Event Loop.",
        "concept": """JavaScript is single-threaded and relies on an event-driven runtime (Event Loop, Call Stack, Microtask Queue, Macrotask Queue). Promises represent values that may be available now, in the future, or never.

A Promise has three states:
1. **Pending**: Initial state, neither fulfilled nor rejected.
2. **Fulfilled**: The asynchronous operation completed successfully.
3. **Rejected**: The operation failed with an error.

`async/await` is syntactic sugar over Promises that enables developers to write non-blocking asynchronous code with synchronous readability and native `try/catch` error management.""",
        "code_example": """// Production Data Fetching with Timeout, Retry, and Parallelism

async function fetchWithRetry(url, options = {}, retries = 3, delay = 1000) {
  for (let attempt = 1; attempt <= retries; attempt++) {
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 5000); // 5s timeout

      const response = await fetch(url, { ...options, signal: controller.signal });
      clearTimeout(timeoutId);

      if (!response.ok) {
        throw new Error(`HTTP Error ${response.status}: ${response.statusText}`);
      }

      return await response.json();
    } catch (err) {
      const isLastAttempt = attempt === retries;
      if (isLastAttempt) throw err;
      
      console.warn(`Attempt ${attempt} failed. Retrying in ${delay}ms...`);
      await new Promise(resolve => setTimeout(resolve, delay * attempt));
    }
  }
}

// Parallel Concurrency with Promise.allSettled
async function loadDashboardData(userId) {
  try {
    const [userResult, postsResult, statsResult] = await Promise.allSettled([
      fetchWithRetry(`/api/users/${userId}`),
      fetchWithRetry(`/api/users/${userId}/posts`),
      fetchWithRetry(`/api/users/${userId}/analytics`)
    ]);

    return {
      user: userResult.status === 'fulfilled' ? userResult.value : null,
      posts: postsResult.status === 'fulfilled' ? postsResult.value : [],
      stats: statsResult.status === 'fulfilled' ? statsResult.value : {},
    };
  } catch (error) {
    console.error('Critical Dashboard Failure:', error);
    throw error;
  }
}""",
        "best_practices": [
            "Always handle rejections using `try/catch` with `async/await` or `.catch()` chains.",
            "Use `Promise.allSettled` when you want all parallel tasks to execute regardless of individual failures.",
            "Use `Promise.all` only when all operations are mutually dependent and must succeed together.",
            "Always include timeout handling (e.g. `AbortController`) to prevent hung network requests."
        ],
        "common_pitfalls": [
            "Executing independent async calls sequentially in a `for` loop instead of in parallel via `Promise.all`.",
            "Forgetting to `await` a promise, leading to unhandled promise rejections or unexpected timing bugs.",
            "Creating 'Callback Hell' inside `.then()` chains instead of refactoring to clean `async/await` blocks."
        ],
        "use_cases": [
            "REST and GraphQL API consumption, multi-file uploads, database transactions, background sync."
        ]
    },

    "typescript": {
        "title": "TypeScript Type Systems & Advanced Generics",
        "category": "Frontend",
        "description": "Static type checking, interfaces, generic constraints, union discrimination, and utility types for robust full-stack code.",
        "concept": """TypeScript adds static typing to JavaScript to detect bugs at compile-time, enable intelligent IDE autocompletion, and provide reliable refactoring capabilities.

Core Features:
- **Interfaces & Type Aliases**: Define complex object shapes and function signatures.
- **Generics**: Write reusable components and functions that operate over a variety of types.
- **Discriminated Unions**: Pattern matching with a shared literal tag for type-safe state machines.
- **Utility Types**: `Partial<T>`, `Pick<T, K>`, `Omit<T, K>`, `Record<K, T>`, `ReturnType<T>`.""",
        "code_example": """// Discriminated Union for State Machine
type AsyncState<T> =
  | { status: 'idle'; data: null; error: null }
  | { status: 'loading'; data: null; error: null }
  | { status: 'success'; data: T; error: null }
  | { status: 'error'; data: null; error: string };

// Generic API Service Client
interface ApiResponse<T> {
  data: T;
  statusCode: number;
  timestamp: string;
}

class ApiClient {
  constructor(private baseUrl: string) {}

  async get<T>(endpoint: string): Promise<ApiResponse<T>> {
    const res = await fetch(`${this.baseUrl}${endpoint}`);
    if (!res.ok) {
      throw new Error(`API Error: ${res.statusText}`);
    }
    const data: T = await res.json();
    return {
      data,
      statusCode: res.status,
      timestamp: new Date().toISOString(),
    };
  }
}

// User Entity Type
interface User {
  id: string;
  username: string;
  email: string;
  role: 'admin' | 'developer' | 'viewer';
}

// Example usage
async function run() {
  const client = new ApiClient('https://api.example.com');
  const response = await client.get<User[]>('/users');
  console.log(response.data[0].role); // Fully type-safe autocomplete
}""",
        "best_practices": [
            "Enable `strict: true` in `tsconfig.json` for full type safety.",
            "Avoid using `any`; prefer `unknown` when type is indeterminate and narrow with type guards.",
            "Use Discriminated Unions for handling complex UI state and API responses.",
            "Use utility types (`Omit`, `Pick`, `Record`) to avoid duplicating interface definitions."
        ],
        "common_pitfalls": [
            "Over-using type assertions (`as unknown as Type`) which bypass compiler safety checks.",
            "Creating excessively complex nested conditional types that slow down compiler performance."
        ],
        "use_cases": [
            "Enterprise frontend and backend applications, open-source libraries, microservice DTO contracts."
        ]
    },

    "fastapi": {
        "title": "High-Performance Python REST APIs with FastAPI",
        "category": "Backend",
        "description": "Modern, fast (high-performance) web framework for building APIs with Python 3.8+ based on standard Python type hints.",
        "concept": """FastAPI is built on Starlette (for web routing) and Pydantic (for data validation and serialization). It delivers exceptional performance comparable to Node.js and Go.

Key Capabilities:
- **Automatic Validation**: Validates request body, query parameters, and headers using Pydantic models.
- **Dependency Injection**: Reusable dependencies for authentication, database sessions, and permissions.
- **Async/Await Native**: Native concurrency support with Python ASGI.
- **Automatic OpenAPI Docs**: Generates interactive Swagger UI (`/docs`) and ReDoc (`/redoc`) out of the box.""",
        "code_example": """from fastapi import FastAPI, Depends, HTTPException, status, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, EmailStr
from typing import List, Optional
import time

app = FastAPI(
    title="WebDev Knowledge Hub API",
    version="1.0.0",
    description="Production REST API for Knowledge Cards and Text Generation"
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic Schemas
class CardCreate(BaseModel):
    topic: str = Field(..., min_length=2, max_length=100, example="React Hooks")
    card_type: str = Field(default="concept", example="code_example")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)

class CardResponse(BaseModel):
    id: int
    topic: str
    type: str
    title: str
    content: str
    created_at: float

# In-memory storage mock
cards_db: List[CardResponse] = []

@app.post("/cards", response_model=CardResponse, status_code=status.HTTP_201_CREATED)
async def create_card(card_in: CardCreate):
    new_card = CardResponse(
        id=len(cards_db) + 1,
        topic=card_in.topic,
        type=card_type,
        title=f"{card_in.topic} - {card_in.card_type.title()}",
        content=f"Comprehensive guide and best practices for {card_in.topic}.",
        created_at=time.time()
    )
    cards_db.append(new_card)
    return new_card

@app.get("/cards", response_model=List[CardResponse])
async def list_cards(limit: int = Query(10, ge=1, le=100)):
    return cards_db[:limit]

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "fastapi-engine"}""",
        "best_practices": [
            "Use Pydantic V2 schemas for strict input validation and response filtering.",
            "Use `Depends()` for database session lifecycle management and JWT user authentication.",
            "Structure endpoints using `APIRouter` in separate modular files for maintainability.",
            "Leverage `async def` for I/O bound tasks (database queries, network requests)."
        ],
        "common_pitfalls": [
            "Using blocking I/O calls (e.g. `time.sleep()`, synchronous DB drivers) inside `async def` routes.",
            "Failing to handle exceptions properly with custom HTTP exception handlers."
        ],
        "use_cases": [
            "AI/ML model serving, real-time microservices, mobile backends, high-throughput APIs."
        ]
    },

    "docker": {
        "title": "Containerization & DevOps Workflows with Docker",
        "category": "DevOps",
        "description": "Packaging applications, dependencies, and runtime environments into lightweight, deterministic containers.",
        "concept": """Docker isolates applications into containers that run consistently across development, staging, and production environments.

Core Concepts:
- **Dockerfile**: Blueprint instructions to build a container image.
- **Image**: An immutable, layered snapshot containing application code, runtime, and OS libraries.
- **Container**: A running instance of an image with isolated filesystem and network namespaces.
- **Multi-Stage Builds**: Optimize production image sizes by separating compilation/build tooling from minimal runtime environments.""",
        "code_example": """# Multi-Stage Dockerfile for a React + Python FastAPI Web Application

# Stage 1: Build Frontend Assets
FROM node:20-alpine AS frontend-builder
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build || echo "Static assets ready"

# Stage 2: Production Python Backend Container
FROM python:3.11-slim AS production

# Set environment variables
ENV PYTHONUNBUFFERED=1 \\
    PYTHONDONTWRITEBYTECODE=1 \\
    PORT=8000

WORKDIR /app

# Install OS dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend application code
COPY api/ ./api/
COPY src/ ./src/
COPY data/ ./data/
COPY models/ ./models/

# Copy built frontend assets from stage 1
COPY --from=frontend-builder /app/frontend ./frontend

# Create non-root user for security
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Healthcheck
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \\
  CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]""",
        "best_practices": [
            "Use `.dockerignore` to exclude `node_modules`, `.git`, `venv`, and build artifacts.",
            "Use official slim or alpine base images to minimize security vulnerabilities and image sizes.",
            "Order Dockerfile instructions from least frequently changed to most frequently changed to maximize layer caching.",
            "Never run containers as root in production; create a dedicated user."
        ],
        "common_pitfalls": [
            "Installing build tools into the final runtime image instead of using multi-stage builds.",
            "Hardcoding secrets (API keys, passwords) in Dockerfiles instead of using environment variables or Docker secrets."
        ],
        "use_cases": [
            "Microservice architectures, CI/CD automated test pipelines, Kubernetes deployments."
        ]
    },

    "sql": {
        "title": "SQL Relational Database Design & Query Optimization",
        "category": "Database",
        "description": "Relational data modeling, ACID transactions, complex joins, indexing strategies, and query performance tuning.",
        "concept": """Relational Database Management Systems (RDBMS) like PostgreSQL, MySQL, and SQLite organize data into tables with predefined schemas and relationships.

Key Concepts:
- **ACID Properties**: Atomicity, Consistency, Isolation, Durability guarantee transaction integrity.
- **Normalization (1NF to 3NF)**: Eliminating data redundancy while preserving integrity.
- **B-Tree & Hash Indexes**: Speeding up search and filtering queries from $O(N)$ full table scans to $O(\\log N)$ index lookups.
- **Joins**: `INNER JOIN`, `LEFT JOIN`, `RIGHT JOIN`, `FULL OUTER JOIN` for relational combinations.""",
        "code_example": """-- Production Schema with Foreign Keys, Constraints, and Indexes

CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE knowledge_cards (
    id SERIAL PRIMARY KEY,
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    topic VARCHAR(100) NOT NULL,
    card_type VARCHAR(50) NOT NULL,
    title VARCHAR(255) NOT NULL,
    content TEXT NOT NULL,
    view_count INT DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Performance Indexes
CREATE INDEX idx_cards_topic ON knowledge_cards(topic);
CREATE INDEX idx_cards_user_created ON knowledge_cards(user_id, created_at DESC);

-- Analytics Query with Aggregation and Window Functions
SELECT 
    u.username,
    COUNT(c.id) AS total_cards,
    SUM(c.view_count) AS total_views,
    AVG(c.view_count)::NUMERIC(10,2) AS avg_views_per_card
FROM users u
LEFT JOIN knowledge_cards c ON u.id = c.user_id
GROUP BY u.id, u.username
HAVING COUNT(c.id) > 0
ORDER BY total_views DESC
LIMIT 10;""",
        "best_practices": [
            "Add indexes to foreign keys and columns frequently used in `WHERE`, `JOIN`, and `ORDER BY` clauses.",
            "Use database transactions (`BEGIN ... COMMIT`) for multi-step operations that must succeed or fail together.",
            "Avoid `SELECT *`; explicitly request only the required columns.",
            "Use parameter binding / prepared statements to prevent SQL Injection attacks."
        ],
        "common_pitfalls": [
            "Over-indexing tables which slows down `INSERT`, `UPDATE`, and `DELETE` operations.",
            "The N+1 Query Problem in ORMs: querying relations inside loops instead of eager loading joins."
        ],
        "use_cases": [
            "E-commerce orders, user identity and billing, financial ledgers, transactional applications."
        ]
    }
}


def find_knowledge_topic(query: str) -> Optional[Dict[str, Any]]:
    """Match a user query against the curated knowledge base"""
    q_clean = query.strip().lower()
    
    # Direct match
    if q_clean in WEBDEV_KNOWLEDGE:
        return WEBDEV_KNOWLEDGE[q_clean]
    
    # Keyword search
    for key, data in WEBDEV_KNOWLEDGE.items():
        if key in q_clean or q_clean in key or q_clean in data["title"].lower():
            return data
            
    # Fuzzy keyword search
    for key, data in WEBDEV_KNOWLEDGE.items():
        tokens = key.split()
        if any(token in q_clean for token in tokens if len(token) > 2):
            return data
            
    return None
