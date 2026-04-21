import { useState } from 'react'
import './App.css'


const API_BASE = 'http://localhost:9000'

// ── Endpoint definitions ──────────────────────────────────────────────────────

type FieldDef = {
  name: string
  label: string
  type: 'text' | 'number' | 'select' | 'checkbox'
  default: string
  options?: string[]
}

type EndpointDef = {
  method: 'GET' | 'POST'
  path: string
  description: string
  fields?: FieldDef[]
  buildUrl: (vals: Record<string, string>) => string
  buildBody?: (vals: Record<string, string>) => unknown
}

const ENDPOINTS: EndpointDef[] = [
  {
    method: 'GET',
    path: '/',
    description: 'Health check — returns corpus size and index status.',
    buildUrl: () => `${API_BASE}/`,
  },
  {
    method: 'GET',
    path: '/movies',
    description: 'List all indexed movies, sorted alphabetically.',
    buildUrl: () => `${API_BASE}/movies`,
  },
  {
    method: 'GET',
    path: '/movies/{movie_id}',
    description: 'Get metadata for a single movie by its TMDB ID.',
    fields: [
      { name: 'movie_id', label: 'Movie ID (TMDB)', type: 'text', default: '680' },
    ],
    buildUrl: (v) => `${API_BASE}/movies/${v.movie_id}`,
  },
  {
    method: 'POST',
    path: '/classify',
    description: 'Run LLM intent classification on a query and return the QueryPlan.',
    fields: [
      { name: 'query', label: 'Query', type: 'text', default: 'A hacker discovers reality is fake and someone says there is no spoon' },
    ],
    buildUrl: () => `${API_BASE}/classify`,
    buildBody: (v) => ({ query: v.query }),
  },
  {
    method: 'POST',
    path: '/query',
    description: 'Classify intent + search in one call. Returns ranked movie results.',
    fields: [
      { name: 'query', label: 'Query', type: 'text', default: 'car chase with loud music' },
      {
        name: 'mode', label: 'Search mode', type: 'select', default: 'hybrid',
        options: ['hybrid', 'faiss', 'bm25'],
      },
      { name: 'top_k', label: 'Top K', type: 'number', default: '5' },
    ],
    buildUrl: () => `${API_BASE}/query`,
    buildBody: (v) => ({
      query: v.query,
      mode: v.mode,
      top_k: Math.max(1, Math.min(20, parseInt(v.top_k) || 5)),
    }),
  },
  {
    method: 'POST',
    path: '/search',
    description: 'Search with an explicit QueryPlan (skips LLM classification).',
    fields: [
      { name: 'query', label: 'Query', type: 'text', default: 'heist gone wrong' },
      {
        name: 'intent', label: 'Intent', type: 'select', default: 'full',
        options: ['full', 'scene', 'dialogue', 'character', 'plot'],
      },
      {
        name: 'mode', label: 'Search mode', type: 'select', default: 'hybrid',
        options: ['hybrid', 'faiss', 'bm25'],
      },
      { name: 'top_k', label: 'Top K', type: 'number', default: '5' },
    ],
    buildUrl: () => `${API_BASE}/search`,
    buildBody: (v) => ({
      query: v.query,
      plan: {
        query_type: 'manual',
        intents: [v.intent],
        filters: { genre: null, year_min: null, year_max: null },
        exclude_titles: [],
        reference_title: null,
        rewrite: null,
        sub_queries: null,
      },
      mode: v.mode,
      top_k: Math.max(1, Math.min(20, parseInt(v.top_k) || 5)),
    }),
  },
  {
    method: 'POST',
    path: '/reload',
    description: 'Reload registry from disk after re-indexing — call this once the index pipeline step finishes.',
    buildUrl: () => `${API_BASE}/reload`,
    buildBody: () => ({}),
  },
  {
    method: 'POST',
    path: '/pipeline/{step}',
    description: 'Trigger a pipeline step in the background. Output is appended to pipeline.log.',
    fields: [
      {
        name: 'step', label: 'Step', type: 'select', default: 'sync',
        options: ['sync', 'fetch', 'convert', 'parse', 'reconcile', 'index'],
      },
      { name: 'dry_run',     label: 'Dry run',     type: 'checkbox', default: 'false' },
      { name: 'incremental', label: 'Incremental', type: 'checkbox', default: 'true' },
    ],
    buildUrl: (v) => `${API_BASE}/pipeline/${v.step}`,
    buildBody: (v) => ({ dry_run: v.dry_run === 'true', incremental: v.incremental === 'true' }),
  },
]

// ── EndpointCard component ────────────────────────────────────────────────────

function EndpointCard({ ep }: { ep: EndpointDef }) {
  const initVals = Object.fromEntries(
    (ep.fields ?? []).map((f) => [f.name, f.default])
  )
  const [vals, setVals] = useState<Record<string, string>>(initVals)
  const [response, setResponse] = useState<string | null>(null)
  const [status, setStatus] = useState<number | null>(null)
  const [loading, setLoading] = useState(false)

  async function send() {
    setLoading(true)
    setResponse(null)
    setStatus(null)
    try {
      const url = ep.buildUrl(vals)
      const opts: RequestInit = { method: ep.method }
      if (ep.buildBody) {
        opts.headers = { 'Content-Type': 'application/json' }
        opts.body = JSON.stringify(ep.buildBody(vals))
      }
      const res = await fetch(url, opts)
      setStatus(res.status)
      const json = await res.json()
      setResponse(JSON.stringify(json, null, 2))
    } catch (err) {
      setStatus(0)
      setResponse(String(err))
    } finally {
      setLoading(false)
    }
  }

  const isOk = status !== null && status >= 200 && status < 300

  return (
    <div className="card">
      <div className="card-header">
        <span className={`badge badge-${ep.method.toLowerCase()}`}>{ep.method}</span>
        <code className="path">{ep.path}</code>
      </div>
      <p className="description">{ep.description}</p>

      {ep.fields && ep.fields.length > 0 && (
        <div className="fields">
          {ep.fields.map((f) => {
            const cls = `field${f.type === 'checkbox' ? ' inline' : f.type === 'text' ? ' wide' : ''}`
            return (
              <label key={f.name} className={cls}>
                <span className="field-label">{f.label}</span>
                {f.type === 'select' ? (
                  <select
                    value={vals[f.name]}
                    onChange={(e) => setVals({ ...vals, [f.name]: e.target.value })}
                  >
                    {f.options!.map((o) => (
                      <option key={o} value={o}>{o}</option>
                    ))}
                  </select>
                ) : f.type === 'checkbox' ? (
                  <input
                    type="checkbox"
                    checked={vals[f.name] === 'true'}
                    onChange={(e) =>
                      setVals({ ...vals, [f.name]: e.target.checked ? 'true' : 'false' })
                    }
                  />
                ) : (
                  <input
                    type={f.type}
                    value={vals[f.name]}
                    onChange={(e) => setVals({ ...vals, [f.name]: e.target.value })}
                    onKeyDown={(e) => e.key === 'Enter' && send()}
                  />
                )}
              </label>
            )
          })}
        </div>
      )}

      <button className="send-btn" onClick={send} disabled={loading}>
        {loading ? 'Loading…' : 'Send'}
      </button>

      {response !== null && (
        <div className="response">
          <div className={`status-line ${isOk ? 'ok' : 'err'}`}>
            {status === 0 ? 'Network error' : `HTTP ${status}`}
          </div>
          <pre className="response-body">{response}</pre>
        </div>
      )}
    </div>
  )
}

// ── App ───────────────────────────────────────────────────────────────────────

export default function App() {
  return (
    <div className="app">
      <header className="app-header">
        <h1>Screenplay Retrieval API</h1>
        <p className="subtitle">
          Interactive explorer — <code>{API_BASE}</code>
        </p>
      </header>
      <main className="endpoint-list">
        {ENDPOINTS.map((ep) => (
          <EndpointCard key={`${ep.method}-${ep.path}`} ep={ep} />
        ))}
      </main>
    </div>
  )
}
