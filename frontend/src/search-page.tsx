import { useState } from 'react'
import type { FormEvent } from 'react'
import { createRoot } from 'react-dom/client'
import './search-page.css'

const API_BASE = `http://${window.location.hostname || 'localhost'}:9000`

type SearchMode = 'video' | 'screenplay'

type QueryResponse = {
  query: string
  plan: Record<string, unknown>
  entity_names: string[]
  results: QueryResult[]
  total: number
}

type QueryResult = {
  rank: number
  movie_id: string
  title: string
  score: number
  scene_ids: string[]
}

type MovieDetail = {
  movie_id: string
  title: string
  year: string
  genres: string[]
  directors: string[]
  overview: string
  poster_url: string | null
  tmdb_url: string | null
  slug: string
  scene_count: number | null
}

type ResultCardData = QueryResult & Partial<MovieDetail>

type VideoResult = {
  doc_id: string
  clip_id: string
  movie: string
  field_type: string
  score: number
  content: string
  youtube_link: string
  timestamp: string
  description: string
  rrf_score: number
  matched_fields: string[]
  retrieval_rank: number
  llm_rerank_rank: number
}

type NormalizedVideoResult = Omit<VideoResult, 'rrf_score' | 'retrieval_rank' | 'llm_rerank_rank'> & {
  rrf_score: number | null
  retrieval_rank: number | null
  llm_rerank_rank: number | null
}

function normalizeVideoResult(raw: Partial<VideoResult> & Record<string, unknown>): NormalizedVideoResult {
  return {
    doc_id: String(raw.doc_id ?? ''),
    clip_id: String(raw.clip_id ?? ''),
    movie: String(raw.movie ?? 'Unknown title'),
    field_type: String(raw.field_type ?? 'unknown'),
    score: typeof raw.score === 'number' ? raw.score : 0,
    content: String(raw.content ?? ''),
    youtube_link: String(raw.youtube_link ?? ''),
    timestamp: String(raw.timestamp ?? 'Timestamp unavailable'),
    description: String(raw.description ?? ''),
    rrf_score: typeof raw.rrf_score === 'number' ? raw.rrf_score : null,
    matched_fields: Array.isArray(raw.matched_fields)
      ? raw.matched_fields.map((field) => String(field))
      : [],
    retrieval_rank: typeof raw.retrieval_rank === 'number' ? raw.retrieval_rank : null,
    llm_rerank_rank: typeof raw.llm_rerank_rank === 'number' ? raw.llm_rerank_rank : null,
  }
}

function getYouTubeEmbedUrl(url: string) {
  try {
    const parsed = new URL(url)
    const videoId = parsed.searchParams.get('v')

    if (videoId) {
      const start = parsed.searchParams.get('t')
      return `https://www.youtube.com/embed/${videoId}${start ? `?start=${parseYouTubeTime(start)}` : ''}`
    }

    if (parsed.hostname.includes('youtu.be')) {
      const videoIdFromPath = parsed.pathname.replace('/', '')
      return videoIdFromPath ? `https://www.youtube.com/embed/${videoIdFromPath}` : null
    }

    return null
  } catch {
    return null
  }
}

function parseYouTubeTime(value: string) {
  if (/^\d+$/.test(value)) {
    return value
  }

  const parts = value.match(/(?:(\d+)h)?(?:(\d+)m)?(?:(\d+)s)?/)
  if (!parts) {
    return '0'
  }

  const hours = Number(parts[1] ?? 0)
  const minutes = Number(parts[2] ?? 0)
  const seconds = Number(parts[3] ?? 0)
  return String((hours * 60 * 60) + (minutes * 60) + seconds)
}

function SearchPage() {
  const [query, setQuery] = useState('')
  const [selectedMode, setSelectedMode] = useState<SearchMode>('screenplay')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [status, setStatus] = useState<string>('')
  const [response, setResponse] = useState<QueryResponse | null>(null)
  const [results, setResults] = useState<ResultCardData[]>([])
  const [videoResults, setVideoResults] = useState<NormalizedVideoResult[]>([])
  const [showAlternatives, setShowAlternatives] = useState(false)
  const [acceptedFirstResult, setAcceptedFirstResult] = useState(false)

  const primaryResult = results[0] ?? null
  const alternativeResults = showAlternatives ? results.slice(1) : []
  const primaryVideoResult = videoResults[0] ?? null
  const alternativeVideoResults = videoResults.slice(1)

  async function handleSearch(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()

    const trimmedQuery = query.trim()
    if (!trimmedQuery) {
      setError('Enter a search query first.')
      setStatus('Waiting for a valid query.')
      setResponse(null)
      setResults([])
      setVideoResults([])
      setShowAlternatives(false)
      setAcceptedFirstResult(false)
      return
    }

    setLoading(true)
    setError(null)
    setStatus('')
    setResponse(null)
    setResults([])
    setVideoResults([])
    setShowAlternatives(false)
    setAcceptedFirstResult(false)

    try {
      if (selectedMode === 'video') {
        const nextVideoResults = await fetchVideo(trimmedQuery)
        setVideoResults(nextVideoResults)
        setStatus(
          nextVideoResults.length === 0
            ? 'No video matches found.'
            : `Showing ${nextVideoResults.length} ranked video match${nextVideoResults.length === 1 ? '' : 'es'}.`,
        )
        return
      }

      const queryResponse = await fetchQuery(trimmedQuery)
      const enrichedResults = await enrichResults(queryResponse.results)

      setResponse(queryResponse)
      setResults(enrichedResults)

      if (enrichedResults.length === 0) {
        setStatus('No matches returned from the backend.')
      } else {
        setStatus(`Showing the top result first. ${enrichedResults.length - 1} more are ready if needed.`)
      }
    } catch (caughtError) {
      const message = caughtError instanceof Error ? caughtError.message : 'Unknown error'
      setError(message)
      setStatus('Search failed.')
      setResponse(null)
      setResults([])
      setVideoResults([])
    } finally {
      setLoading(false)
    }
  }

  async function fetchQuery(searchText: string): Promise<QueryResponse> {
    const requestBody = {
      query: searchText,
      mode: 'hybrid',
      top_k: 5,
    }

    const res = await fetch(`${API_BASE}/query`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(requestBody),
    })

    if (!res.ok) {
      throw new Error(`Backend returned HTTP ${res.status}`)
    }

    return res.json() as Promise<QueryResponse>
  }

  async function fetchVideo(searchText: string): Promise<NormalizedVideoResult[]> {
    const res = await fetch(`${API_BASE}/video_query`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        query: searchText,
        top_k: 3,
      }),
    })

    if (!res.ok) {
      throw new Error(`Video backend returned HTTP ${res.status}`)
    }

    const data = await res.json() as unknown
    if (!Array.isArray(data)) {
      throw new Error('Video backend returned an unexpected response shape')
    }

    return data.map((item) => normalizeVideoResult((item ?? {}) as Record<string, unknown>))
  }

  async function enrichResults(items: QueryResult[]): Promise<ResultCardData[]> {
    const details = await Promise.all(
      items.map(async (item) => {
        try {
          const res = await fetch(`${API_BASE}/movies/${item.movie_id}`)
          if (!res.ok) {
            return item
          }

          const detail = await res.json() as MovieDetail
          return { ...item, ...detail }
        } catch {
          return item
        }
      }),
    )

    return details
  }

  return (
    <main className="search-page">
      <div className="shell">
        <section className="hero">
          <span className="eyebrow">CineSearch</span>
          <h1>
            Describe anything you remember —<br />
            a scene, a quote, a plot, a character, a feeling.<br />
            We'll find the movie.
          </h1>
        </section>

        <section className="search-panel">
          <form className="search-form" onSubmit={handleSearch}>
            <div className="search-row">
              <input
                className="search-input"
                type="text"
                placeholder="Search for a scene, plot beat, line, or character moment..."
                value={query}
                onChange={(event) => setQuery(event.target.value)}
              />
              <button className="primary-btn" type="submit" disabled={loading}>
                {loading ? 'Searching...' : 'Search'}
              </button>
            </div>

            <div className="mode-row">
              <span className="mode-label">Mode</span>
              <div className="mode-chip-group" role="tablist" aria-label="Search mode">
                {(['video', 'screenplay'] as const).map((mode) => (
                  <button
                    key={mode}
                    className={`mode-chip ${selectedMode === mode ? 'active' : ''}`}
                    type="button"
                    onClick={() => setSelectedMode(mode)}
                    aria-pressed={selectedMode === mode}
                  >
                    {mode === 'video' ? 'Video' : 'Screenplay'}
                  </button>
                ))}
              </div>
              <span className="mode-copy">
                {selectedMode === 'video'
                  ? 'Queries the video retrieval backend and returns ranked YouTube clips with timestamps and matched fields.'
                  : 'Uses the screenplay backend and returns the best matching scenes.'}
              </span>
            </div>
          </form>

          <div className={`status-line ${error ? 'error' : response || primaryVideoResult ? 'success' : ''}`}>
            {error ?? status}
          </div>
        </section>

        {primaryVideoResult && (
          <section className="results-panel">
            <div className="results-head">
              <div>
                <h2>Video matches</h2>
                <p className="results-subtitle">
                  Query: <strong>{query.trim()}</strong>
                </p>
              </div>
              <span className="result-count">{videoResults.length} video results returned</span>
            </div>

            <article className="featured-card video-card">
              <VideoPreview result={primaryVideoResult} />
              <div className="result-body">
                <div className="result-header">
                  <div>
                    <h3 className="result-title">{primaryVideoResult.movie}</h3>
                    <p className="clip-heading">{primaryVideoResult.description}</p>
                    <div className="result-meta">
                      <span>Timestamp {primaryVideoResult.timestamp}</span>
                    </div>
                  </div>
                </div>

                <div className="result-meta">
                  <span className="pill">
                    Rank #{primaryVideoResult.retrieval_rank ?? 1}
                  </span>
                </div>

                <div className="detail-block">
                  <span className="detail-label">Matched clip description</span>
                  <p>{primaryVideoResult.content}</p>
                </div>
                <div className="feedback-bar">
                  <span className="feedback-copy">
                    {acceptedFirstResult ? 'Keeping this result selected.' : 'Is this the right result?'}
                  </span>
                  <div className="feedback-actions">
                    <button
                      className="ghost-btn yes"
                      type="button"
                      onClick={() => {
                        setAcceptedFirstResult(true)
                        setShowAlternatives(false)
                      }}
                    >
                      Yes
                    </button>
                    <button
                      className="ghost-btn no"
                      type="button"
                      onClick={() => {
                        setAcceptedFirstResult(false)
                        setShowAlternatives(true)
                      }}
                    >
                      No
                    </button>
                    <a
                      className="primary-link"
                      href={primaryVideoResult.youtube_link}
                      target="_blank"
                      rel="noreferrer"
                    >
                      Watch on YouTube
                    </a>
                  </div>
                </div>
              </div>
            </article>

            {showAlternatives && alternativeVideoResults.length > 0 && (
              <div className="thumbnail-grid">
                {alternativeVideoResults.map((result, index) => (
                  <article key={result.doc_id} className="result-card">
                    <VideoPreview result={result} compact />
                    <div className="result-body">
                      <div className="result-header">
                        <div>
                          <h3 className="result-title">{result.movie}</h3>
                          <p className="clip-heading">{result.description}</p>
                        </div>
                      </div>
                      <div className="result-meta">
                        <span className="pill">Rank #{result.retrieval_rank ?? index + 2}</span>
                        <span>{result.timestamp}</span>
                      </div>
                      <div className="feedback-bar">
                        <span className="feedback-copy">Open this alternate clip.</span>
                        <div className="feedback-actions">
                          <a
                            className="primary-link"
                            href={result.youtube_link}
                            target="_blank"
                            rel="noreferrer"
                          >
                            Watch
                          </a>
                        </div>
                      </div>
                    </div>
                  </article>
                ))}
              </div>
            )}
          </section>
        )}

        {primaryResult && (
          <section className="results-panel">
            <div className="results-head">
              <div>
                <h2>Top match</h2>
                <p className="results-subtitle">
                  Query: <strong>{response?.query}</strong>
                </p>
              </div>
              <span className="result-count">{response?.total ?? results.length} results returned</span>
            </div>

            <article className="featured-card">
              <PosterCard result={primaryResult} />
              <div className="result-body">
                <div className="result-header">
                  <div>
                    <h3 className="result-title">{primaryResult.title}</h3>
                    <div className="result-meta">
                      <span>{primaryResult.year || 'Year unavailable'}</span>
                      <span>{selectedMode === 'video' ? 'Video match' : 'Screenplay match'}</span>
                      {primaryResult.directors?.[0] && <span>Dir. {primaryResult.directors[0]}</span>}
                    </div>
                  </div>
                </div>

                <div className="result-meta">
                  <span className="pill">Rank #{primaryResult.rank}</span>
                  {primaryResult.genres?.[0] ? <span className="pill">{primaryResult.genres[0]}</span> : null}
                </div>

                <p className="overview">
                  {primaryResult.overview || 'No overview was returned for this title.'}
                </p>

                <div className="feedback-bar">
                  <span className="feedback-copy">
                    {acceptedFirstResult ? 'Keeping this result selected.' : 'Is this the right result?'}
                  </span>
                  <div className="feedback-actions">
                    <button
                      className="ghost-btn yes"
                      type="button"
                      onClick={() => {
                        setAcceptedFirstResult(true)
                        setShowAlternatives(false)
                      }}
                    >
                      Yes
                    </button>
                    <button
                      className="ghost-btn no"
                      type="button"
                      onClick={() => {
                        setAcceptedFirstResult(false)
                        setShowAlternatives(true)
                      }}
                    >
                      No
                    </button>
                  </div>
                </div>
              </div>
            </article>

            {alternativeResults.length > 0 && (
              <div className="thumbnail-grid">
                {alternativeResults.map((result) => (
                  <article key={result.movie_id} className="result-card">
                    <PosterCard result={result} />
                    <div className="result-body">
                      <div className="result-header">
                        <h3 className="result-title">{result.title}</h3>
                      </div>
                      <div className="result-meta">
                        <span className="pill">Rank #{result.rank}</span>
                        <span>{result.year || 'Year unavailable'}</span>
                      </div>
                      <p className="overview">
                        {result.overview || 'No overview available for this result.'}
                      </p>
                    </div>
                  </article>
                ))}
              </div>
            )}
          </section>
        )}

        {loading && !error && (
          <div className="loading-dock" aria-live="polite" aria-label="Searching">
            <span className="loading-dot" />
            <span className="loading-dot" />
            <span className="loading-dot" />
          </div>
        )}
      </div>
    </main>
  )
}

function PosterCard({ result }: { result: ResultCardData }) {
  if (result.poster_url) {
    return <img className="poster" src={result.poster_url} alt={`${result.title} poster`} />
  }

  return <div className="poster-placeholder">No poster</div>
}

function VideoPreview({ result, compact = false }: { result: NormalizedVideoResult; compact?: boolean }) {
  const embedUrl = getYouTubeEmbedUrl(result.youtube_link)

  if (embedUrl) {
    return (
      <div className={`video-frame-shell ${compact ? 'compact' : ''}`}>
        <iframe
          className="video-frame"
          src={embedUrl}
          title={`${result.movie} - ${result.description}`}
          allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
          allowFullScreen
          loading="lazy"
          referrerPolicy="strict-origin-when-cross-origin"
        />
        <span className="video-badge">{compact ? 'Alt Match' : 'YouTube Result'}</span>
      </div>
    )
  }

  return (
    <div className={`video-placeholder ${compact ? 'compact' : ''}`}>
      <span className="video-badge">{compact ? 'Alt Match' : 'YouTube Result'}</span>
    </div>
  )
}

createRoot(document.getElementById('root')!).render(<SearchPage />)
