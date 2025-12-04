import type { FC } from 'react'

interface HomeProps {
  documentedClassCount: number
  hasReport: boolean
  onNavigate: (page: 'home' | 'upload' | 'report') => void
}

const Home: FC<HomeProps> = ({ documentedClassCount, hasReport, onNavigate }) => {
  return (
    <>
      <section className="panel hero-panel">
        <div className="hero-grid">
          <div>
            <p className="eyebrow">AI‑powered decision support</p>
            <h2 className="hero-title">Skin disease analysis in seconds</h2>
            <p className="subtitle">
              Upload a skin photo and get instant AI-powered analysis using advanced medical imaging models.
            </p>
            <div className="hero-cta-row">
              <button
                type="button"
                className="primary-btn"
                onClick={() => onNavigate('upload')}
              >
                Start with an image
              </button>
              <button
                type="button"
                className="ghost-btn"
                onClick={() => onNavigate('report')}
                disabled={!hasReport}
              >
                View latest report
              </button>
            </div>
          </div>
          <div className="hero-stat-grid">
            <div className="stat-card">
              <p className="stat-label">Disease categories</p>
              <p className="stat-value">{documentedClassCount}</p>
              <p className="stat-helper">Skin conditions we can identify</p>
            </div>
            <div className="stat-card">
              <p className="stat-label">Dual AI models</p>
              <p className="stat-value">2 Models</p>
              <p className="stat-helper">
                Clinical and dermoscopic analysis
              </p>
            </div>
            <div className="stat-card">
              <p className="stat-label">Purpose</p>
              <p className="stat-value">Research tool</p>
              <p className="stat-helper">
                For educational and research use only
              </p>
            </div>
            <div className="stat-card">
              <p className="stat-label">Privacy</p>
              <p className="stat-value">Secure</p>
              <p className="stat-helper">
                Images processed locally, not stored
              </p>
            </div>
          </div>
        </div>
      </section>

      <section className="panel flow-panel">
        <h2>How it works</h2>
        <div className="flow-steps">
          <article className="flow-step">
            <span className="flow-pill">01</span>
            <h3>Upload image</h3>
            <p>
              Take or upload a clear photo of the skin condition
            </p>
          </article>
          <article className="flow-step">
            <span className="flow-pill">02</span>
            <h3>AI analysis</h3>
            <p>
              Our AI models analyze your image using advanced medical imaging
            </p>
          </article>
          <article className="flow-step">
            <span className="flow-pill">03</span>
            <h3>Get results</h3>
            <p>
              Receive instant analysis with confidence scores and explanations
            </p>
          </article>
        </div>
      </section>
    </>
  )
}

export default Home
