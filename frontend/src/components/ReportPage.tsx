import { useState } from 'react'
import type { FC } from 'react'
import type { PredictionResponse } from '../services/api'

interface ReportPageProps {
  result: PredictionResponse | null
  probabilities: [string, number][]
  classDescriptions: Record<string, string>
  formatLabel: (label: string) => string
  formatPercentage: (value: number) => string
  onNavigateUpload: () => void
  onNavigateClasses: () => void
}

const ReportPage: FC<ReportPageProps> = ({
  result,
  probabilities,
  classDescriptions,
  formatLabel,
  formatPercentage,
  onNavigateUpload,
  onNavigateClasses,
}) => {
  const [showDisclaimer, setShowDisclaimer] = useState(true)

  return (
    <>
      {showDisclaimer && (
        <div className="full-width-disclaimer">
          <div className="report-card disclaimer">
            <div className="disclaimer-header">
              <h3>Clinical disclaimer</h3>
              <button 
                type="button" 
                className="close-disclaimer-btn" 
                onClick={() => setShowDisclaimer(false)}
                aria-label="Close disclaimer"
              >
                ×
              </button>
            </div>
            <p>
              This is a research tool, not a medical device. Always consult qualified healthcare professionals 
              for diagnosis. The model may make errors, especially with rare conditions or unclear images. 
              Use clinical judgment when interpreting results.
            </p>
          </div>
        </div>
      )}

      <section className="panel report-panel">
        <div className="report-header">
          <div>
            <h2>Prediction report</h2>
            <p className="helper-text">
              A structured summary of the latest model inference for your uploaded image.
            </p>
          </div>
          {result && (
            <span className="status-pill neutral report-pill">
              {result.primary_model_type === 'clinical'
                ? 'Clinical photo model'
                : 'Dermoscopy lesion model'}
            </span>
          )}
        </div>

        {!result && (
          <p className="placeholder">
            No prediction available yet. Upload an image on the{' '}
            <button type="button" className="link-button" onClick={onNavigateUpload}>
              Upload page
            </button>{' '}
            to generate a detailed report.
          </p>
        )}

        {result && (
          <div className="report-grid">
            <div className="report-main">
              <div className="report-card">
                <p className="eyebrow">Primary prediction</p>
                <h3>{formatLabel(result.predicted_class)}</h3>
                <p className="confidence">Model confidence: {formatPercentage(result.confidence)}</p>
                <p className="report-description">
                  {classDescriptions[result.predicted_class] ??
                    'Clinical description for this class is coming soon.'}
                </p>
                {result.warning && <p className="warning">{result.warning}</p>}
              </div>

              {result.secondary_prediction && (
                <div className="report-card secondary">
                  <p className="eyebrow">Secondary model signal</p>
                  <p className="helper-text">
                    The other model ({result.secondary_prediction.model_type}) predicted{' '}
                    <strong>{formatLabel(result.secondary_prediction.predicted_class)}</strong> with{' '}
                    <strong>{formatPercentage(result.secondary_prediction.confidence)}</strong>{' '}
                    confidence.
                  </p>
                </div>
              )}
            </div>

            <aside className="report-aside">
              <div className="report-card">
                <h3>Top differential diagnoses</h3>
                <ul className="probability-list">
                  {probabilities.slice(0, 5).map(([label, score]) => (
                    <li key={label}>
                      <div className="probability-row">
                        <span>{formatLabel(label)}</span>
                        <span>{formatPercentage(score)}</span>
                      </div>
                      <div className="progress">
                        <span style={{ width: `${score * 100}%` }} />
                      </div>
                    </li>
                  ))}
                </ul>
              </div>
            </aside>
          </div>
        )}
      </section>

      <div className="classes-redirect">
        <p className="helper-text">
          Want to learn more about the different skin conditions our AI can identify?
        </p>
        <button 
          type="button" 
          className="primary-btn" 
          onClick={onNavigateClasses}
        >
          View Full Disease Categories
        </button>
      </div>
    </>
  )
}

export default ReportPage

