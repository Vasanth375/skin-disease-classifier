import { useEffect, useMemo, useState } from 'react'
import type { ChangeEvent, FormEvent } from 'react'
import { API_BASE_URL, checkHealth, fetchClasses, predictDisease } from './services/api'
import type { PredictionResponse } from './services/api'

type HealthStatus = 'checking' | 'healthy' | 'unreachable'

const formatLabel = (label: string) =>
  label
    .split('_')
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ')

const formatPercentage = (value: number) => `${Math.round(value * 1000) / 10}%`

const CLASS_DESCRIPTIONS: Record<string, string> = {
  // HAM10000 dermoscopic lesion classes
  actinic_keratosis:
    'Precancerous scaly patches linked to sun damage; dermatology follow-up recommended.',
  basal_cell_carcinoma:
    'Slow-growing skin cancer that rarely spreads but needs prompt excision.',
  benign_keratosis:
    'Non-cancerous keratin growths (seborrheic keratosis/lichen planus-like keratosis).',
  dermatofibroma: 'Firm, harmless bump often triggered by minor skin trauma or insect bites.',
  melanoma:
    'Aggressive skin cancer originating from melanocytes; urgent specialist review required.',
  nevus: 'Commonly known as a mole; most are benign but monitor for ABCDE changes.',
  vascular_lesion: 'Cluster of abnormal blood vessels such as hemangiomas or angiokeratomas.',

  // Clinical model classes (New Dataset 3)
  'Acne and Rosacea Photos':
    'Inflammatory conditions causing pimples, redness, and bumps on the face and upper body.',
  'Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions':
    'Precancerous and cancerous lesions often related to chronic sun exposure; require prompt evaluation.',
  'Atopic Dermatitis Photos':
    'Chronic itchy eczema often starting in childhood, associated with allergies and dry skin.',
  'Bullous Disease Photos':
    'Conditions that cause fluid-filled blisters on the skin or mucous membranes (e.g., pemphigus).',
  'Cellulitis Impetigo and other Bacterial Infections':
    'Bacterial infections of the skin presenting with redness, warmth, crusting, or pus.',
  'Eczema Photos':
    'Inflammatory, itchy rashes with redness, scaling, and sometimes oozing or crusting.',
  'Exanthems and Drug Eruptions':
    'Widespread rashes often triggered by infections or medications; may need urgent review.',
  'Hair Loss Photos Alopecia and other Hair Diseases':
    'Patchy or diffuse hair loss from various causes including alopecia areata or androgenetic alopecia.',
  'Herpes HPV and other STDs Photos':
    'Viral infections of the skin and mucosa, often sexually transmitted and sometimes recurrent.',
  'Light Diseases and Disorders of Pigmentation':
    'Conditions causing lighter or darker patches due to pigment changes or sun sensitivity.',
  'Lupus and other Connective Tissue diseases':
    'Autoimmune conditions affecting skin and internal organs, often with photosensitive rashes.',
  'Melanoma Skin Cancer Nevi and Moles':
    'Pigmented lesions ranging from benign moles to malignant melanoma; require careful monitoring.',
  'Nail Fungus and other Nail Disease':
    'Thickened, discolored, or deformed nails from fungal infection or other nail disorders.',
  'Poison Ivy Photos and other Contact Dermatitis':
    'Allergic or irritant reactions to substances touching the skin, leading to redness and blisters.',
  'Psoriasis pictures Lichen Planus and related diseases':
    'Chronic inflammatory conditions with scaly plaques or flat-topped violaceous papules.',
  'Scabies Lyme Disease and other Infestations and Bites':
    'Infestations and bites causing intense itching, burrows, or target-like rashes.',
  'Seborrheic Keratoses and other Benign Tumors':
    'Warty, stuck-on growths and other usually harmless skin tumors.',
  'Systemic Disease':
    'Skin signs that may be linked to internal organ disease or systemic illness.',
  'Tinea Ringworm Candidiasis and other Fungal Infections':
    'Fungal infections causing ring-shaped rashes, scaling, or moist red patches.',
  'Urticaria Hives':
    'Transient itchy welts or bumps that appear and fade over hours, often allergic or idiopathic.',
  'Vascular Tumors':
    'Benign or malignant growths of blood vessels presenting as red, purple, or bluish lesions.',
  'Vasculitis Photos':
    'Inflammation of blood vessels causing purpura, ulcers, or nodules; may be systemic.',
  'Warts Molluscum and other Viral Infections':
    'Localized viral growths such as warts or molluscum contagiosum with characteristic bumps.',
}

function App() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)
  const [result, setResult] = useState<PredictionResponse | null>(null)
  const [classes, setClasses] = useState<string[]>([])
  const [healthStatus, setHealthStatus] = useState<HealthStatus>('checking')
  const [error, setError] = useState<string | null>(null)
  const [uploading, setUploading] = useState(false)

  useEffect(() => {
    let isMounted = true

    const bootstrap = async () => {
      try {
        const [classList, health] = await Promise.all([fetchClasses(), checkHealth()])
        if (isMounted) {
          setClasses(classList)
          setHealthStatus(health.status === 'healthy' ? 'healthy' : 'unreachable')
        }
      } catch {
        if (isMounted) {
          setHealthStatus('unreachable')
        }
      }
    }

    bootstrap()

    return () => {
      isMounted = false
    }
  }, [])

  useEffect(() => {
    return () => {
      if (previewUrl) {
        URL.revokeObjectURL(previewUrl)
      }
    }
  }, [previewUrl])

  const probabilities = useMemo(() => {
    if (!result) return []
    return Object.entries(result.probabilities ?? {}).sort((a, b) => b[1] - a[1])
  }, [result])

  const handleFileChange = (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    setSelectedFile(file ?? null)
    setResult(null)
    setError(null)

    if (previewUrl) {
      URL.revokeObjectURL(previewUrl)
    }
    setPreviewUrl(file ? URL.createObjectURL(file) : null)
  }

  const handleSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault()

    if (!selectedFile) {
      setError('Please choose an image before requesting a prediction.')
      return
    }

    setUploading(true)
    setError(null)

    try {
      const response = await predictDisease(selectedFile)
      setResult(response)
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Prediction failed. Please retry.'
      setError(message)
    } finally {
      setUploading(false)
    }
  }

  const displayedClasses = Object.keys(CLASS_DESCRIPTIONS)

  return (
    <div className="app-shell">
      <header className="hero">
        <div>
          <p className="eyebrow">FastAPI + React</p>
          <h1>Skin Disease Classifier</h1>
          <p className="subtitle">
            Upload a skin image (dermoscopy or clinical photo) to run two CNN models in parallel.
            The backend automatically chooses the more confident prediction and shows per-class
            probabilities.
          </p>
        </div>

        <div className="status-row">
          <span className={`status-pill ${healthStatus}`}>
            API {healthStatus === 'healthy' ? 'online' : healthStatus === 'checking' ? 'checking...' : 'offline'}
          </span>
          <span className="status-pill neutral">
            {Object.keys(CLASS_DESCRIPTIONS).length} documented classes
          </span>
          <span className="status-pill neutral">{API_BASE_URL}</span>
        </div>
      </header>

      <main className="grid">
        <section className="panel upload-panel">
          <h2>Upload a skin lesion image</h2>
          <p className="helper-text">
            Supported formats: JPEG, PNG. Clear, well-lit clinical or dermoscopic images yield the
            best results.
          </p>

          <form onSubmit={handleSubmit}>
            <label className="upload-dropzone">
              <input
                type="file"
                accept="image/png,image/jpeg,image/jpg"
                onChange={handleFileChange}
                disabled={uploading}
              />
              {previewUrl ? (
                <img src={previewUrl} alt="Preview" className="preview-image" />
              ) : (
                <p>
                  Drag & drop or <span>browse</span> your image
                </p>
              )}
            </label>

            <button type="submit" disabled={uploading || !selectedFile} className="primary-btn">
              {uploading ? 'Analyzing...' : 'Run prediction'}
            </button>

            {error && <p className="error">{error}</p>}
          </form>
        </section>

        <section className="panel result-panel">
          <h2>Prediction</h2>
          {!result && <p className="placeholder">Upload an image to view the model output.</p>}

          {result && (
            <>
              <div className="prediction-heading">
                <p className="eyebrow">Likely condition</p>
                <h3>{formatLabel(result.predicted_class)}</h3>
                <p className="confidence">Confidence: {formatPercentage(result.confidence)}</p>
                {result.primary_model_type && (
                  <p className="helper-text">
                    Model used:{' '}
                    {result.primary_model_type === 'clinical'
                      ? 'Clinical photo model'
                      : 'Dermoscopy lesion model'}
                  </p>
                )}
                {result.secondary_prediction && (
                  <p className="helper-text">
                    Other model ({result.secondary_prediction.model_type}) predicted{' '}
                    {formatLabel(result.secondary_prediction.predicted_class)} with{' '}
                    {formatPercentage(result.secondary_prediction.confidence)} confidence.
                  </p>
                )}
              </div>

              {result.warning && <p className="warning">{result.warning}</p>}

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
            </>
          )}
        </section>
      </main>

      <section className="panel classes-panel">
        <div>
          <h2>All supported classes</h2>
          <p className="helper-text">
            These are the diagnostic categories exposed by the FastAPI endpoint.
          </p>
          {!classes.length && (
            <p className="helper-text muted">
              API did not return the class list, so the default reference set is shown below.
            </p>
          )}
        </div>
        <div className="class-grid">
          {displayedClasses.map((name) => (
            <article key={name} className="class-card">
              <h3>{formatLabel(name)}</h3>
              <p>{CLASS_DESCRIPTIONS[name] ?? 'Diagnostic description coming soon.'}</p>
            </article>
          ))}
        </div>
      </section>
    </div>
  )
}

export default App

