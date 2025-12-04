import { useEffect, useMemo, useState } from 'react'
import type { ChangeEvent, FormEvent } from 'react'
import { checkHealth, predictDisease } from './services/api'
import type { PredictionResponse } from './services/api'
import Home from './components/Home'
import UploadPage from './components/UploadPage'
import ReportPage from './components/ReportPage'
import ClassesPage from './components/ClassesPage'

type HealthStatus = 'checking' | 'healthy' | 'unreachable'
type Page = 'home' | 'upload' | 'report' | 'classes'

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
  const [, setHealthStatus] = useState<HealthStatus>('checking')
  const [error, setError] = useState<string | null>(null)
  const [uploading, setUploading] = useState(false)
  const [activePage, setActivePage] = useState<Page>('home')

  useEffect(() => {
    let isMounted = true

    const bootstrap = async () => {
      try {
        const health = await checkHealth()
        if (isMounted) {
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

    // As soon as the user selects an image, guide them to the upload page
    setActivePage('upload')
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
      setActivePage('report')
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Prediction failed. Please retry.'
      setError(message)
    } finally {
      setUploading(false)
    }
  }


  return (
    <div className="app-shell">
      <header className="app-header">
        <div className="header-content">
          <h1>Skin Disease Classifier</h1>
          <nav className="nav-tabs">
            <button
              type="button"
              className={`nav-tab ${activePage === 'home' ? 'active' : ''}`}
              onClick={() => setActivePage('home')}
            >
              Home overview
            </button>
            <button
              type="button"
              className={`nav-tab ${activePage === 'upload' ? 'active' : ''}`}
              onClick={() => setActivePage('upload')}
            >
              Upload image
            </button>
            <button
              type="button"
              className={`nav-tab ${activePage === 'report' ? 'active' : ''}`}
              onClick={() => setActivePage('report')}
            >
              Prediction report
            </button>
            <button
              type="button"
              className={`nav-tab ${activePage === 'classes' ? 'active' : ''}`}
              onClick={() => setActivePage('classes')}
            >
              Disease Categories
            </button>
          </nav>
        </div>
      </header>

      {activePage === 'home' && (
        <Home
          documentedClassCount={Object.keys(CLASS_DESCRIPTIONS).length}
          hasReport={!!result}
          onNavigate={setActivePage}
        />
      )}
      {activePage === 'upload' && (
        <UploadPage
          previewUrl={previewUrl}
          selectedFile={selectedFile}
          uploading={uploading}
          error={error}
          probabilities={probabilities}
          onFileChange={handleFileChange}
          onSubmit={handleSubmit}
          formatLabel={formatLabel}
          formatPercentage={formatPercentage}
        />
      )}
      {activePage === 'report' && (
        <ReportPage
          result={result}
          probabilities={probabilities}
          classDescriptions={CLASS_DESCRIPTIONS}
          formatLabel={formatLabel}
          formatPercentage={formatPercentage}
          onNavigateUpload={() => setActivePage('upload')}
          onNavigateClasses={() => setActivePage('classes')}
        />
      )}
      {activePage === 'classes' && (
        <ClassesPage />
      )}
    </div>
  )
}

export default App

