import type { FC, ChangeEvent, FormEvent } from 'react'

interface UploadPageProps {
  previewUrl: string | null
  selectedFile: File | null
  uploading: boolean
  error: string | null
  probabilities: [string, number][]
  onFileChange: (event: ChangeEvent<HTMLInputElement>) => void
  onSubmit: (event: FormEvent<HTMLFormElement>) => void
  formatLabel: (label: string) => string
  formatPercentage: (value: number) => string
}

const UploadPage: FC<UploadPageProps> = ({
  previewUrl,
  selectedFile,
  uploading,
  error,
  probabilities,
  onFileChange,
  onSubmit,
  formatLabel,
  formatPercentage,
}) => {
  return (
    <main className="grid">
      <section className="panel upload-panel">
        <h2>Upload a skin lesion image</h2>
        <p className="helper-text">
          Supported formats: JPEG, PNG. Clear, well-lit clinical or dermoscopic images yield the
          best results.
        </p>

        <form onSubmit={onSubmit}>
          <label className="upload-dropzone">
            <input
              type="file"
              accept="image/png,image/jpeg,image/jpg"
              onChange={onFileChange}
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
        <h2>Quick prediction glance</h2>
        {!probabilities.length && <p className="placeholder">Run a prediction to see the model output.</p>}

        {!!probabilities.length && (
          <ul className="probability-list compact">
            {probabilities.slice(0, 3).map(([label, score]) => (
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
        )}
      </section>
    </main>
  )
}

export default UploadPage


