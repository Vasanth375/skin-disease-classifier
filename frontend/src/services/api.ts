export type PredictionResponse = {
  predicted_class: string
  confidence: number
  probabilities: Record<string, number>
  all_classes: string[]
  warning?: string | null
}

export type HealthResponse = {
  status: string
}

const DEFAULT_API_BASE_URL = 'http://localhost:8000'

export const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL ?? DEFAULT_API_BASE_URL).replace(/\/$/, '')

const ROUTES = {
  predict: `${API_BASE_URL}/api/v1/predict`,
  classes: `${API_BASE_URL}/api/v1/classes`,
  health: `${API_BASE_URL}/health`,
}

const buildError = async (response: Response) => {
  try {
    const body = await response.json()
    const detail = body?.detail ?? body?.message
    return new Error(detail ?? `Request failed with status ${response.status}`)
  } catch {
    return new Error(`Request failed with status ${response.status}`)
  }
}

export const predictDisease = async (file: File): Promise<PredictionResponse> => {
  const formData = new FormData()
  formData.append('file', file)

  const response = await fetch(ROUTES.predict, {
    method: 'POST',
    body: formData,
  })

  if (!response.ok) {
    throw await buildError(response)
  }

  return (await response.json()) as PredictionResponse
}

export const fetchClasses = async (): Promise<string[]> => {
  const response = await fetch(ROUTES.classes)
  if (!response.ok) {
    throw await buildError(response)
  }

  const payload = (await response.json()) as { classes?: string[] }
  return payload.classes ?? []
}

export const checkHealth = async (): Promise<HealthResponse> => {
  const response = await fetch(ROUTES.health)
  if (!response.ok) {
    throw await buildError(response)
  }

  return (await response.json()) as HealthResponse
}

