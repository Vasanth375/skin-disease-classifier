import type { FC } from 'react'

const ClassesPage: FC = () => {
  const dermoscopicClasses = [
    { name: 'actinic_keratosis', description: 'Rough, scaly patches from sun damage' },
    { name: 'basal_cell_carcinoma', description: 'Common skin cancer, grows slowly' },
    { name: 'benign_keratosis', description: 'Harmless skin growths, not cancer' },
    { name: 'dermatofibroma', description: 'Small firm bumps, usually on legs' },
    { name: 'melanoma', description: 'Serious skin cancer, needs quick treatment' },
    { name: 'nevus', description: 'Common moles, usually harmless' },
    { name: 'vascular_lesion', description: 'Red/purple spots from blood vessels' }
  ]

  const clinicalClasses = [
    { name: 'Acne and Rosacea Photos', description: 'Pimples, redness, bumps on face' },
    { name: 'Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions', description: 'Sun damage spots and skin cancers' },
    { name: 'Atopic Dermatitis Photos', description: 'Itchy rash, often in families with allergies' },
    { name: 'Bullous Disease Photos', description: 'Blisters on skin, can be large' },
    { name: 'Cellulitis Impetigo and other Bacterial Infections', description: 'Red, swollen, infected skin' },
    { name: 'Eczema Photos', description: 'Dry, itchy, red skin patches' },
    { name: 'Exanthems and Drug Eruptions', description: 'Widespread rash from drugs or illness' },
    { name: 'Hair Loss Photos Alopecia and other Hair Diseases', description: 'Thinning or missing hair' },
    { name: 'Herpes HPV and other STDs Photos', description: 'Sores or bumps from viruses' },
    { name: 'Light Diseases and Disorders of Pigmentation', description: 'Dark or light spots on skin' },
    { name: 'Lupus and other Connective Tissue diseases', description: 'Immune system attacking skin' },
    { name: 'Melanoma Skin Cancer Nevi and Moles', description: 'Moles and serious skin cancer' },
    { name: 'Nail Fungus and other Nail Disease', description: 'Thick, discolored nails' },
    { name: 'Poison Ivy Photos and other Contact Dermatitis', description: 'Rash from touching plants/chemicals' },
    { name: 'Psoriasis pictures Lichen Planus and related diseases', description: 'Thick, scaly red patches' },
    { name: 'Scabies Lyme Disease and other Infestations and Bites', description: 'Itchy bites or bug infestations' },
    { name: 'Seborrheic Keratoses and other Benign Tumors', description: 'Warty, stuck-on growths' },
    { name: 'Systemic Disease', description: 'Skin signs of body-wide illness' },
    { name: 'Tinea Ringworm Candidiasis and other Fungal Infections', description: 'Ring-shaped rashes, fungal issues' },
    { name: 'Urticaria Hives', description: 'Itchy welts that come and go' },
    { name: 'Vascular Tumors', description: 'Growths of blood vessels' },
    { name: 'Vasculitis Photos', description: 'Inflamed blood vessels in skin' },
    { name: 'Warts Molluscum and other Viral Infections', description: 'Small bumps from viruses' }
  ]

  const formatLabel = (label: string) =>
    label
      .split('_')
      .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
      .join(' ')

  return (
    <>
      <section className="panel classes-panel">
        <h2>Skin Disease Classes</h2>
        <p className="helper-text">
          Our AI models can identify these skin conditions. Always consult a doctor for diagnosis.
        </p>
        
        <div className="classes-container">
          <div className="classes-section">
            <h3>Dermoscopic Classes</h3>
            <p className="section-description">
              Specialized analysis using magnified skin images for detailed examination
            </p>
            <div className="classes-horizontal-grid">
              {dermoscopicClasses.map((item) => (
                <div key={item.name} className="class-card-horizontal">
                  <h4>{formatLabel(item.name)}</h4>
                  <p>{item.description}</p>
                </div>
              ))}
            </div>
          </div>

          <div className="classes-section">
            <h3>Clinical Classes</h3>
            <p className="section-description">
              Regular photographs showing how conditions appear to the naked eye
            </p>
            <div className="classes-horizontal-grid">
              {clinicalClasses.map((item) => (
                <div key={item.name} className="class-card-horizontal">
                  <h4>{item.name}</h4>
                  <p>{item.description}</p>
                </div>
              ))}
            </div>
          </div>
        </div>
      </section>
    </>
  )
}

export default ClassesPage
