"""
Training script for clinical skin disease classification model (New Dataset 3)
"""
import os
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import logging
from PIL import Image as PILImage
import shutil

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ClinicalSkinModelTrainer:
    """Train CNN model for clinical skin disease classification (New Dataset 3)"""

    # Class names taken from folder names in `dataset/New Dataset 3/train`
    CLASS_NAMES = [
        "Acne and Rosacea Photos",
        "Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions",
        "Atopic Dermatitis Photos",
        "Bullous Disease Photos",
        "Cellulitis Impetigo and other Bacterial Infections",
        "Eczema Photos",
        "Exanthems and Drug Eruptions",
        "Hair Loss Photos Alopecia and other Hair Diseases",
        "Herpes HPV and other STDs Photos",
        "Light Diseases and Disorders of Pigmentation",
        "Lupus and other Connective Tissue diseases",
        "Melanoma Skin Cancer Nevi and Moles",
        "Nail Fungus and other Nail Disease",
        "Poison Ivy Photos and other Contact Dermatitis",
        "Psoriasis pictures Lichen Planus and related diseases",
        "Scabies Lyme Disease and other Infestations and Bites",
        "Seborrheic Keratoses and other Benign Tumors",
        "Systemic Disease",
        "Tinea Ringworm Candidiasis and other Fungal Infections",
        "Urticaria Hives",
        "Vascular Tumors",
        "Vasculitis Photos",
        "Warts Molluscum and other Viral Infections",
    ]

    def __init__(
        self,
        data_dir: str = "./dataset/New Dataset 3/train",
        model_save_path: str = "./models/clinical_skin_model.h5",
        input_size: int = 224,
        batch_size: int = 32,
        epochs: int = 30,
    ):
        self.data_dir = data_dir
        self.model_save_path = model_save_path
        self.input_size = input_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.model = None

        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    def validate_images(self):
        """
        Validate all images in the data directory and remove corrupted files.
        Same logic as the main trainer but adapted for this dataset.
        """
        logger.info("Validating images in clinical dataset...")
        valid_count = 0
        invalid_count = 0
        removed_files = []

        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

        backup_dir = os.path.join(self.data_dir, "_corrupted_images")
        os.makedirs(backup_dir, exist_ok=True)

        for class_name in self.CLASS_NAMES:
            class_dir = os.path.join(self.data_dir, class_name)
            if not os.path.exists(class_dir):
                logger.warning(f"Class directory not found (skipping): {class_dir}")
                continue

            logger.info(f"Validating images in {class_name}...")
            for filename in os.listdir(class_dir):
                filepath = os.path.join(class_dir, filename)

                if os.path.isdir(filepath):
                    continue

                _, ext = os.path.splitext(filename.lower())
                if ext not in image_extensions:
                    continue

                try:
                    with PILImage.open(filepath) as img:
                        img.verify()
                    img = PILImage.open(filepath)
                    if img.mode != "RGB":
                        img = img.convert("RGB")
                    img.load()
                    img.close()
                    valid_count += 1
                except Exception as e:  # noqa: BLE001
                    logger.warning(f"Corrupted image found: {filepath} - {str(e)}")
                    invalid_count += 1
                    backup_filepath = os.path.join(backup_dir, f"{class_name}_{filename}")
                    try:
                        shutil.move(filepath, backup_filepath)
                        removed_files.append(filepath)
                    except Exception as move_error:  # noqa: BLE001
                        logger.error(f"Failed to move corrupted file {filepath}: {move_error}")

        logger.info("=" * 60)
        logger.info("Clinical dataset image validation complete!")
        logger.info("  Valid images: %d", valid_count)
        logger.info("  Invalid/removed images: %d", invalid_count)
        logger.info("  Corrupted files moved to: %s", backup_dir)
        logger.info("=" * 60)
        return valid_count, invalid_count, removed_files

    def build_model(self) -> keras.Model:
        """Build CNN model architecture (same style as main trainer)."""
        model = keras.Sequential(
            [
                layers.RandomFlip("horizontal", input_shape=(self.input_size, self.input_size, 3)),
                layers.RandomRotation(0.1),
                layers.RandomZoom(0.1),
                layers.Conv2D(32, (3, 3), activation="relu"),
                layers.MaxPooling2D(2, 2),
                layers.Conv2D(64, (3, 3), activation="relu"),
                layers.MaxPooling2D(2, 2),
                layers.Conv2D(128, (3, 3), activation="relu"),
                layers.MaxPooling2D(2, 2),
                layers.Conv2D(256, (3, 3), activation="relu"),
                layers.MaxPooling2D(2, 2),
                layers.Flatten(),
                layers.Dropout(0.5),
                layers.Dense(512, activation="relu"),
                layers.Dropout(0.5),
                layers.Dense(len(self.CLASS_NAMES), activation="softmax"),
            ]
        )

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss="categorical_crossentropy",
            metrics=["accuracy", "top_k_categorical_accuracy"],
        )
        return model

    def prepare_data_generators(self):
        """Prepare data generators for training and validation."""
        train_datagen = ImageDataGenerator(
            rescale=1.0 / 255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            zoom_range=0.2,
            validation_split=0.2,
        )

        val_datagen = ImageDataGenerator(rescale=1.0 / 255, validation_split=0.2)

        train_gen = train_datagen.flow_from_directory(
            self.data_dir,
            target_size=(self.input_size, self.input_size),
            batch_size=self.batch_size,
            class_mode="categorical",
            classes=self.CLASS_NAMES,
            subset="training",
            shuffle=True,
        )

        val_gen = val_datagen.flow_from_directory(
            self.data_dir,
            target_size=(self.input_size, self.input_size),
            batch_size=self.batch_size,
            class_mode="categorical",
            classes=self.CLASS_NAMES,
            subset="validation",
            shuffle=False,
        )

        return train_gen, val_gen

    def train(self):
        """Train the clinical model."""
        logger.info("Step 1: Validating clinical dataset images...")
        valid_count, invalid_count, _ = self.validate_images()
        if valid_count == 0:
            raise ValueError("No valid images found in clinical dataset!")
        if invalid_count > 0:
            logger.warning("Removed %d corrupted images from clinical dataset.", invalid_count)

        logger.info("Step 2: Building clinical model...")
        self.model = self.build_model()
        self.model.summary()

        logger.info("Step 3: Preparing clinical data generators...")
        train_gen, val_gen = self.prepare_data_generators()

        if train_gen.samples == 0:
            raise ValueError("No images found for clinical training set!")

        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=8,
                restore_best_weights=True,
            ),
            keras.callbacks.ModelCheckpoint(
                self.model_save_path,
                monitor="val_accuracy",
                save_best_only=True,
                verbose=1,
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=4,
                min_lr=1e-5,
            ),
        ]

        logger.info("Starting clinical model training...")
        history = self.model.fit(
            train_gen,
            epochs=self.epochs,
            validation_data=val_gen,
            callbacks=callbacks,
            verbose=1,
        )

        logger.info("Clinical model training complete. Saved to %s", self.model_save_path)
        self.save_training_history(history)
        return history

    def save_training_history(self, history):
        """Save training curves for the clinical model."""
        try:
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            axes[0].plot(history.history["accuracy"], label="Train Acc")
            axes[0].plot(history.history["val_accuracy"], label="Val Acc")
            axes[0].set_title("Clinical Model Accuracy")
            axes[0].set_xlabel("Epoch")
            axes[0].set_ylabel("Accuracy")
            axes[0].legend()

            axes[1].plot(history.history["loss"], label="Train Loss")
            axes[1].plot(history.history["val_loss"], label="Val Loss")
            axes[1].set_title("Clinical Model Loss")
            axes[1].set_xlabel("Epoch")
            axes[1].set_ylabel("Loss")
            axes[1].legend()

            plt.tight_layout()
            plt.savefig("./models/clinical_training_history.png")
            logger.info("Clinical training history saved to ./models/clinical_training_history.png")
        except Exception as e:  # noqa: BLE001
            logger.warning("Could not save clinical training history: %s", str(e))


if __name__ == "__main__":
    trainer = ClinicalSkinModelTrainer()
    trainer.train()


