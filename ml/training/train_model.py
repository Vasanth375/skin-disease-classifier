"""
Training script for skin disease classification model
"""
import os
import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from pathlib import Path
import logging
from PIL import Image as PILImage
import shutil

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SkinDiseaseModelTrainer:
    """Train CNN model for skin disease classification"""
    
    CLASS_NAMES = [
        "actinic_keratosis",
        "basal_cell_carcinoma",
        "benign_keratosis",
        "dermatofibroma",
        "melanoma",
        "nevus",
        "vascular_lesion"
    ]
    
    def __init__(
        self,
        data_dir: str = "./data/processed",
        model_save_path: str = "./models/skin_disease_model.h5",
        input_size: int = 224,
        batch_size: int = 32,
        epochs: int = 50,
        checkpoint_dir: str = "./models/checkpoints",
        resume_from_checkpoint: str = None
    ):
        """
        Initialize trainer
        
        Args:
            data_dir: Directory containing processed images
            model_save_path: Path to save trained model
            input_size: Input image size
            batch_size: Training batch size
            epochs: Number of training epochs
        """
        self.data_dir = data_dir
        self.model_save_path = model_save_path
        self.input_size = input_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.checkpoint_dir = checkpoint_dir
        self.resume_from_checkpoint = resume_from_checkpoint
        self.model = None
        self.initial_epoch = 0
        
        # Create models directory if it doesn't exist
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        # Create checkpoints directory if it doesn't exist
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    def validate_images(self):
        """
        Validate all images in the data directory and remove corrupted files
        
        Returns:
            Tuple of (valid_count, invalid_count, removed_files)
        """
        logger.info("Validating images in dataset...")
        valid_count = 0
        invalid_count = 0
        removed_files = []
        
        # Image extensions to check
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        
        # Create backup directory for corrupted files
        backup_dir = os.path.join(self.data_dir, '_corrupted_images')
        os.makedirs(backup_dir, exist_ok=True)
        
        # Walk through all class directories
        for class_name in self.CLASS_NAMES:
            class_dir = os.path.join(self.data_dir, class_name)
            if not os.path.exists(class_dir):
                continue
            
            logger.info(f"Validating images in {class_name}...")
            class_valid = 0
            class_invalid = 0
            
            # Check all files in the class directory
            for filename in os.listdir(class_dir):
                filepath = os.path.join(class_dir, filename)
                
                # Skip directories
                if os.path.isdir(filepath):
                    continue
                
                # Check if it's an image file
                _, ext = os.path.splitext(filename.lower())
                if ext not in image_extensions:
                    # Skip non-image files
                    continue
                
                # Try to open and verify the image
                try:
                    # Open and verify the image
                    with PILImage.open(filepath) as img:
                        # Verify the image by attempting to load it
                        img.verify()
                    
                    # Re-open because verify() closes the file
                    img = PILImage.open(filepath)
                    # Try to convert to RGB to ensure it's valid
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    
                    # Try to get basic properties
                    img.load()
                    img.close()  # Explicitly close the image
                    
                    class_valid += 1
                    valid_count += 1
                        
                except Exception as e:
                    # Image is corrupted or invalid
                    logger.warning(f"Corrupted image found: {filepath} - {str(e)}")
                    invalid_count += 1
                    class_invalid += 1
                    
                    # Move corrupted file to backup directory
                    backup_filepath = os.path.join(backup_dir, f"{class_name}_{filename}")
                    try:
                        shutil.move(filepath, backup_filepath)
                        removed_files.append(filepath)
                        logger.info(f"Moved corrupted image to: {backup_filepath}")
                    except Exception as move_error:
                        logger.error(f"Failed to move corrupted file {filepath}: {move_error}")
            
            if class_invalid > 0:
                logger.info(f"{class_name}: {class_valid} valid, {class_invalid} invalid (removed)")
        
        logger.info("="*60)
        logger.info(f"Image validation complete!")
        logger.info(f"  Valid images: {valid_count}")
        logger.info(f"  Invalid/removed images: {invalid_count}")
        logger.info(f"  Corrupted files moved to: {backup_dir}")
        logger.info("="*60)
        
        if invalid_count > 0:
            logger.warning(f"Removed {invalid_count} corrupted images. They are backed up in {backup_dir}")
        
        return valid_count, invalid_count, removed_files
    
    def build_model(self) -> keras.Model:
        """
        Build CNN model architecture
        
        Returns:
            Compiled Keras model
        """
        model = keras.Sequential([
            # Data augmentation
            layers.RandomFlip("horizontal", input_shape=(self.input_size, self.input_size, 3)),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
            
            # First Convolutional Block
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.MaxPooling2D(2, 2),
            
            # Second Convolutional Block
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D(2, 2),
            
            # Third Convolutional Block
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D(2, 2),
            
            # Fourth Convolutional Block
            layers.Conv2D(256, (3, 3), activation='relu'),
            layers.MaxPooling2D(2, 2),
            
            # Flatten and Dense layers
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(len(self.CLASS_NAMES), activation='softmax')
        ])
        
        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'top_k_categorical_accuracy']
        )
        
        return model
    
    def prepare_data_generators(self):
        """
        Prepare data generators for training and validation
        
        Returns:
            Tuple of (train_generator, validation_generator)
        """
        # Data augmentation for training
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            zoom_range=0.2,
            validation_split=0.2
        )
        
        # Only rescaling for validation
        validation_datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=0.2
        )
        
        # Training generator - explicitly specify classes to avoid _corrupted_images folder
        train_generator = train_datagen.flow_from_directory(
            self.data_dir,
            target_size=(self.input_size, self.input_size),
            batch_size=self.batch_size,
            class_mode='categorical',
            classes=self.CLASS_NAMES,  # Only use the 7 expected classes
            subset='training',
            shuffle=True
        )
        
        # Validation generator - explicitly specify classes
        validation_generator = validation_datagen.flow_from_directory(
            self.data_dir,
            target_size=(self.input_size, self.input_size),
            batch_size=self.batch_size,
            class_mode='categorical',
            classes=self.CLASS_NAMES,  # Only use the 7 expected classes
            subset='validation',
            shuffle=False
        )
        
        # Verify class counts match and log class information
        logger.info(f"Found {len(train_generator.class_indices)} classes in dataset")
        logger.info(f"Class mapping: {train_generator.class_indices}")
        
        if len(train_generator.class_indices) != len(self.CLASS_NAMES):
            found_classes = list(train_generator.class_indices.keys())
            expected_classes = self.CLASS_NAMES
            missing_classes = set(expected_classes) - set(found_classes)
            extra_classes = set(found_classes) - set(expected_classes)
            
            error_msg = f"Class mismatch! Expected {len(self.CLASS_NAMES)} classes, found {len(train_generator.class_indices)}.\n"
            if missing_classes:
                error_msg += f"Missing classes: {missing_classes}\n"
            if extra_classes:
                error_msg += f"Extra classes: {extra_classes}\n"
            error_msg += f"Expected: {expected_classes}\n"
            error_msg += f"Found: {found_classes}"
            raise ValueError(error_msg)
        
        return train_generator, validation_generator
    
    def train(self):
        """Train the model"""
        # Check if resuming from checkpoint
        if self.resume_from_checkpoint and os.path.exists(self.resume_from_checkpoint):
            logger.info(f"Resuming training from checkpoint: {self.resume_from_checkpoint}")
            self.model = keras.models.load_model(self.resume_from_checkpoint)
            # Try to extract epoch number from checkpoint filename
            try:
                import re
                epoch_match = re.search(r'epoch[_-]?(\d+)', self.resume_from_checkpoint)
                if epoch_match:
                    self.initial_epoch = int(epoch_match.group(1))
                    logger.info(f"Resuming from epoch {self.initial_epoch + 1}")
                else:
                    logger.warning("Could not determine epoch number from checkpoint filename.")
            except Exception as e:
                logger.warning(f"Could not extract epoch number: {e}")
        else:
            # Validate images first to remove corrupted files (only if not resuming)
            logger.info("Step 1: Validating images...")
            valid_count, invalid_count, removed_files = self.validate_images()
            
            if valid_count == 0:
                raise ValueError("No valid images found in the dataset! Cannot proceed with training.")
            
            if invalid_count > 0:
                logger.warning(f"{invalid_count} corrupted images were removed. Training will continue with {valid_count} valid images.")
            
            logger.info("Step 2: Building model...")
            self.model = self.build_model()
        
        logger.info("Model architecture:")
        self.model.summary()
        
        logger.info("Step 3: Preparing data generators...")
        train_gen, val_gen = self.prepare_data_generators()
        
        # Check if generators found any images
        if train_gen.samples == 0:
            raise ValueError("No images found in training set! Please check your data directory.")
        
        # Verify the model output size matches the number of classes from data
        last_layer_units = self.model.layers[-1].output_shape[-1]
        num_classes_found = len(train_gen.class_indices)
        if last_layer_units != num_classes_found:
            raise ValueError(
                f"Model output size ({last_layer_units}) doesn't match number of classes found in data ({num_classes_found}). "
                f"Expected {len(self.CLASS_NAMES)} classes: {self.CLASS_NAMES}"
            )
        
        logger.info(f"Training samples: {train_gen.samples}")
        logger.info(f"Validation samples: {val_gen.samples}")
        logger.info(f"Number of classes: {num_classes_found} (matches model output size: {last_layer_units})")
        
        # Setup checkpoint callback - saves every epoch
        checkpoint_path = os.path.join(
            self.checkpoint_dir, 
            "checkpoint-epoch_{epoch:02d}-val_acc_{val_accuracy:.4f}.h5"
        )
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            # Save best model (final output)
            keras.callbacks.ModelCheckpoint(
                self.model_save_path,
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
            # Save checkpoint every epoch (for resume)
            keras.callbacks.ModelCheckpoint(
                checkpoint_path,
                monitor='val_accuracy',
                save_best_only=False,  # Save every epoch
                verbose=0  # Less verbose for frequent saves
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=0.00001
            )
        ]
        
        logger.info("Starting training...")
        if self.initial_epoch > 0:
            logger.info(f"Training will continue from epoch {self.initial_epoch + 1} to {self.epochs}")
        
        history = self.model.fit(
            train_gen,
            epochs=self.epochs,
            initial_epoch=self.initial_epoch,  # Resume from this epoch
            validation_data=val_gen,
            callbacks=callbacks,
            verbose=1
        )
        
        logger.info(f"Training completed. Model saved to {self.model_save_path}")
        
        # Save training history
        self.save_training_history(history)
        
        return history
    
    def save_training_history(self, history):
        """Save training history plots"""
        try:
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            
            # Plot accuracy
            axes[0].plot(history.history['accuracy'], label='Training Accuracy')
            axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy')
            axes[0].set_title('Model Accuracy')
            axes[0].set_xlabel('Epoch')
            axes[0].set_ylabel('Accuracy')
            axes[0].legend()
            
            # Plot loss
            axes[1].plot(history.history['loss'], label='Training Loss')
            axes[1].plot(history.history['val_loss'], label='Validation Loss')
            axes[1].set_title('Model Loss')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Loss')
            axes[1].legend()
            
            plt.tight_layout()
            plt.savefig('./models/training_history.png')
            logger.info("Training history saved to ./models/training_history.png")
        except Exception as e:
            logger.warning(f"Could not save training history: {str(e)}")


if __name__ == "__main__":
    trainer = SkinDiseaseModelTrainer(
        data_dir="./data/processed",
        model_save_path="./models/skin_disease_model.h5",
        epochs=50
    )
    trainer.train()

