import os
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, f1_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_class_weight
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
import pickle
from datetime import datetime
warnings.filterwarnings('ignore')

# Grafik ve görselleştirme ayarları
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# GPU Optimization
try:
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"🚀 GPU enabled: {len(gpus)} GPU(s)")
    else:
        print("💻 Using CPU")
    tf.keras.backend.clear_session()
except Exception as e:
    print(f"💻 GPU setup: {e}")

# Paths
project_path = "/content/drive/MyDrive/makale_calismalari/knee"
dataset_path = os.path.join(project_path, "dataset")
results_path = os.path.join(project_path, "results")
models_path = os.path.join(project_path, "models")
visualizations_path = os.path.join(project_path, "visualizations")

# Create directories
for path in [results_path, models_path, visualizations_path]:
    os.makedirs(path, exist_ok=True)

class AdvancedPreprocessing:
    """Advanced preprocessing for maximum accuracy"""

    def __init__(self, target_size=(224, 224)):
        self.target_size = target_size

    def advanced_clahe(self, image):
        """Advanced CLAHE with adaptive parameters"""
        hist = cv2.calcHist([image], [0], None, [256], [0, 256])
        peak_intensity = np.argmax(hist)

        if peak_intensity < 85:  # Dark image
            clip_limit = 4.0
            grid_size = (16, 16)
        elif peak_intensity > 170:  # Bright image
            clip_limit = 2.5
            grid_size = (8, 8)
        else:  # Normal image
            clip_limit = 3.0
            grid_size = (12, 12)

        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
        return clahe.apply(image)

    def unsharp_masking(self, image, strength=1.8):
        """Unsharp masking for edge enhancement"""
        gaussian = cv2.GaussianBlur(image, (9, 9), 2.0)
        unsharp_mask = cv2.addWeighted(image, 1.0 + strength, gaussian, -strength, 0)
        return np.clip(unsharp_mask, 0, 255).astype(np.uint8)

    def morphological_enhancement(self, image):
        """Morphological operations for structural details"""
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        opened = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
        tophat = cv2.morphologyEx(opened, cv2.MORPH_TOPHAT, kernel)
        enhanced = cv2.add(opened, tophat)
        blackhat = cv2.morphologyEx(enhanced, cv2.MORPH_BLACKHAT, kernel)
        enhanced = cv2.subtract(enhanced, blackhat)
        return enhanced

    def multi_scale_enhancement(self, image, version=0):
        """Multiple enhancement strategies"""
        if version == 0:
            enhanced = self.advanced_clahe(image)
            enhanced = cv2.bilateralFilter(enhanced, 9, 75, 75)
        elif version == 1:
            enhanced = self.advanced_clahe(image)
            enhanced = self.unsharp_masking(enhanced, strength=2.0)
            enhanced = cv2.bilateralFilter(enhanced, 11, 80, 80)
        elif version == 2:
            enhanced = self.advanced_clahe(image)
            enhanced = self.morphological_enhancement(enhanced)
            enhanced = cv2.bilateralFilter(enhanced, 9, 75, 75)
        elif version == 3:
            enhanced = self.advanced_clahe(image)
            gamma = 1.3
            enhanced = np.power(enhanced / 255.0, gamma) * 255.0
            enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
            enhanced = cv2.bilateralFilter(enhanced, 9, 75, 75)
        else:
            enhanced = cv2.equalizeHist(image)
            enhanced = cv2.bilateralFilter(enhanced, 7, 50, 50)
        return enhanced

    def preprocess_medical_image(self, img_path, version=0):
        """Advanced medical image preprocessing"""
        try:
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if img is None:
                return None

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            enhanced = self.multi_scale_enhancement(gray, version)
            resized = cv2.resize(enhanced, self.target_size, interpolation=cv2.INTER_LANCZOS4)
            rgb_img = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
            normalized = rgb_img.astype('float32') / 255.0

            # ImageNet normalization
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            normalized = (normalized - mean) / std

            return normalized

        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            return None

class DenseNetKneeClassifier:
    """DenseNet201 based knee OA classifier with deep feature engineering"""

    def __init__(self):
        self.preprocessor = AdvancedPreprocessing()
        self.model = None
        self.feature_extractor = None
        self.deep_features = None
        self.feature_classifiers = {}
        self.history = None
        self.class_names = ['Grade_0', 'Grade_1', 'Grade_2', 'Grade_3', 'Grade_4']

    def load_balanced_data(self, samples_per_class=600):
        """Load balanced data with advanced preprocessing"""
        print("🔬 Loading data with advanced preprocessing...")

        all_images = []
        all_labels = []
        all_paths = []

        grades = ['Grade_0', 'Grade_1', 'Grade_2', 'Grade_3', 'Grade_4']

        for grade_idx, grade in enumerate(grades):
            grade_path = os.path.join(dataset_path, grade)

            if not os.path.exists(grade_path):
                print(f"❌ Path not found: {grade_path}")
                continue

            image_files = [f for f in os.listdir(grade_path)
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

            print(f"   📁 {grade}: {len(image_files)} images found")

            selected_files = image_files[:samples_per_class]

            for img_file in tqdm(selected_files, desc=f"Processing {grade}"):
                img_path = os.path.join(grade_path, img_file)

                # Use 5 enhancement versions for diversity
                for version in range(5):
                    processed_img = self.preprocessor.preprocess_medical_image(img_path, version)

                    if processed_img is not None:
                        all_images.append(processed_img)
                        all_labels.append(grade_idx)
                        all_paths.append(img_path)

                # Memory management
                if len(all_images) % 200 == 0:
                    import gc
                    gc.collect()

        X = np.array(all_images, dtype=np.float32)
        y = np.array(all_labels, dtype=np.int32)
        paths = np.array(all_paths)

        # Cleanup
        del all_images, all_labels
        import gc
        gc.collect()

        # Shuffle
        indices = np.random.permutation(len(X))
        X = X[indices]
        y = y[indices]
        paths = paths[indices]

        print(f"✅ Dataset prepared: {X.shape}")
        print(f"💾 Memory usage: {X.nbytes / 1024**3:.2f} GB")

        # Class distribution
        unique, counts = np.unique(y, return_counts=True)
        print("📊 Class distribution:")
        for grade_idx, count in zip(unique, counts):
            print(f"   Grade_{grade_idx}: {count} samples ({count/len(y)*100:.1f}%)")

        return X, y, paths

    def create_densenet_model(self, input_shape=(224, 224, 3), num_classes=5):
        """Create optimized DenseNet201 model"""
        print("🏗️ Creating optimized DenseNet201 model...")

        base_model = tf.keras.applications.DenseNet201(
            weights='imagenet',
            include_top=False,
            input_shape=input_shape
        )

        # Unfreeze top layers for fine-tuning
        for layer in base_model.layers[-60:]:
            layer.trainable = True

        inputs = tf.keras.layers.Input(shape=input_shape)
        x = base_model(inputs, training=False)

        # Multi-scale pooling
        gap = tf.keras.layers.GlobalAveragePooling2D()(x)
        gmp = tf.keras.layers.GlobalMaxPooling2D()(x)
        x = tf.keras.layers.Concatenate()([gap, gmp])

        # Batch normalization and dropout
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.4)(x)

        # Dense layers
        x = tf.keras.layers.Dense(512, activation='relu', name='feature_dense')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.3)(x)

        outputs = tf.keras.layers.Dense(num_classes, activation='softmax', name='classification_output')(x)

        model = tf.keras.Model(inputs, outputs, name='DenseNet201_Optimized')

        # Compile model
        model.compile(
            optimizer=tf.keras.optimizers.AdamW(
                learning_rate=1e-4,
                weight_decay=0.01,
                beta_1=0.9,
                beta_2=0.999
            ),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        self.model = model
        return model

    def create_feature_extractor(self):
        """Create feature extractor from trained model"""
        if self.model is None:
            print("❌ Model not found!")
            return None

        # Extract features from the dense layer before classification
        feature_layer = self.model.get_layer('feature_dense')
        self.feature_extractor = tf.keras.Model(
            inputs=self.model.input,
            outputs=feature_layer.output,
            name='Feature_Extractor'
        )

        print("✅ Feature extractor created")
        return self.feature_extractor

    def train_model(self, X_train, X_val, y_train, y_val):
        """Train DenseNet201 model"""
        print("🚀 Training DenseNet201 model...")

        # Class weights
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(y_train),
            y=y_train
        )
        class_weight_dict = dict(enumerate(class_weights))

        # Advanced callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_accuracy',
                factor=0.3,
                patience=8,
                min_lr=1e-7,
                verbose=1
            ),
            tf.keras.callbacks.ModelCheckpoint(
                os.path.join(models_path, 'best_densenet_model.h5'),
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
            tf.keras.callbacks.LearningRateScheduler(
                lambda epoch: 1e-4 * (0.95 ** epoch),
                verbose=0
            )
        ]

        # Data augmentation
        train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rotation_range=30,
            width_shift_range=0.2,
            height_shift_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            brightness_range=[0.7, 1.3],
            shear_range=0.15,
            fill_mode='nearest'
        )

        train_generator = train_datagen.flow(
            X_train, y_train,
            batch_size=16,
            shuffle=True
        )

        val_generator = tf.keras.preprocessing.image.ImageDataGenerator().flow(
            X_val, y_val,
            batch_size=16,
            shuffle=False
        )

        # Train model
        self.history = self.model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=45,
            steps_per_epoch=len(X_train) // 16,
            validation_steps=len(X_val) // 16,
            class_weight=class_weight_dict,
            callbacks=callbacks,
            verbose=1
        )

        # Evaluate
        val_acc = self.model.evaluate(X_val, y_val, verbose=0, batch_size=16)[1]
        print(f"✅ DenseNet201 validation accuracy: {val_acc:.4f}")

        return self.history

    def extract_deep_features(self, X_data):
        """Extract deep features using trained model"""
        print("🔍 Extracting deep features...")

        if self.feature_extractor is None:
            self.create_feature_extractor()

        features = self.feature_extractor.predict(X_data, batch_size=8, verbose=1)
        self.deep_features = features

        print(f"✅ Deep features extracted: {features.shape}")
        return features

    def engineer_statistical_features(self, features):
        """Engineer statistical features from deep features"""
        print("🔧 Engineering statistical features...")

        statistical_features = []

        # Basic statistics
        mean_features = np.mean(features, axis=1, keepdims=True)
        std_features = np.std(features, axis=1, keepdims=True)
        max_features = np.max(features, axis=1, keepdims=True)
        min_features = np.min(features, axis=1, keepdims=True)

        # Advanced statistics
        skewness = []
        kurtosis = []
        percentiles_25 = []
        percentiles_75 = []
        entropy = []

        for i in range(features.shape[0]):
            feature_vector = features[i]

            # Skewness and kurtosis
            from scipy.stats import skew, kurtosis as kurt
            skewness.append([skew(feature_vector)])
            kurtosis.append([kurt(feature_vector)])

            # Percentiles
            percentiles_25.append([np.percentile(feature_vector, 25)])
            percentiles_75.append([np.percentile(feature_vector, 75)])

            # Shannon entropy
            hist, _ = np.histogram(feature_vector, bins=50, density=True)
            hist = hist + 1e-8  # Avoid log(0)
            entropy.append([-np.sum(hist * np.log(hist))])

        skewness = np.array(skewness)
        kurtosis = np.array(kurtosis)
        percentiles_25 = np.array(percentiles_25)
        percentiles_75 = np.array(percentiles_75)
        entropy = np.array(entropy)

        # Combine all statistical features
        enhanced_features = np.hstack([
            features,
            mean_features,
            std_features,
            max_features,
            min_features,
            skewness,
            kurtosis,
            percentiles_25,
            percentiles_75,
            entropy
        ])

        print(f"✅ Statistical features engineered: {enhanced_features.shape}")
        return enhanced_features

    def train_feature_classifiers(self, X_features, y_train):
        """Train multiple classifiers on deep features"""
        print("🎯 Training feature-based classifiers...")

        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_features)

        classifiers = {
            'RandomForest': RandomForestClassifier(
                n_estimators=1000,
                max_depth=15,
                min_samples_split=5,
                random_state=42,
                class_weight='balanced',
                n_jobs=-1
            ),
            'GradientBoosting': GradientBoostingClassifier(
                n_estimators=500,
                max_depth=10,
                learning_rate=0.1,
                random_state=42
            ),
            'SVM': SVC(
                kernel='rbf',
                C=10,
                gamma='scale',
                class_weight='balanced',
                probability=True,
                random_state=42
            )
        }

        trained_classifiers = {}

        for name, clf in classifiers.items():
            print(f"   Training {name}...")

            # Cross-validation
            cv_scores = cross_val_score(clf, X_scaled, y_train, cv=5, scoring='accuracy', n_jobs=-1)
            print(f"   {name} CV accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

            # Train on full data
            clf.fit(X_scaled, y_train)
            trained_classifiers[name] = clf

        self.feature_classifiers = trained_classifiers
        self.feature_scaler = scaler

        print("✅ Feature classifiers trained")
        return trained_classifiers

    def evaluate_comprehensive(self, X_test, y_test):
        """Comprehensive evaluation of all approaches"""
        print("\n🏆 Comprehensive Evaluation...")

        results = []

        # 1. Original DenseNet201 evaluation
        print("📊 DenseNet201 Direct Classification:")
        y_pred_proba = self.model.predict(X_test, verbose=0, batch_size=8)
        y_pred = np.argmax(y_pred_proba, axis=1)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')

        print(f"   Accuracy: {acc:.4f}")
        print(f"   F1-Score: {f1:.4f}")
        print(f"   AUC: {auc:.4f}")

        results.append({
            'Method': 'DenseNet201_Direct',
            'Accuracy': acc,
            'F1_Score': f1,
            'AUC': auc
        })

        # 2. Deep feature extraction and engineering
        print("\n📊 Deep Feature Based Classification:")
        test_features = self.extract_deep_features(X_test)
        enhanced_features = self.engineer_statistical_features(test_features)
        X_test_scaled = self.feature_scaler.transform(enhanced_features)

        # Evaluate feature-based classifiers
        for name, clf in self.feature_classifiers.items():
            y_pred = clf.predict(X_test_scaled)
            y_pred_proba = clf.predict_proba(X_test_scaled)

            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')

            print(f"   {name}: Acc={acc:.4f}, F1={f1:.4f}, AUC={auc:.4f}")

            results.append({
                'Method': f'DeepFeatures_{name}',
                'Accuracy': acc,
                'F1_Score': f1,
                'AUC': auc
            })

        # 3. Ensemble approach
        print("\n📊 Ensemble Approach:")
        ensemble_predictions = []

        # DenseNet direct prediction
        densenet_pred = self.model.predict(X_test, verbose=0, batch_size=8)
        ensemble_predictions.append(densenet_pred)

        # Feature-based predictions
        for clf in self.feature_classifiers.values():
            feat_pred = clf.predict_proba(X_test_scaled)
            ensemble_predictions.append(feat_pred)

        # Average ensemble
        ensemble_pred_proba = np.mean(ensemble_predictions, axis=0)
        ensemble_pred = np.argmax(ensemble_pred_proba, axis=1)

        acc = accuracy_score(y_test, ensemble_pred)
        f1 = f1_score(y_test, ensemble_pred, average='weighted')
        auc = roc_auc_score(y_test, ensemble_pred_proba, multi_class='ovr', average='weighted')

        print(f"   Ensemble: Acc={acc:.4f}, F1={f1:.4f}, AUC={auc:.4f}")

        results.append({
            'Method': 'Ensemble_All',
            'Accuracy': acc,
            'F1_Score': f1,
            'AUC': auc
        })

        return results, y_pred, ensemble_pred_proba

    def create_visualizations(self, X_test, y_test, y_pred, y_pred_proba):
        """Create comprehensive visualizations"""
        print("\n🎨 Creating comprehensive visualizations...")

        # 1. Training history plots
        if self.history is not None:
            self.plot_training_history()

        # 2. Confusion matrix
        self.plot_confusion_matrix(y_test, y_pred)

        # 3. ROC curves
        self.plot_roc_curves(y_test, y_pred_proba)

        # 4. Class distribution
        self.plot_class_distribution(y_test, y_pred)

        # 5. t-SNE visualization
        if self.deep_features is not None:
            self.plot_tsne_visualization(y_test)

        # 6. Feature importance (for Random Forest)
        if 'RandomForest' in self.feature_classifiers:
            self.plot_feature_importance()

        # 7. GradCAM visualization
        self.create_gradcam_visualizations(X_test, y_test)

        print("✅ All visualizations created")

    def plot_training_history(self):
        """Plot training history"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        # Accuracy
        ax1.plot(self.history.history['accuracy'], label='Train Accuracy', linewidth=2)
        ax1.plot(self.history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
        ax1.set_title('Model Accuracy', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Loss
        ax2.plot(self.history.history['loss'], label='Train Loss', linewidth=2)
        ax2.plot(self.history.history['val_loss'], label='Validation Loss', linewidth=2)
        ax2.set_title('Model Loss', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Learning rate (if available)
        if 'lr' in self.history.history:
            ax3.plot(self.history.history['lr'], linewidth=2, color='red')
            ax3.set_title('Learning Rate', fontsize=14, fontweight='bold')
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('Learning Rate')
            ax3.set_yscale('log')
            ax3.grid(True, alpha=0.3)

        # Validation accuracy zoomed
        ax4.plot(self.history.history['val_accuracy'], linewidth=2, color='green')
        ax4.set_title('Validation Accuracy (Zoomed)', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Validation Accuracy')
        ax4.set_ylim([min(self.history.history['val_accuracy']) - 0.01,
                      max(self.history.history['val_accuracy']) + 0.01])
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(visualizations_path, 'training_history.png'), dpi=300, bbox_inches='tight')
        plt.show()

    def plot_confusion_matrix(self, y_true, y_pred):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.class_names,
                    yticklabels=self.class_names,
                    cbar_kws={'label': 'Number of Samples'})
        plt.title('Confusion Matrix - DenseNet201 Classification', fontsize=16, fontweight='bold')
        plt.xlabel('Predicted Grade', fontsize=12)
        plt.ylabel('True Grade', fontsize=12)

        # Add accuracy per class
        accuracy_per_class = cm.diagonal() / cm.sum(axis=1)
        for i, acc in enumerate(accuracy_per_class):
            plt.text(i, i-0.3, f'{acc:.3f}', ha='center', va='center',
                    fontweight='bold', color='white', fontsize=10)

        plt.tight_layout()
        plt.savefig(os.path.join(visualizations_path, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
        plt.show()

    def plot_roc_curves(self, y_true, y_pred_proba):
        """Plot ROC curves for each class"""
        from sklearn.metrics import roc_curve, auc
        from sklearn.preprocessing import label_binarize

        # Binarize labels
        y_true_bin = label_binarize(y_true, classes=[0, 1, 2, 3, 4])
        n_classes = y_true_bin.shape[1]

        plt.figure(figsize=(12, 8))
        colors = ['red', 'blue', 'green', 'orange', 'purple']

        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
            roc_auc = auc(fpr, tpr)

            plt.plot(fpr, tpr, color=colors[i], linewidth=2,
                    label=f'Grade_{i} (AUC = {roc_auc:.3f})')

        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves - Multi-class Classification', fontsize=16, fontweight='bold')
        plt.legend(loc="lower right", fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(visualizations_path, 'roc_curves.png'), dpi=300, bbox_inches='tight')
        plt.show()

    def plot_class_distribution(self, y_true, y_pred):
        """Plot class distribution comparison"""
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

        # True distribution
        unique_true, counts_true = np.unique(y_true, return_counts=True)
        ax1.bar([f'Grade_{i}' for i in unique_true], counts_true, color='skyblue', alpha=0.7)
        ax1.set_title('True Class Distribution', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Number of Samples')
        for i, v in enumerate(counts_true):
            ax1.text(i, v + 0.5, str(v), ha='center', fontweight='bold')

        # Predicted distribution
        unique_pred, counts_pred = np.unique(y_pred, return_counts=True)
        ax2.bar([f'Grade_{i}' for i in unique_pred], counts_pred, color='lightcoral', alpha=0.7)
        ax2.set_title('Predicted Class Distribution', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Number of Samples')
        for i, v in enumerate(counts_pred):
            ax2.text(i, v + 0.5, str(v), ha='center', fontweight='bold')

        # Comparison
        grades = [f'Grade_{i}' for i in range(5)]
        true_counts = [np.sum(y_true == i) for i in range(5)]
        pred_counts = [np.sum(y_pred == i) for i in range(5)]

        x = np.arange(len(grades))
        width = 0.35

        ax3.bar(x - width/2, true_counts, width, label='True', color='skyblue', alpha=0.7)
        ax3.bar(x + width/2, pred_counts, width, label='Predicted', color='lightcoral', alpha=0.7)
        ax3.set_title('Class Distribution Comparison', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Number of Samples')
        ax3.set_xticks(x)
        ax3.set_xticklabels(grades)
        ax3.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(visualizations_path, 'class_distribution.png'), dpi=300, bbox_inches='tight')
        plt.show()

    def plot_tsne_visualization(self, y_true, perplexity=30, n_iter=1000):
        """Plot t-SNE visualization of deep features"""
        print("   Creating t-SNE visualization...")

        # Sample subset for t-SNE (computational efficiency)
        if len(self.deep_features) > 1000:
            indices = np.random.choice(len(self.deep_features), 1000, replace=False)
            features_subset = self.deep_features[indices]
            labels_subset = y_true[indices]
        else:
            features_subset = self.deep_features
            labels_subset = y_true

        # Apply t-SNE
        tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter,
                   random_state=42, verbose=1)
        features_2d = tsne.fit_transform(features_subset)

        # Create visualization
        plt.figure(figsize=(12, 8))
        colors = ['red', 'blue', 'green', 'orange', 'purple']

        for i in range(5):
            mask = labels_subset == i
            plt.scatter(features_2d[mask, 0], features_2d[mask, 1],
                       c=colors[i], label=f'Grade_{i}', alpha=0.7, s=50)

        plt.title('t-SNE Visualization of Deep Features', fontsize=16, fontweight='bold')
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(visualizations_path, 'tsne_visualization.png'), dpi=300, bbox_inches='tight')
        plt.show()

    def plot_feature_importance(self):
        """Plot feature importance from Random Forest"""
        if 'RandomForest' not in self.feature_classifiers:
            return

        rf = self.feature_classifiers['RandomForest']
        importances = rf.feature_importances_

        # Get top 30 most important features
        indices = np.argsort(importances)[::-1][:30]

        plt.figure(figsize=(12, 8))
        plt.bar(range(30), importances[indices], color='steelblue', alpha=0.7)
        plt.title('Top 30 Feature Importances (Random Forest)', fontsize=16, fontweight='bold')
        plt.xlabel('Feature Index')
        plt.ylabel('Importance')
        plt.xticks(range(30), [f'F{i}' for i in indices], rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(visualizations_path, 'feature_importance.png'), dpi=300, bbox_inches='tight')
        plt.show()

    def create_gradcam_visualizations(self, X_test, y_test, num_samples=10):
        """Create GradCAM visualizations"""
        print("   Creating GradCAM visualizations...")

        # Select representative samples from each class
        selected_indices = []
        for class_id in range(5):
            class_indices = np.where(y_test == class_id)[0]
            if len(class_indices) > 0:
                selected_indices.extend(class_indices[:2])  # 2 samples per class

        selected_indices = selected_indices[:num_samples]

        fig, axes = plt.subplots(2, len(selected_indices), figsize=(20, 8))
        if len(selected_indices) == 1:
            axes = axes.reshape(2, 1)

        for idx, sample_idx in enumerate(selected_indices):
            # Original image
            original_img = X_test[sample_idx]

            # Denormalize for display
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            display_img = original_img * std + mean
            display_img = np.clip(display_img, 0, 1)

            # Generate GradCAM
            gradcam_heatmap = self.generate_gradcam(
                np.expand_dims(original_img, axis=0),
                class_index=y_test[sample_idx]
            )

            # Plot original
            axes[0, idx].imshow(display_img)
            axes[0, idx].set_title(f'Original - Grade_{y_test[sample_idx]}', fontsize=10)
            axes[0, idx].axis('off')

            # Plot GradCAM overlay
            axes[1, idx].imshow(display_img)
            axes[1, idx].imshow(gradcam_heatmap, cmap='jet', alpha=0.4)
            axes[1, idx].set_title(f'GradCAM - Grade_{y_test[sample_idx]}', fontsize=10)
            axes[1, idx].axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(visualizations_path, 'gradcam_visualization.png'), dpi=300, bbox_inches='tight')
        plt.show()

    def generate_gradcam(self, img_array, class_index, layer_name='conv5_block32_2_conv'):
        """Generate GradCAM heatmap"""
        try:
            # Get the last convolutional layer
            last_conv_layer = None
            for layer in reversed(self.model.layers):
                if len(layer.output_shape) == 4:  # Convolutional layer
                    last_conv_layer = layer
                    break

            if last_conv_layer is None:
                # Fallback to a known layer in DenseNet201
                try:
                    last_conv_layer = self.model.get_layer('conv5_block32_2_conv')
                except:
                    # Use the base model's last conv layer
                    for layer in self.model.layers:
                        if 'densenet201' in layer.name.lower():
                            base_model = layer
                            for base_layer in reversed(base_model.layers):
                                if len(base_layer.output_shape) == 4:
                                    last_conv_layer = base_layer
                                    break
                            break

            if last_conv_layer is None:
                print("Could not find convolutional layer for GradCAM")
                return np.zeros((224, 224))

            # Create gradient model
            grad_model = tf.keras.models.Model(
                [self.model.inputs],
                [last_conv_layer.output, self.model.output]
            )

            # Compute gradients
            with tf.GradientTape() as tape:
                conv_outputs, predictions = grad_model(img_array)
                class_channel = predictions[:, class_index]

            grads = tape.gradient(class_channel, conv_outputs)
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

            # Weight the feature maps
            conv_outputs = conv_outputs[0]
            heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
            heatmap = tf.squeeze(heatmap)

            # Normalize heatmap
            heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
            heatmap = heatmap.numpy()

            # Resize to input image size
            heatmap = cv2.resize(heatmap, (224, 224))

            return heatmap

        except Exception as e:
            print(f"GradCAM generation failed: {e}")
            return np.zeros((224, 224))

    def create_performance_report(self, results):
        """Create comprehensive performance report"""
        print("\n📊 Creating performance report...")

        # Convert results to DataFrame
        df_results = pd.DataFrame(results)
        df_results = df_results.sort_values('Accuracy', ascending=False)

        # Create performance comparison plot
        plt.figure(figsize=(14, 8))

        methods = df_results['Method']
        accuracies = df_results['Accuracy'] * 100
        f1_scores = df_results['F1_Score'] * 100
        auc_scores = df_results['AUC'] * 100

        x = np.arange(len(methods))
        width = 0.25

        plt.bar(x - width, accuracies, width, label='Accuracy', alpha=0.8, color='skyblue')
        plt.bar(x, f1_scores, width, label='F1-Score', alpha=0.8, color='lightcoral')
        plt.bar(x + width, auc_scores, width, label='AUC', alpha=0.8, color='lightgreen')

        plt.xlabel('Methods', fontsize=12)
        plt.ylabel('Performance (%)', fontsize=12)
        plt.title('Performance Comparison Across Different Methods', fontsize=16, fontweight='bold')
        plt.xticks(x, methods, rotation=45, ha='right')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Add value labels on bars
        for i, (acc, f1, auc) in enumerate(zip(accuracies, f1_scores, auc_scores)):
            plt.text(i - width, acc + 0.5, f'{acc:.1f}%', ha='center', va='bottom', fontsize=8)
            plt.text(i, f1 + 0.5, f'{f1:.1f}%', ha='center', va='bottom', fontsize=8)
            plt.text(i + width, auc + 0.5, f'{auc:.1f}%', ha='center', va='bottom', fontsize=8)

        plt.tight_layout()
        plt.savefig(os.path.join(visualizations_path, 'performance_comparison.png'), dpi=300, bbox_inches='tight')
        plt.show()

        # Save detailed results
        results_file = os.path.join(results_path, f"detailed_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        df_results.to_csv(results_file, index=False)

        # Print summary
        best_result = df_results.iloc[0]
        print(f"\n{'='*80}")
        print("🏆 DENSENET201 KNEE OA CLASSIFICATION - COMPREHENSIVE RESULTS")
        print(f"{'='*80}")
        print(df_results.to_string(index=False, float_format='%.4f'))

        print(f"\n🏆 BEST PERFORMANCE ACHIEVED:")
        print(f"   Method: {best_result['Method']}")
        print(f"   🎯 ACCURACY: {best_result['Accuracy']:.4f} ({best_result['Accuracy']*100:.2f}%)")
        print(f"   🎯 F1-SCORE: {best_result['F1_Score']:.4f} ({best_result['F1_Score']*100:.2f}%)")
        print(f"   🎯 AUC: {best_result['AUC']:.4f} ({best_result['AUC']*100:.2f}%)")

        return df_results

    def save_models_and_results(self):
        """Save trained models and results"""
        print("\n💾 Saving models and results...")

        # Save main model
        self.model.save(os.path.join(models_path, 'densenet201_knee_classifier.h5'))

        # Save feature extractor
        if self.feature_extractor is not None:
            self.feature_extractor.save(os.path.join(models_path, 'feature_extractor.h5'))

        # Save feature classifiers
        for name, clf in self.feature_classifiers.items():
            with open(os.path.join(models_path, f'{name}_classifier.pkl'), 'wb') as f:
                pickle.dump(clf, f)

        # Save feature scaler
        if hasattr(self, 'feature_scaler'):
            with open(os.path.join(models_path, 'feature_scaler.pkl'), 'wb') as f:
                pickle.dump(self.feature_scaler, f)

        # Save training history
        if self.history is not None:
            with open(os.path.join(results_path, 'training_history.pkl'), 'wb') as f:
                pickle.dump(self.history.history, f)

        print("✅ All models and results saved")

    def run_complete_pipeline(self):
        """Execute complete DenseNet201 pipeline with deep feature engineering"""
        print("🏆 DENSENET201 KNEE OA CLASSIFICATION WITH DEEP FEATURE ENGINEERING")
        print("🎯 ADVANCED PREPROCESSING + DEEP FEATURES + COMPREHENSIVE ANALYSIS")
        print("="*80)

        # Load data
        X, y, paths = self.load_balanced_data(samples_per_class=600)

        # Memory cleanup
        import gc
        gc.collect()

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.15, random_state=42, stratify=y_train
        )

        print(f"\n📊 Data splits:")
        print(f"   Train: {X_train.shape}")
        print(f"   Validation: {X_val.shape}")
        print(f"   Test: {X_test.shape}")
        print(f"   Total samples: {len(X):,}")

        # Create and train DenseNet201 model
        model = self.create_densenet_model()
        history = self.train_model(X_train, X_val, y_train, y_val)

        # Extract and engineer deep features
        print(f"\n🔍 Deep Feature Engineering Phase...")
        train_features = self.extract_deep_features(X_train)
        enhanced_train_features = self.engineer_statistical_features(train_features)

        # Train feature-based classifiers
        trained_classifiers = self.train_feature_classifiers(enhanced_train_features, y_train)

        # Comprehensive evaluation
        results, y_pred, y_pred_proba = self.evaluate_comprehensive(X_test, y_test)

        # Create all visualizations
        self.create_visualizations(X_test, y_test, y_pred, y_pred_proba)

        # Create performance report
        df_results = self.create_performance_report(results)

        # Save everything
        self.save_models_and_results()

        # Final achievement analysis
        best_accuracy = df_results.iloc[0]['Accuracy'] * 100

        print(f"\n{'='*80}")
        print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
        print(f"{'='*80}")

        if best_accuracy >= 95.0:
            print(f"🎉🎉🎉 OUTSTANDING! ACCURACY ≥ 95% - BREAKTHROUGH SUCCESS!")
            print(f"🔥🔥🔥 READY FOR TOP-TIER SCI PUBLICATION!")
            print(f"🏆🏆🏆 WORLD-CLASS PERFORMANCE!")
        elif best_accuracy >= 92.0:
            print(f"🎉🎉 EXCELLENT! ACCURACY ≥ 92% - MAJOR BREAKTHROUGH!")
            print(f"🔥🔥 HIGH-IMPACT PUBLICATION CANDIDATE!")
        elif best_accuracy >= 90.0:
            print(f"🎉 GREAT! ACCURACY ≥ 90% - STRONG ACHIEVEMENT!")
            print(f"🔥 PUBLICATION READY!")
        else:
            print(f"📊 Achieved {best_accuracy:.1f}% - Good progress!")

        print(f"\n🔬 TECHNICAL ACHIEVEMENTS:")
        print(f"   ✅ DenseNet201 with Advanced Preprocessing")
        print(f"   ✅ Deep Feature Extraction & Engineering")
        print(f"   ✅ Multiple Statistical Feature Transformations")
        print(f"   ✅ Ensemble of Feature-based Classifiers")
        print(f"   ✅ Comprehensive Evaluation & Visualization")
        print(f"   ✅ GradCAM, t-SNE, ROC, Confusion Matrix")
        print(f"   ✅ Training History & Performance Analysis")

        print(f"\n📁 OUTPUTS GENERATED:")
        print(f"   📊 All visualizations saved in: {visualizations_path}")
        print(f"   🤖 All models saved in: {models_path}")
        print(f"   📈 All results saved in: {results_path}")

        return df_results

def main():
    """Main execution function"""
    print("🚀 DENSENET201 KNEE OA CLASSIFICATION 2025")
    print("🎯 ADVANCED DEEP FEATURE ENGINEERING APPROACH")
    print("🔬 COMPREHENSIVE ANALYSIS FOR SCI PUBLICATION")
    print("="*80)

    # Create classifier instance
    classifier = DenseNetKneeClassifier()

    # Run complete pipeline
    results = classifier.run_complete_pipeline()

    print("\n✅ COMPLETE PIPELINE FINISHED!")
    print("🔬 Ready for scientific publication")
    print("🎯 All analyses completed")
    print("💾 All outputs saved")
    print("🚀 Mission accomplished!")

if __name__ == "__main__":
    print("🚀 Starting DenseNet201 Knee OA Classification...")
    main()
    print("🏁 Program completed successfully!")
