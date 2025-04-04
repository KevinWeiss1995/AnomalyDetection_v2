import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow import keras
import tensorflow as tf
import json

from model import create_model
from optimizers.kfac import KFACCallback
from optimizers.cyclic_lr import CyclicLR
from losses.focal_loss import focal_loss

def train_with_cross_validation(X, y, n_splits=5):
    """
    Trains model using k-fold CV to check how well it generalizes.
    Returns validation scores for each fold and final trained model.
    
    Args:
        X: Training data
        y: Training labels
        n_splits: Number of CV folds
    
    Returns:
        fold_scores: List of validation scores for each fold
        final_model: Trained model on full dataset
    """
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_scores = []

    print("\nPerforming 5-fold cross-validation...")
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X, y)):
        print(f'\nFold {fold + 1}')
        
        X_train_fold = X[train_idx]
        y_train_fold = y[train_idx]
        X_val_fold = X[val_idx]
        y_val_fold = y[val_idx]
        
        model = create_model(X.shape[1])
        print("\nModel Architecture:")
        model.summary() 
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss=focal_loss(gamma=3.0, alpha=0.3),
            metrics=['accuracy', keras.metrics.AUC(),
                    keras.metrics.Precision(), 
                    keras.metrics.Recall()]
        )
        
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            KFACCallback(damping=0.001, momentum=0.9),
            CyclicLR(base_lr=0.00005, max_lr=0.0005, step_size=2000)
        ]
        
        history = model.fit(
            X_train_fold, y_train_fold,
            epochs=100,
            batch_size=64,  
            validation_data=(X_val_fold, y_val_fold),
            callbacks=callbacks,
            verbose=1
        )
        
        scores = model.evaluate(X_val_fold, y_val_fold, verbose=0)
        fold_scores.append(scores[1])

    print(f"\nCross-validation scores: {fold_scores}")
    print(f"Mean CV accuracy: {np.mean(fold_scores):.3f} (+/- {np.std(fold_scores) * 2:.3f})")
    
    return fold_scores

def train_final_model(X, y, X_test, y_test):
    """
    Trains the final model on all training data and evaluates on test set.
    Returns the trained model ready for predictions.
    
    Args:
        X: Training data
        y: Training labels
        X_test: Test data
        y_test: Test labels
    
    Returns:
        final_model: Trained model
    """
    final_model = create_model(X.shape[1])
    final_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.AUC()]
    )

    final_history = final_model.fit(
        X, y,
        epochs=100,
        batch_size=64, 
        validation_split=0.2,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            )
        ],
        verbose=1
    )


    with open('final_history.json', 'w') as f:
        json.dump(final_history.history, f)

    predictions = (final_model.predict(X_test) > 0.5).astype(int)
    print("\nFinal Model Results:")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, predictions))
    print("\nClassification Report:")
    print(classification_report(y_test, predictions))
    
    return final_model 