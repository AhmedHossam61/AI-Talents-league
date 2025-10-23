"""
Improved Prediction Model for Y
This script provides a complete pipeline with:
- Better data preprocessing
- Multiple model comparison
- Proper validation
- Feature engineering
- Hyperparameter tuning (optional)
"""

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

print("=" * 80)
print("IMPROVED PREDICTION MODEL FOR Y")
print("=" * 80)

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("\n[STEP 1] Loading Data...")
df_train = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")

print(f"Train shape: {df_train.shape}")
print(f"Test shape: {df_test.shape}")

# ============================================================================
# 2. DATA EXPLORATION
# ============================================================================
print("\n" + "=" * 80)
print("[STEP 2] Data Exploration")
print("=" * 80)

print("\nFirst few rows:")
print(df_train.head())

print("\nMissing values (%):")
print(df_train.isnull().sum() / len(df_train) * 100)

print("\nTarget variable statistics:")
print(df_train['Y'].describe())

# ============================================================================
# 3. DATA PREPROCESSING
# ============================================================================
print("\n" + "=" * 80)
print("[STEP 3] Data Preprocessing")
print("=" * 80)

# Function to preprocess data
def preprocess_data(df, is_train=True):
    """Preprocess the dataframe"""
    df = df.copy()
    
    # Drop X1 (ID column - not useful for prediction)
    if 'X1' in df.columns:
        df = df.drop('X1', axis=1)
    
    # Standardize X3 values (Low Fat vs Regular)
    if 'X3' in df.columns:
        df['X3'] = df['X3'].replace({
            'low fat': 'Low Fat', 
            'LF': 'Low Fat', 
            'reg': 'Regular'
        })
    
    # Handle X9 (lots of missing values)
    # Option 1: Drop it (as you did) - uncomment if you prefer
    # df = df.drop('X9', axis=1)
    
    # Option 2: Keep it and fill with mode (better for some models)
    if 'X9' in df.columns:
        df['X9'] = df['X9'].fillna('Medium')  # Fill with most common value
    
    # Fill missing values in X2 with mean
    if 'X2' in df.columns:
        df['X2'] = df['X2'].fillna(df['X2'].mean())
    
    # Remove duplicates (only for training data)
    if is_train:
        df = df.drop_duplicates()
    
    return df

# Preprocess both datasets
df_train_processed = preprocess_data(df_train, is_train=True)
df_test_processed = preprocess_data(df_test, is_train=False)

print(f"\nAfter preprocessing:")
print(f"Train shape: {df_train_processed.shape}")
print(f"Test shape: {df_test_processed.shape}")
print(f"Missing values in train: {df_train_processed.isnull().sum().sum()}")

# ============================================================================
# 4. FEATURE ENGINEERING
# ============================================================================
print("\n" + "=" * 80)
print("[STEP 4] Feature Engineering")
print("=" * 80)

def engineer_features(df):
    """Create new features"""
    df = df.copy()
    
    # Store age (years since establishment)
    df['X8_Age'] = 2025 - df['X8']
    
    # Visibility categories
    if 'X4' in df.columns:
        df['X4_IsZero'] = (df['X4'] == 0).astype(int)
        df['X4_Category'] = pd.cut(df['X4'], 
                                     bins=[-0.001, 0, 0.05, 0.1, 1.0],
                                     labels=['Zero', 'Low', 'Medium', 'High'])
    
    # Price categories
    if 'X2' in df.columns:
        df['X2_Category'] = pd.cut(df['X2'], 
                                     bins=[0, 10, 15, 20, 100],
                                     labels=['Low', 'Medium', 'High', 'Very High'])
    
    # Interaction features
    if 'X2' in df.columns and 'X6' in df.columns:
        df['X2_X6_Interaction'] = df['X2'] * df['X6']
    
    return df

# Apply feature engineering
df_train_processed = engineer_features(df_train_processed)
df_test_processed = engineer_features(df_test_processed)

print("✓ Created new features:")
print("  - X8_Age (years since establishment)")
print("  - X4_IsZero (zero visibility flag)")
print("  - X4_Category (visibility categories)")
print("  - X2_Category (price categories)")
print("  - X2_X6_Interaction (price × MRP interaction)")

# ============================================================================
# 5. ENCODE CATEGORICAL VARIABLES
# ============================================================================
print("\n" + "=" * 80)
print("[STEP 5] Encoding Categorical Variables")
print("=" * 80)

# Identify categorical columns
categorical_cols = df_train_processed.select_dtypes(include=['object']).columns.tolist()

# Remove target variable if present
if 'Y' in categorical_cols:
    categorical_cols.remove('Y')

print(f"\nCategorical columns to encode: {categorical_cols}")

# Encode categorical variables using LabelEncoder
encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    
    # Fit on training data
    df_train_processed[col] = le.fit_transform(df_train_processed[col].astype(str))
    encoders[col] = le
    
    # Transform test data
    # Handle unseen labels in test set
    df_test_processed[col] = df_test_processed[col].astype(str)
    df_test_processed[col] = df_test_processed[col].apply(
        lambda x: le.transform([x])[0] if x in le.classes_ else -1
    )
    
    print(f"  - Encoded {col}: {len(le.classes_)} unique values")

# ============================================================================
# 6. PREPARE TRAIN/VALIDATION SPLIT
# ============================================================================
print("\n" + "=" * 80)
print("[STEP 6] Preparing Train/Validation Split")
print("=" * 80)

# Separate features and target
X = df_train_processed.drop('Y', axis=1)
y = df_train_processed['Y']

print(f"\nFeatures shape: {X.shape}")
print(f"Target shape: {y.shape}")

# Split data
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTraining set: {X_train.shape[0]} samples")
print(f"Validation set: {X_val.shape[0]} samples")

# ============================================================================
# 7. TRAIN MULTIPLE MODELS
# ============================================================================
print("\n" + "=" * 80)
print("[STEP 7] Training Multiple Models")
print("=" * 80)

# Dictionary to store results
results = {}

# Define models
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=10.0),
    'Lasso Regression': Lasso(alpha=0.1),
    'Random Forest': RandomForestRegressor(
        n_estimators=200, 
        max_depth=15,
        min_samples_split=5,
        random_state=42, 
        n_jobs=-1
    ),
    'Gradient Boosting': GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        random_state=42
    )
}

# Train and evaluate each model
for model_name, model in models.items():
    print(f"\n[7.{list(models.keys()).index(model_name) + 1}] Training {model_name}...")
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(X_val)
    
    # Calculate metrics
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    val_rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
    train_mae = mean_absolute_error(y_train, y_pred_train)
    val_mae = mean_absolute_error(y_val, y_pred_val)
    train_r2 = r2_score(y_train, y_pred_train)
    val_r2 = r2_score(y_val, y_pred_val)
    
    # Calculate cross-validation score (on training data)
    cv_scores = cross_val_score(model, X_train, y_train, 
                                 cv=5, 
                                 scoring='neg_mean_squared_error',
                                 n_jobs=-1)
    cv_rmse = np.sqrt(-cv_scores.mean())
    
    # Store results
    results[model_name] = {
        'model': model,
        'train_rmse': train_rmse,
        'val_rmse': val_rmse,
        'train_mae': train_mae,
        'val_mae': val_mae,
        'train_r2': train_r2,
        'val_r2': val_r2,
        'cv_rmse': cv_rmse,
        'predictions': y_pred_val
    }
    
    print(f"  Train RMSE: {train_rmse:.4f} | Val RMSE: {val_rmse:.4f} | CV RMSE: {cv_rmse:.4f}")
    print(f"  Train R²:   {train_r2:.4f} | Val R²:   {val_r2:.4f}")

# ============================================================================
# 8. MODEL COMPARISON
# ============================================================================
print("\n" + "=" * 80)
print("[STEP 8] Model Comparison")
print("=" * 80)

# Create comparison dataframe
comparison_df = pd.DataFrame({
    'Model': list(results.keys()),
    'Val RMSE': [results[m]['val_rmse'] for m in results.keys()],
    'Val MAE': [results[m]['val_mae'] for m in results.keys()],
    'Val R²': [results[m]['val_r2'] for m in results.keys()],
    'CV RMSE': [results[m]['cv_rmse'] for m in results.keys()],
    'Train RMSE': [results[m]['train_rmse'] for m in results.keys()],
})

comparison_df = comparison_df.sort_values('Val RMSE')
print("\n", comparison_df.to_string(index=False))

# Save comparison
comparison_df.to_csv('/mnt/user-data/outputs/model_comparison_improved.csv', index=False)

# Identify best model
best_model_name = comparison_df.iloc[0]['Model']
best_model = results[best_model_name]['model']

print(f"\n🏆 Best Model: {best_model_name}")
print(f"   Validation RMSE: {results[best_model_name]['val_rmse']:.4f}")
print(f"   Validation R²: {results[best_model_name]['val_r2']:.4f}")
print(f"   Cross-Val RMSE: {results[best_model_name]['cv_rmse']:.4f}")

# ============================================================================
# 9. FEATURE IMPORTANCE
# ============================================================================
if hasattr(best_model, 'feature_importances_'):
    print("\n" + "=" * 80)
    print("[STEP 9] Feature Importance Analysis")
    print("=" * 80)
    
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': best_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("\nTop 10 Most Important Features:")
    print(feature_importance.head(10).to_string(index=False))
    
    # Save feature importance
    feature_importance.to_csv('/mnt/user-data/outputs/feature_importance_improved.csv', index=False)

# ============================================================================
# 10. GENERATE PREDICTIONS ON TEST SET
# ============================================================================
print("\n" + "=" * 80)
print("[STEP 10] Generating Predictions on Test Set")
print("=" * 80)

# Make predictions using the best model
test_predictions = best_model.predict(df_test_processed)

print(f"\nPredictions generated: {len(test_predictions)}")
print(f"Prediction range: [{test_predictions.min():.2f}, {test_predictions.max():.2f}]")
print(f"Prediction mean: {test_predictions.mean():.2f}")

# ============================================================================
# 11. CREATE SUBMISSION FILE
# ============================================================================
print("\n" + "=" * 80)
print("[STEP 11] Creating Submission File")
print("=" * 80)

# Load sample submission
sub_example = pd.read_csv("sample_submission.csv")

# Create submission dataframe
submission = pd.DataFrame({
    "row_id": sub_example["row_id"],  
    "Y": test_predictions  
})

# Save submission
submission.to_csv("/mnt/user-data/outputs/submission_improved.csv", index=False)
print("✓ Submission file saved: submission_improved.csv")

print("\nFirst few predictions:")
print(submission.head(10))

# ============================================================================
# 12. VISUALIZATIONS
# ============================================================================
print("\n" + "=" * 80)
print("[STEP 12] Creating Visualizations")
print("=" * 80)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. Model Comparison - RMSE
ax1 = axes[0, 0]
comparison_sorted = comparison_df.sort_values('Val RMSE', ascending=False)
bars = ax1.barh(comparison_sorted['Model'], comparison_sorted['Val RMSE'], color='skyblue')
ax1.axvline(x=comparison_sorted['Val RMSE'].min(), color='red', linestyle='--', alpha=0.5)
ax1.set_xlabel('Validation RMSE', fontsize=12, fontweight='bold')
ax1.set_title('Model Comparison - Validation RMSE', fontsize=14, fontweight='bold')
ax1.grid(axis='x', alpha=0.3)

# Add value labels
for bar in bars:
    width = bar.get_width()
    ax1.text(width, bar.get_y() + bar.get_height()/2, 
             f'{width:.4f}',
             ha='left', va='center', fontsize=10)

# 2. Model Comparison - R²
ax2 = axes[0, 1]
comparison_sorted_r2 = comparison_df.sort_values('Val R²', ascending=True)
bars = ax2.barh(comparison_sorted_r2['Model'], comparison_sorted_r2['Val R²'], color='lightcoral')
ax2.set_xlabel('Validation R²', fontsize=12, fontweight='bold')
ax2.set_title('Model Comparison - Validation R²', fontsize=14, fontweight='bold')
ax2.grid(axis='x', alpha=0.3)

# Add value labels
for bar in bars:
    width = bar.get_width()
    ax2.text(width, bar.get_y() + bar.get_height()/2, 
             f'{width:.4f}',
             ha='left', va='center', fontsize=10)

# 3. Actual vs Predicted (Best Model)
ax3 = axes[1, 0]
y_pred_best = results[best_model_name]['predictions']
ax3.scatter(y_val, y_pred_best, alpha=0.5, s=30, edgecolors='black', linewidths=0.5)
ax3.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--', lw=2, label='Perfect Prediction')
ax3.set_xlabel('Actual Values', fontsize=12, fontweight='bold')
ax3.set_ylabel('Predicted Values', fontsize=12, fontweight='bold')
ax3.set_title(f'Actual vs Predicted - {best_model_name}', fontsize=14, fontweight='bold')
ax3.legend()
ax3.grid(alpha=0.3)

# Add R² to plot
r2_text = f'R² = {results[best_model_name]["val_r2"]:.4f}'
ax3.text(0.05, 0.95, r2_text, transform=ax3.transAxes, 
         fontsize=12, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# 4. Feature Importance
ax4 = axes[1, 1]
if hasattr(best_model, 'feature_importances_'):
    top_features = feature_importance.head(10)
    bars = ax4.barh(top_features['Feature'], top_features['Importance'], color='lightgreen')
    ax4.set_xlabel('Importance', fontsize=12, fontweight='bold')
    ax4.set_title(f'Top 10 Features - {best_model_name}', fontsize=14, fontweight='bold')
    ax4.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for bar in bars:
        width = bar.get_width()
        ax4.text(width, bar.get_y() + bar.get_height()/2, 
                 f'{width:.3f}',
                 ha='left', va='center', fontsize=9)
else:
    ax4.text(0.5, 0.5, 'Feature importance\nnot available for this model', 
             ha='center', va='center', fontsize=12, fontweight='bold')
    ax4.axis('off')

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/model_evaluation_improved.png', dpi=300, bbox_inches='tight')
print("✓ Visualization saved: model_evaluation_improved.png")

# ============================================================================
# 13. SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

print(f"""
✓ Data preprocessed with feature engineering
✓ Trained and evaluated {len(models)} models
✓ Best model: {best_model_name}
✓ Validation RMSE: {results[best_model_name]['val_rmse']:.4f}
✓ Validation R²: {results[best_model_name]['val_r2']:.4f}
✓ Cross-validation RMSE: {results[best_model_name]['cv_rmse']:.4f}

KEY IMPROVEMENTS OVER ORIGINAL CODE:
1. ✓ Feature engineering (age, categories, interactions)
2. ✓ Multiple model comparison (5 models)
3. ✓ Cross-validation for robust evaluation
4. ✓ Better hyperparameters for ensemble models
5. ✓ Comprehensive visualizations
6. ✓ Feature importance analysis

FILES CREATED:
- submission_improved.csv (your predictions)
- model_comparison_improved.csv (all model metrics)
- feature_importance_improved.csv (feature rankings)
- model_evaluation_improved.png (visualizations)

NEXT STEPS TO IMPROVE FURTHER:
1. Try XGBoost/LightGBM for better performance
2. Hyperparameter tuning using GridSearchCV
3. Create more interaction features
4. Try ensemble methods (stacking/blending)
5. Analyze residuals for patterns
""")

print("\n" + "=" * 80)
print("PIPELINE COMPLETE!")
print("=" * 80)
