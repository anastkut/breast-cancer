# Enhanced Breast Cancer Analysis Project using scikit-learn dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.cluster import KMeans
from scipy import stats
from sklearn.ensemble import RandomForestClassifier

# Keep the same random state for reproducibility as in original code
RANDOM_STATE = 44
CV_FOLDS = 5  # Number of cross-validation folds

# Function to load and prepare the breast cancer dataset from scikit-learn
def load_data():
    """Load the breast cancer dataset from scikit-learn"""
    data = load_breast_cancer()
    
    # Convert to DataFrame (similar to original)
    feature_names = data.feature_names
    bc_df = pd.DataFrame(data.data, columns=feature_names)
    
    # Add diagnosis column (0 = benign, 1 = malignant)
    bc_df['diagnosis'] = data.target
    
    # Add ID column for consistency with original
    bc_df['id'] = range(len(bc_df))
    
    print(f"Dataset shape: {bc_df.shape}")
    print(f"Number of benign samples: {sum(bc_df['diagnosis'] == 0)}")
    print(f"Number of malignant samples: {sum(bc_df['diagnosis'] == 1)}")
    
    return bc_df

# Enhanced feature selection function
def enhanced_feature_selection(X, y):
    """Enhanced feature selection using L1 regularization and correlation analysis"""
    # Scale features for L1 regularization
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Method 1: L1 regularization (as in original)
    print("PERFORMING L1 REGULARIZATION FEATURE SELECTION")
    logreg_l1 = LogisticRegression(random_state=RANDOM_STATE, 
                                  solver='saga', 
                                  penalty='l1', 
                                  max_iter=10000)
    logreg_l1.fit(X_scaled, y)
    
    # Get features with non-zero coefficients
    feature_names = X.columns
    l1_coefs = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': logreg_l1.coef_[0]
    })
    l1_selected = l1_coefs[l1_coefs['Coefficient'] != 0].sort_values(by='Coefficient', ascending=False)
    print("Features selected by L1 regularization:")
    print(l1_selected)
    
    # Calculate correlation matrix for selected features
    selected_features = l1_selected['Feature'].tolist()
    X_selected = X[selected_features]
    corr_matrix = X_selected.corr()
    
    # Plot correlation heatmap for selected features
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix of Selected Features')
    plt.tight_layout()
    plt.savefig('selected_features_correlation.png', dpi=300)
    
    return selected_features, X_scaled

# Cross-validation with confidence intervals
def cross_validate_models(X, y, selected_features):
    """Perform cross-validation for multiple models with confidence intervals"""
    print("\n====== CROSS-VALIDATION WITH CONFIDENCE INTERVALS ======")
    
    # Prepare data with only selected features
    X_selected = X[selected_features]
    
    # Scale features
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_selected)
    
    # Define models to evaluate
    models = {
        'KMeans': KMeans(n_clusters=2, random_state=RANDOM_STATE, n_init=10),
        'Logistic Regression': LogisticRegression(random_state=RANDOM_STATE),
        'SVM': svm.SVC(probability=True, random_state=RANDOM_STATE),
        'Random Forest': RandomForestClassifier(random_state=RANDOM_STATE)
    }
    
    # Perform repeated cross-validation (10 times with different seeds)
    n_repeats = 10
    cv_results = {}
    
    for name, model in models.items():
        if name == 'KMeans':
            # KMeans doesn't support cross_val_score directly
            continue
        
        # Store results for each repeat
        repeat_scores = []
        
        for repeat in range(n_repeats):
            # Create new CV splitter with different random seed for each repeat
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=repeat)
            scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='accuracy')
            repeat_scores.append(scores)
        
        # Calculate mean and std across all repeats
        all_scores = np.concatenate(repeat_scores)
        cv_results[name] = {
            'mean': all_scores.mean(),
            'std': all_scores.std(),
            'all_scores': all_scores,
            'ci_lower': np.percentile(all_scores, 2.5),
            'ci_upper': np.percentile(all_scores, 97.5)
        }
        print(f"{name} CV Accuracy: {all_scores.mean():.4f} ± {all_scores.std():.4f}")
    
    # For KMeans, we need to handle differently since it's unsupervised
    kmeans_scores = []
    
    # Perform repeated KMeans clustering with train/test splits
    for i in range(30):  # 30 iterations
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.15, random_state=i)
        
        # Train KMeans
        kmeans = KMeans(n_clusters=2, random_state=i, n_init=10)
        kmeans.fit(X_train)
        
        # Make predictions
        y_pred = kmeans.predict(X_test)
        
        # Map cluster labels to diagnosis labels (0 and 1)
        # This is necessary as KMeans might assign clusters arbitrarily
        cluster_diagnosis = {}
        for cluster in [0, 1]:
            # Find most common diagnosis in each cluster
            cluster_mask = (y_pred == cluster)
            if sum(cluster_mask) > 0:
                diagnosis_counts = np.bincount(y_test[cluster_mask])
                most_common = np.argmax(diagnosis_counts)
                cluster_diagnosis[cluster] = most_common
        
        # Remap predictions
        y_pred_remapped = np.array([cluster_diagnosis.get(cluster, 0) for cluster in y_pred])
        
        # Calculate accuracy
        accuracy = np.mean(y_pred_remapped == y_test)
        kmeans_scores.append(accuracy)
    
    # Calculate confidence interval for KMeans
    kmeans_mean = np.mean(kmeans_scores)
    kmeans_std = np.std(kmeans_scores)
    kmeans_ci_lower = np.percentile(kmeans_scores, 2.5)
    kmeans_ci_upper = np.percentile(kmeans_scores, 97.5)
    
    cv_results['KMeans'] = {
        'mean': kmeans_mean,
        'std': kmeans_std,
        'all_scores': np.array(kmeans_scores),
        'ci_lower': kmeans_ci_lower,
        'ci_upper': kmeans_ci_upper
    }
    
    print(f"KMeans CV Accuracy: {kmeans_mean:.4f} ± {kmeans_std:.4f}")
    
    # Visualize cross-validation results
    plt.figure(figsize=(12, 6))
    
    # Prepare data for plotting
    model_names = list(cv_results.keys())
    means = [cv_results[name]['mean'] for name in model_names]
    errors = [cv_results[name]['std'] for name in model_names]
    
    # Create bar plot with error bars
    x_pos = np.arange(len(model_names))
    plt.bar(x_pos, means, yerr=errors, align='center', alpha=0.7, capsize=10)
    
    plt.xlabel('Model')
    plt.ylabel('Accuracy')
    plt.title('Cross-Validation Results with 95% Confidence Intervals')
    plt.xticks(x_pos, model_names, rotation=45)
    plt.ylim([0.8, 1.0])  # Adjust as needed
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('cross_validation_results.png', dpi=300)
    
    # Perform statistical comparison between models
    print("\n====== STATISTICAL COMPARISON OF MODELS ======")
    
    # Perform paired t-test between best and second best models
    sorted_models = sorted(model_names, key=lambda name: cv_results[name]['mean'], reverse=True)
    best_model = sorted_models[0]
    second_best = sorted_models[1]
    
    t_stat, p_value = stats.ttest_rel(
        cv_results[best_model]['all_scores'],
        cv_results[second_best]['all_scores']
    )
    
    print(f"Best model: {best_model} (Accuracy: {cv_results[best_model]['mean']:.4f})")
    print(f"Second best model: {second_best} (Accuracy: {cv_results[second_best]['mean']:.4f})")
    print(f"Paired t-test: t={t_stat:.4f}, p={p_value:.4f}")
    
    if p_value < 0.05:
        print(f"The difference between {best_model} and {second_best} is statistically significant (p<0.05)")
    else:
        print(f"The difference between {best_model} and {second_best} is NOT statistically significant (p>=0.05)")
    
    return cv_results

# Replace the current analyze_fp_fn_rates function with this cross-validation based version
def analyze_fp_fn_rates(X, y, selected_features):
    """Analyze false positive and false negative rates using cross-validation"""
    print("\n====== FALSE POSITIVE/NEGATIVE ANALYSIS WITH CROSS-VALIDATION ======")
    
    # Prepare data with only selected features
    X_selected = X[selected_features]
    
    # Scale features
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_selected)
    
    # Use logistic regression for analysis
    model = LogisticRegression(random_state=RANDOM_STATE)
    
    # Use repeated stratified k-fold cross validation
    cv = RepeatedStratifiedKFold(n_splits=CV_FOLDS, n_repeats=10, random_state=RANDOM_STATE)
    
    # Track metrics across all CV folds
    fp_rates = []
    fn_rates = []
    
    # Perform cross-validation and collect FP/FN rates
    fold_count = 0
    for train_idx, test_idx in cv.split(X_scaled, y):
        # Split data using CV indices
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Train model
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_test)
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        # Calculate rates
        fp_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
        fn_rate = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        fp_rates.append(fp_rate)
        fn_rates.append(fn_rate)
        fold_count += 1
    
    # Calculate statistics
    fp_mean = np.mean(fp_rates)
    fp_std = np.std(fp_rates)
    fp_ci_lower = np.percentile(fp_rates, 2.5)
    fp_ci_upper = np.percentile(fp_rates, 97.5)
    
    fn_mean = np.mean(fn_rates)
    fn_std = np.std(fn_rates)
    fn_ci_lower = np.percentile(fn_rates, 2.5)
    fn_ci_upper = np.percentile(fn_rates, 97.5)
    
    print(f"False Positive Rate: {fp_mean:.4f} ± {fp_std:.4f}")
    print(f"95% CI: [{fp_ci_lower:.4f}, {fp_ci_upper:.4f}]")
    print(f"False Negative Rate: {fn_mean:.4f} ± {fn_std:.4f}")
    print(f"95% CI: [{fn_ci_lower:.4f}, {fn_ci_upper:.4f}]")
    
    # Plot FP and FN rates across CV folds
    plt.figure(figsize=(10, 6))
    plt.scatter(range(fold_count), fp_rates, color='red', label='False Positive Rate')
    plt.scatter(range(fold_count), fn_rates, color='blue', label='False Negative Rate')
    
    # Add means and confidence intervals
    plt.axhline(y=fp_mean, color='red', linestyle='--', 
               label=f'Mean FP: {fp_mean:.4f}')
    plt.axhline(y=fn_mean, color='blue', linestyle='--', 
               label=f'Mean FN: {fn_mean:.4f}')
    
    # Add confidence interval bands
    plt.fill_between(range(fold_count), 
                    [fp_ci_lower] * fold_count, 
                    [fp_ci_upper] * fold_count, 
                    color='red', alpha=0.2)
    plt.fill_between(range(fold_count), 
                    [fn_ci_lower] * fold_count, 
                    [fn_ci_upper] * fold_count, 
                    color='blue', alpha=0.2)
    
    plt.xlabel('CV Fold')
    plt.ylabel('Rate')
    plt.title('False Positive and False Negative Rates Across Cross-Validation Folds')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('fp_fn_rates_cv.png', dpi=300)
    
    # Threshold analysis using CV
    # This section needs to examine thresholds across CV folds
    
    # Set up storage for threshold analysis
    num_thresholds = 16  # Number of thresholds to evaluate
    thresholds = np.linspace(0.1, 0.9, num=num_thresholds)
    
    # Initialize arrays to store results for each threshold
    threshold_fp_rates = np.zeros((fold_count, num_thresholds))
    threshold_fn_rates = np.zeros((fold_count, num_thresholds))
    threshold_accuracies = np.zeros((fold_count, num_thresholds))
    
    # Perform cross-validation with different thresholds
    fold_idx = 0
    for train_idx, test_idx in cv.split(X_scaled, y):
        # Split data using CV indices
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Train model
        model.fit(X_train, y_train)
        
        # Get predicted probabilities
        y_prob = model.predict_proba(X_test)[:, 1]
        
        # Evaluate each threshold
        for i, threshold in enumerate(thresholds):
            y_pred = (y_prob >= threshold).astype(int)
            
            # Calculate confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            tn, fp, fn, tp = cm.ravel()
            
            # Calculate rates
            fp_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
            fn_rate = fn / (fn + tp) if (fn + tp) > 0 else 0
            accuracy = (tp + tn) / (tp + tn + fp + fn)
            
            # Store rates
            threshold_fp_rates[fold_idx, i] = fp_rate
            threshold_fn_rates[fold_idx, i] = fn_rate
            threshold_accuracies[fold_idx, i] = accuracy
        
        fold_idx += 1
    
    # Calculate mean rates for each threshold
    mean_fp_rates = np.mean(threshold_fp_rates, axis=0)
    mean_fn_rates = np.mean(threshold_fn_rates, axis=0)
    mean_accuracies = np.mean(threshold_accuracies, axis=0)
    
    # Create DataFrame for results
    threshold_results_df = pd.DataFrame({
        'threshold': thresholds,
        'fp_rate': mean_fp_rates,
        'fn_rate': mean_fn_rates,
        'accuracy': mean_accuracies
    })
    
    # Plot threshold impact
    plt.figure(figsize=(10, 6))
    plt.plot(threshold_results_df['threshold'], threshold_results_df['fp_rate'], 'r-', label='False Positive Rate')
    plt.plot(threshold_results_df['threshold'], threshold_results_df['fn_rate'], 'b-', label='False Negative Rate')
    plt.plot(threshold_results_df['threshold'], threshold_results_df['accuracy'], 'g-', label='Accuracy')
    
    plt.xlabel('Decision Threshold')
    plt.ylabel('Rate')
    plt.title('Impact of Decision Threshold on Performance Metrics (Cross-Validated)')
    plt.grid(True)
    plt.legend()
    plt.savefig('threshold_impact_cv.png', dpi=300)
    
    # Find optimal thresholds for different criteria
    
    # 1. Minimize FN (important for medical diagnostics)
    min_fn_idx = threshold_results_df['fn_rate'].idxmin()
    min_fn_threshold = threshold_results_df.loc[min_fn_idx, 'threshold']
    
    # 2. Balanced (FP ≈ FN)
    diffs = abs(threshold_results_df['fp_rate'] - threshold_results_df['fn_rate'])
    balanced_idx = diffs.idxmin()
    balanced_threshold = threshold_results_df.loc[balanced_idx, 'threshold']
    
    print("\nOptimal thresholds (cross-validated):")
    print(f"For minimizing False Negatives: {min_fn_threshold:.2f}")
    print(f"  FP rate: {threshold_results_df.loc[min_fn_idx, 'fp_rate']:.4f}")
    print(f"  FN rate: {threshold_results_df.loc[min_fn_idx, 'fn_rate']:.4f}")
    print(f"  Accuracy: {threshold_results_df.loc[min_fn_idx, 'accuracy']:.4f}")
    
    print(f"\nFor balanced FP/FN rates: {balanced_threshold:.2f}")
    print(f"  FP rate: {threshold_results_df.loc[balanced_idx, 'fp_rate']:.4f}")
    print(f"  FN rate: {threshold_results_df.loc[balanced_idx, 'fn_rate']:.4f}")
    print(f"  Accuracy: {threshold_results_df.loc[balanced_idx, 'accuracy']:.4f}")
    
    return fp_mean, fn_mean, min_fn_threshold, balanced_threshold

# ROC analysis with confidence bands
def roc_analysis_with_confidence(X, y, selected_features):
    """Perform ROC analysis with confidence bands"""
    print("\n====== ROC ANALYSIS WITH CONFIDENCE BANDS ======")
    
    # Prepare data with only selected features
    X_selected = X[selected_features]
    
    # Scale features
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_selected)
    
    # Models to compare
    models = {
        'Logistic Regression': LogisticRegression(random_state=RANDOM_STATE),
        'SVM': svm.SVC(probability=True, random_state=RANDOM_STATE),
        'Random Forest': RandomForestClassifier(random_state=RANDOM_STATE)
    }
    
    # Perform bootstrap to get confidence bands for ROC curves
    n_bootstraps = 100
    roc_results = {name: {'tprs': [], 'aucs': []} for name in models}
    
    # Set up common FPR points for interpolation
    mean_fpr = np.linspace(0, 1, 100)
    
    for i in range(n_bootstraps):
        # Create bootstrap sample (with replacement)
        indices = np.random.choice(len(y), len(y), replace=True)
        X_bootstrap = X_scaled[indices]
        y_bootstrap = y.iloc[indices] if isinstance(y, pd.Series) else y[indices]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_bootstrap, y_bootstrap, test_size=0.15, random_state=i)
        
        # Train each model and compute ROC
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_prob = model.predict_proba(X_test)[:, 1]
            
            # Compute ROC and interpolate
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)
            
            # Interpolate TPR at fixed FPR points
            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0  # Force starting point at (0,0)
            
            roc_results[name]['tprs'].append(interp_tpr)
            roc_results[name]['aucs'].append(roc_auc)
    
    # Plot ROC curves with confidence bands
    plt.figure(figsize=(10, 8))
    
    colors = ['blue', 'red', 'green']
    
    for i, (name, results) in enumerate(roc_results.items()):
        # Calculate mean TPR and confidence intervals
        mean_tpr = np.mean(results['tprs'], axis=0)
        mean_auc = np.mean(results['aucs'])
        std_auc = np.std(results['aucs'])
        
        # 95% confidence intervals for TPR
        tpr_upper = np.percentile(results['tprs'], 97.5, axis=0)
        tpr_lower = np.percentile(results['tprs'], 2.5, axis=0)
        
        # Plot mean ROC curve
        plt.plot(mean_fpr, mean_tpr, color=colors[i], 
                label=f'{name} (AUC = {mean_auc:.3f} ± {std_auc:.3f})')
        
        # Plot confidence band
        plt.fill_between(mean_fpr, tpr_lower, tpr_upper, 
                        color=colors[i], alpha=0.2,
                        label=f'95% CI')
    
    # Plot diagonal
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves with 95% Confidence Bands')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.savefig('roc_curves_with_confidence.png', dpi=300)
    
    # Display AUC statistics
    print("AUC Statistics:")
    for name, results in roc_results.items():
        mean_auc = np.mean(results['aucs'])
        std_auc = np.std(results['aucs'])
        ci_lower = np.percentile(results['aucs'], 2.5)
        ci_upper = np.percentile(results['aucs'], 97.5)
        
        print(f"{name}: AUC = {mean_auc:.4f} ± {std_auc:.4f}")
        print(f"95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
    
    return roc_results

# Enhanced F1 score analysis for breast cancer classification
def analyze_f1_scores(X, y, selected_features):
    """Analyze F1 scores across models and decision thresholds"""
    print("\n====== F1 SCORE ANALYSIS ======")
    
    # Prepare data with only selected features
    X_selected = X[selected_features]
    
    # Scale features
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_selected)
    
    # Models to evaluate
    models = {
        'Logistic Regression': LogisticRegression(random_state=RANDOM_STATE),
        'SVM': svm.SVC(probability=True, random_state=RANDOM_STATE),
        'Random Forest': RandomForestClassifier(random_state=RANDOM_STATE)
    }
    
    # Use repeated stratified k-fold cross validation
    cv = RepeatedStratifiedKFold(n_splits=CV_FOLDS, n_repeats=10, random_state=RANDOM_STATE)
    
    # Store results for each model
    model_f1_results = {name: [] for name in models}
    model_precision_results = {name: [] for name in models}
    model_recall_results = {name: [] for name in models}
    
    # For each model, collect F1 scores across CV folds
    for name, model in models.items():
        print(f"\nEvaluating {name}...")
        fold_count = 0
        
        for train_idx, test_idx in cv.split(X_scaled, y):
            # Split data using CV indices
            X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # Train model
            model.fit(X_train, y_train)
            
            # Predict
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            precision = metrics.precision_score(y_test, y_pred)
            recall = metrics.recall_score(y_test, y_pred)
            f1 = metrics.f1_score(y_test, y_pred)
            
            # Store results
            model_precision_results[name].append(precision)
            model_recall_results[name].append(recall)
            model_f1_results[name].append(f1)
            fold_count += 1
        
        # Calculate statistics
        mean_precision = np.mean(model_precision_results[name])
        std_precision = np.std(model_precision_results[name])
        ci_precision = (
            np.percentile(model_precision_results[name], 2.5),
            np.percentile(model_precision_results[name], 97.5)
        )
        
        mean_recall = np.mean(model_recall_results[name])
        std_recall = np.std(model_recall_results[name])
        ci_recall = (
            np.percentile(model_recall_results[name], 2.5),
            np.percentile(model_recall_results[name], 97.5)
        )
        
        mean_f1 = np.mean(model_f1_results[name])
        std_f1 = np.std(model_f1_results[name])
        ci_f1 = (
            np.percentile(model_f1_results[name], 2.5),
            np.percentile(model_f1_results[name], 97.5)
        )
        
        print(f"Precision: {mean_precision:.4f} ± {std_precision:.4f} (95% CI: [{ci_precision[0]:.4f}, {ci_precision[1]:.4f}])")
        print(f"Recall: {mean_recall:.4f} ± {std_recall:.4f} (95% CI: [{ci_recall[0]:.4f}, {ci_recall[1]:.4f}])")
        print(f"F1 Score: {mean_f1:.4f} ± {std_f1:.4f} (95% CI: [{ci_f1[0]:.4f}, {ci_f1[1]:.4f}])")
    
    # Compare F1 scores across models
    plt.figure(figsize=(10, 6))
    
    # Create box plots for F1 scores
    model_names = list(models.keys())
    f1_data = [model_f1_results[name] for name in model_names]
    
    plt.boxplot(f1_data, labels=model_names)
    plt.title('F1 Scores Across Models')
    plt.ylabel('F1 Score')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('f1_scores_comparison.png', dpi=300)
    
    # Analyze F1 score vs decision threshold
    print("\n==== F1 Score vs Decision Threshold Analysis ====")
    
    # Choose logistic regression for threshold analysis
    model = LogisticRegression(random_state=RANDOM_STATE)
    
    # Set up thresholds to evaluate
    thresholds = np.linspace(0.1, 0.9, num=20)
    
    # Storage for threshold analysis
    threshold_f1_means = []
    threshold_f1_stds = []
    threshold_precision_means = []
    threshold_recall_means = []
    
    # For each threshold, compute mean F1 across CV folds
    for threshold in thresholds:
        f1_scores = []
        precision_scores = []
        recall_scores = []
        
        for train_idx, test_idx in cv.split(X_scaled, y):
            # Split data using CV indices
            X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # Train model
            model.fit(X_train, y_train)
            
            # Get predicted probabilities and apply threshold
            y_prob = model.predict_proba(X_test)[:, 1]
            y_pred = (y_prob >= threshold).astype(int)
            
            # Calculate metrics
            precision = metrics.precision_score(y_test, y_pred)
            recall = metrics.recall_score(y_test, y_pred)
            f1 = metrics.f1_score(y_test, y_pred)
            
            precision_scores.append(precision)
            recall_scores.append(recall)
            f1_scores.append(f1)
        
        # Calculate statistics
        threshold_precision_means.append(np.mean(precision_scores))
        threshold_recall_means.append(np.mean(recall_scores))
        threshold_f1_means.append(np.mean(f1_scores))
        threshold_f1_stds.append(np.std(f1_scores))
    
    # Find optimal threshold for F1 score
    optimal_threshold_idx = np.argmax(threshold_f1_means)
    optimal_threshold = thresholds[optimal_threshold_idx]
    
    print(f"Optimal threshold for F1 score: {optimal_threshold:.2f}")
    print(f"F1 Score at optimal threshold: {threshold_f1_means[optimal_threshold_idx]:.4f}")
    print(f"Precision at optimal threshold: {threshold_precision_means[optimal_threshold_idx]:.4f}")
    print(f"Recall at optimal threshold: {threshold_recall_means[optimal_threshold_idx]:.4f}")
    
    # Plot F1, precision, and recall vs threshold
    plt.figure(figsize=(12, 6))
    
    plt.plot(thresholds, threshold_f1_means, 'b-', label='F1 Score')
    plt.fill_between(thresholds, 
                    np.array(threshold_f1_means) - np.array(threshold_f1_stds),
                    np.array(threshold_f1_means) + np.array(threshold_f1_stds),
                    alpha=0.2, color='blue')
    
    plt.plot(thresholds, threshold_precision_means, 'g-', label='Precision')
    plt.plot(thresholds, threshold_recall_means, 'r-', label='Recall')
    
    # Mark optimal threshold
    plt.axvline(x=optimal_threshold, color='k', linestyle='--', 
               label=f'Optimal threshold: {optimal_threshold:.2f}')
    
    plt.xlabel('Decision Threshold')
    plt.ylabel('Score')
    plt.title('F1 Score, Precision, and Recall vs Decision Threshold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('f1_threshold_analysis.png', dpi=300)
    
    # Add precision-recall curve
    plt.figure(figsize=(10, 6))
    
    # For logistic regression, get the PR curve
    model = LogisticRegression(random_state=RANDOM_STATE)
    
    # Perform cross-validation and collect PR curves
    mean_precision = np.linspace(0, 1, 100)
    all_recalls = []
    
    for train_idx, test_idx in list(cv.split(X_scaled, y))[:10]:  # Limit to 10 folds for clarity
        # Split data using CV indices
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Train model
        model.fit(X_train, y_train)
        
        # Get predicted probabilities
        y_prob = model.predict_proba(X_test)[:, 1]
        
        # Calculate precision-recall curve
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        
        # Store interpolated recall at fixed precision points
        recalls = np.interp(mean_precision, precision[::-1], recall[::-1])
        all_recalls.append(recalls)
    
    # Calculate average recall and confidence intervals
    mean_recall = np.mean(all_recalls, axis=0)
    std_recall = np.std(all_recalls, axis=0)
    
    # Plot PR curve with confidence band
    plt.plot(mean_precision, mean_recall, 'b-', label='Mean PR curve')
    plt.fill_between(mean_precision, 
                    mean_recall - std_recall,
                    mean_recall + std_recall,
                    alpha=0.2, color='blue',
                    label='±1 std. dev.')
    
    # Calculate average precision score
    average_precision = metrics.average_precision_score(y_test, y_prob)
    
    plt.xlabel('Precision')
    plt.ylabel('Recall')
    plt.title('Precision-Recall Curve (Cross-Validated)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('precision_recall_curve.png', dpi=300)
    
    # Compare ROC and PR curves
    plt.figure(figsize=(12, 6))
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
    
    # ROC Curve (reuse data from roc_analysis_with_confidence)
    # This is simplified for visualization comparison
    model = LogisticRegression(random_state=RANDOM_STATE)
    
    # ROC curve
    mean_tpr = []
    mean_fpr = np.linspace(0, 1, 100)
    
    for train_idx, test_idx in list(cv.split(X_scaled, y))[:10]:
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        mean_tpr.append(np.interp(mean_fpr, fpr, tpr))
    
    mean_tpr = np.mean(mean_tpr, axis=0)
    
    ax1.plot(mean_fpr, mean_tpr, 'b-')
    ax1.plot([0, 1], [0, 1], 'k--')
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('ROC Curve')
    ax1.grid(True, alpha=0.3)
    
    # PR Curve
    ax2.plot(mean_precision, mean_recall, 'r-')
    ax2.set_xlabel('Precision')
    ax2.set_ylabel('Recall')
    ax2.set_title('Precision-Recall Curve')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('roc_vs_pr_curve.png', dpi=300)
    
    return {
        'optimal_f1_threshold': optimal_threshold,
        'f1_at_optimal': threshold_f1_means[optimal_threshold_idx],
        'precision_at_optimal': threshold_precision_means[optimal_threshold_idx],
        'recall_at_optimal': threshold_recall_means[optimal_threshold_idx]
    }

# Main enhanced analysis script
# Modified main analysis function to include F1 score analysis
def run_enhanced_analysis():
    """Run the complete enhanced breast cancer analysis"""
    # Load data
    print("Loading breast cancer dataset...")
    bc_df = load_data()
    
    # Prepare data for modeling
    X = bc_df.drop(columns=['id', 'diagnosis'])
    y = bc_df['diagnosis']
    
    # Enhanced feature selection
    print("\nPerforming enhanced feature selection...")
    selected_features, X_scaled = enhanced_feature_selection(X, y)
    
    # Cross-validation with confidence intervals
    cv_results = cross_validate_models(X, y, selected_features)
    
    # Enhanced false positive/negative analysis
    fp_mean, fn_mean, min_fn_threshold, balanced_threshold = analyze_fp_fn_rates(X, y, selected_features)
    
    # F1 score analysis (add this line)
    f1_results = analyze_f1_scores(X, y, selected_features)
    
    # ROC analysis with confidence bands
    roc_results = roc_analysis_with_confidence(X, y, selected_features)
    
    print("\nEnhanced breast cancer analysis complete!")
    print("Summary of key findings:")
    
    # Rank models by cross-validation accuracy
    model_ranks = sorted(cv_results.keys(), 
                        key=lambda name: cv_results[name]['mean'], 
                        reverse=True)
    
    print("\nModel Performance Ranking:")
    for i, name in enumerate(model_ranks):
        results = cv_results[name]
        print(f"{i+1}. {name}: {results['mean']:.4f} (95% CI: [{results['ci_lower']:.4f}, {results['ci_upper']:.4f}])")
    
    print(f"\nFalse Positive Rate: {fp_mean:.4f}")
    print(f"False Negative Rate: {fn_mean:.4f}")
    print(f"Recommended threshold for medical context (minimizing false negatives): {min_fn_threshold:.2f}")
    
    # Add F1 summary
    print(f"\nF1 Score Analysis:")
    print(f"Optimal threshold for F1 score: {f1_results['optimal_f1_threshold']:.2f}")
    print(f"F1 score at optimal threshold: {f1_results['f1_at_optimal']:.4f}")
    print(f"Precision at optimal threshold: {f1_results['precision_at_optimal']:.4f}")
    print(f"Recall at optimal threshold: {f1_results['recall_at_optimal']:.4f}")
    
    return {
        'cross_validation_results': cv_results,
        'top_model': model_ranks[0],
        'selected_features': selected_features,
        'fp_rate': fp_mean,
        'fn_rate': fn_mean,
        'optimal_threshold_fn': min_fn_threshold,
        'optimal_threshold_f1': f1_results['optimal_f1_threshold'],  # Add this line
        'f1_results': f1_results  # Add this line
    }

# Run the enhanced analysis
if __name__ == "__main__":
    results = run_enhanced_analysis()