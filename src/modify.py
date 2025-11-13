def predict(self, X: pd.DataFrame, cohort: str) -> Dict:
    if cohort not in self.models:
        raise ValueError(
            f"Cohort {cohort} not trained. Available: {list(self.models.keys())}"
        )

    X_features = X[self.feature_names]

    # ============================================================
    # STEP 1: Get point estimate (this IS the median/q50)
    # ============================================================
    point = self.models[cohort]["point"].predict(X_features)[0]

    # ============================================================
    # STEP 2: Get uncertainty estimate
    # ============================================================
    uncertainty = self.models[cohort]["uncertainty"].predict(X_features)[0]
    
    # Ensure minimum uncertainty (prevent zero-width intervals)
    uncertainty = max(uncertainty, 1.0)  # At least 1% uncertainty

    # ============================================================
    # STEP 3: Generate CONSISTENT quantiles using normal distribution
    # ============================================================
    # Standard normal quantile values
    z_scores = {
        'q10': -1.28,
        'q25': -0.67,
        'q50': 0.00,   # This equals point estimate
        'q75': 0.67,
        'q90': 1.28,
    }
    
    quantiles = {
        q_name: point + z * uncertainty 
        for q_name, z in z_scores.items()
    }

    # ============================================================
    # STEP 4: Get direction probability (NEW)
    # ============================================================
    direction_proba = self.models[cohort]["direction"].predict_proba(X_features)[0]
    prob_up = float(direction_proba[1])  # Probability of positive change

    # ============================================================
    # STEP 5: Get confidence score
    # ============================================================
    confidence = self.models[cohort]["confidence"].predict(X_features)[0]
    confidence = np.clip(confidence, 0, 1)

    return {
        "point_estimate": float(point),
        "quantiles": {k: float(v) for k, v in quantiles.items()},
        "direction_probability": prob_up,  # NEW
        "confidence_score": float(confidence),
        "cohort": cohort,
        # REMOVED: regime_probabilities
    }