import pandas as pd
import numpy as np
from datetime import datetime
import sys

def analyze_signals(csv_path: str):
    """
    Generates TWO reports:
    1. Human-readable with formatting (to terminal)
    2. LLM-friendly compact version (to .txt file)
    """

    df = pd.read_csv(csv_path)
    df['forecast_date'] = pd.to_datetime(df['forecast_date'])
    df['observation_date'] = pd.to_datetime(df['observation_date'])

    evaluated = df[df['actual_vix_change'].notna()].copy()
    pending = df[df['actual_vix_change'].isna()].copy()

    # Calculate trading days
    start_date = df['forecast_date'].min()
    end_date = df['forecast_date'].max()
    trading_days = pd.bdate_range(start=start_date, end=end_date).shape[0]

    # ==================== PREPARE ALL DATA ====================

    # Basic stats
    basic_stats = {
        'file': csv_path,
        'generated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'period_start': df['forecast_date'].min().date(),
        'period_end': df['forecast_date'].max().date(),
        'total_predictions': len(df),
        'evaluated': len(evaluated),
        'pending': len(pending),
        'trading_days': trading_days,
        'actionable_count': df['actionable'].sum(),
        'actionable_pct': df['actionable'].sum()/trading_days*100
    }

    # Direction breakdown
    dir_all = df['direction_prediction'].value_counts()
    actionable = df[df['actionable'] == True]
    dir_actionable = actionable['direction_prediction'].value_counts()

    # Cohort breakdown
    cohort_counts = df['calendar_cohort'].value_counts().sort_index()

    # Confidence breakdown
    conf_counts = df['conf_bucket'].value_counts().sort_index()

    # Performance metrics (if evaluated data exists)
    performance_data = {}
    if len(evaluated) > 0:
        overall_acc = evaluated['direction_correct'].mean()

        up_subset = evaluated[evaluated['direction_prediction'] == 'UP']
        down_subset = evaluated[evaluated['direction_prediction'] == 'DOWN']

        performance_data = {
            'overall_accuracy': overall_acc,
            'overall_correct': evaluated['direction_correct'].sum(),
            'overall_total': len(evaluated),
            'up_accuracy': up_subset['direction_correct'].mean() if len(up_subset) > 0 else None,
            'up_correct': up_subset['direction_correct'].sum() if len(up_subset) > 0 else 0,
            'up_total': len(up_subset),
            'down_accuracy': down_subset['direction_correct'].mean() if len(down_subset) > 0 else None,
            'down_correct': down_subset['direction_correct'].sum() if len(down_subset) > 0 else 0,
            'down_total': len(down_subset),
            'mae_mean': evaluated['magnitude_error'].mean(),
            'mae_median': evaluated['magnitude_error'].median(),
            'mae_std': evaluated['magnitude_error'].std(),
            'mae_min': evaluated['magnitude_error'].min(),
            'mae_max': evaluated['magnitude_error'].max(),
            'mae_q25': evaluated['magnitude_error'].quantile(0.25),
            'mae_q50': evaluated['magnitude_error'].quantile(0.50),
            'mae_q75': evaluated['magnitude_error'].quantile(0.75),
            'mae_q90': evaluated['magnitude_error'].quantile(0.90),
            'mae_q95': evaluated['magnitude_error'].quantile(0.95)
        }

        # Extreme events
        spikes = evaluated[evaluated['actual_vix_change'] > 20]
        crashes = evaluated[evaluated['actual_vix_change'] < -20]

        extreme_data = {
            'spikes_count': len(spikes),
            'spikes_accuracy': spikes['direction_correct'].mean() if len(spikes) > 0 else None,
            'spikes_caught': spikes['direction_correct'].sum() if len(spikes) > 0 else 0,
            'spikes_mae': spikes['magnitude_error'].mean() if len(spikes) > 0 else None,
            'spikes_missed': (len(spikes) - spikes['direction_correct'].sum()) if len(spikes) > 0 else 0,
            'crashes_count': len(crashes),
            'crashes_accuracy': crashes['direction_correct'].mean() if len(crashes) > 0 else None,
            'crashes_caught': crashes['direction_correct'].sum() if len(crashes) > 0 else 0,
            'crashes_mae': crashes['magnitude_error'].mean() if len(crashes) > 0 else None
        }

        # Cohort performance
        cohort_perf = []
        for cohort in evaluated['calendar_cohort'].unique():
            subset = evaluated[evaluated['calendar_cohort'] == cohort]
            cohort_perf.append({
                'cohort': cohort,
                'count': len(subset),
                'accuracy': subset['direction_correct'].mean(),
                'mae': subset['magnitude_error'].mean(),
                'up_count': (subset['direction_prediction'] == 'UP').sum(),
                'down_count': (subset['direction_prediction'] == 'DOWN').sum()
            })
        cohort_perf = sorted(cohort_perf, key=lambda x: x['accuracy'], reverse=True)

        # Confidence calibration
        conf_perf = []
        for bucket in evaluated['conf_bucket'].unique():
            subset = evaluated[evaluated['conf_bucket'] == bucket]
            conf_perf.append({
                'bucket': bucket,
                'count': len(subset),
                'accuracy': subset['direction_correct'].mean(),
                'mean_conf': subset['ensemble_confidence'].mean(),
                'mae': subset['magnitude_error'].mean(),
                'calibration_gap': abs(subset['ensemble_confidence'].mean() - subset['direction_correct'].mean())
            })
        conf_perf = sorted(conf_perf, key=lambda x: x['mean_conf'])

        # Temporal patterns
        evaluated['month'] = evaluated['forecast_date'].dt.month
        evaluated['day_of_week'] = evaluated['forecast_date'].dt.dayofweek

        monthly_perf = []
        for month in sorted(evaluated['month'].unique()):
            subset = evaluated[evaluated['month'] == month]
            monthly_perf.append({
                'month': month,
                'count': len(subset),
                'accuracy': subset['direction_correct'].mean(),
                'mae': subset['magnitude_error'].mean()
            })

        dow_perf = []
        dow_map = {0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thu', 4: 'Fri'}
        for dow in sorted(evaluated['day_of_week'].unique()):
            subset = evaluated[evaluated['day_of_week'] == dow]
            dow_perf.append({
                'day': dow_map.get(dow, str(dow)),
                'count': len(subset),
                'accuracy': subset['direction_correct'].mean(),
                'mae': subset['magnitude_error'].mean()
            })

        # Confusion matrix
        actual_up = (evaluated['actual_vix_change'] > 0).astype(int)
        pred_up = (evaluated['direction_prediction'] == 'UP').astype(int)

        tp = ((pred_up == 1) & (actual_up == 1)).sum()
        tn = ((pred_up == 0) & (actual_up == 0)).sum()
        fp = ((pred_up == 1) & (actual_up == 0)).sum()
        fn = ((pred_up == 0) & (actual_up == 1)).sum()

        confusion_data = {
            'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
            'precision_up': tp/(tp+fp) if (tp+fp) > 0 else 0,
            'recall_up': tp/(tp+fn) if (tp+fn) > 0 else 0,
            'f1_up': 2*tp/(2*tp+fp+fn) if (2*tp+fp+fn) > 0 else 0
        }

        # Insights
        strengths = []
        weaknesses = []

        if overall_acc > 0.65:
            strengths.append(f"Strong overall accuracy: {overall_acc*100:.1f}%")
        if len(up_subset) > 0 and up_subset['direction_correct'].mean() > 0.85:
            strengths.append(f"Excellent UP prediction: {up_subset['direction_correct'].mean()*100:.1f}% ({len(up_subset)} signals)")
        if cohort_perf[0]['accuracy'] > 0.75:
            strengths.append(f"Strong {cohort_perf[0]['cohort']} performance: {cohort_perf[0]['accuracy']*100:.1f}%")
        well_calibrated = all(c['calibration_gap'] < 0.15 for c in conf_perf)
        if well_calibrated:
            strengths.append("Well-calibrated confidence scores")
        correct_mae = evaluated[evaluated['direction_correct'] == 1]['magnitude_error'].mean()
        if correct_mae < 15:
            strengths.append(f"Low magnitude error when correct: {correct_mae:.2f}")

        if len(down_subset) > 0 and down_subset['direction_correct'].mean() < 0.65:
            wrong = (down_subset['direction_correct'] == 0).sum()
            weaknesses.append(f"Weak DOWN predictions: {down_subset['direction_correct'].mean()*100:.1f}% ({wrong:.0f}/{len(down_subset)} wrong)")
        if len(spikes) > 0:
            missed = len(spikes) - spikes['direction_correct'].sum()
            if missed > 5:
                weaknesses.append(f"Missed {missed:.0f} VIX spikes - portfolio risk!")
        bias_ratio = len(down_subset) / len(up_subset) if len(up_subset) > 0 else float('inf')
        if bias_ratio > 3:
            weaknesses.append(f"Heavy DOWN bias: {bias_ratio:.1f}:1 ratio")
        if cohort_perf[-1]['accuracy'] < 0.65 and cohort_perf[-1]['count'] > 10:
            weaknesses.append(f"Poor {cohort_perf[-1]['cohort']} performance: {cohort_perf[-1]['accuracy']*100:.1f}%")
        if evaluated['magnitude_error'].mean() > 20:
            weaknesses.append(f"High mean magnitude error: {evaluated['magnitude_error'].mean():.2f}")
        false_alarms = ((evaluated['direction_prediction'] == 'UP') & (evaluated['direction_correct'] == 0)).sum()
        if false_alarms > 10:
            weaknesses.append(f"High false alarm rate: {false_alarms:.0f} wrong UP predictions")

    # ==================== HUMAN-READABLE OUTPUT (TERMINAL) ====================

    print("\n" + "="*100)
    print("VIX SIGNAL ANALYSIS REPORT (HUMAN-READABLE)")
    print("="*100)
    print(f"Generated: {basic_stats['generated']}")
    print(f"File: {basic_stats['file']}")
    print(f"Period: {basic_stats['period_start']} to {basic_stats['period_end']}")

    print("\n" + "─"*100)
    print("📊 SIGNAL OVERVIEW")
    print("─"*100)
    print(f"Total Predictions:           {basic_stats['total_predictions']:>6}")
    print(f"Evaluated (with actuals):    {basic_stats['evaluated']:>6}")
    print(f"Pending (no actuals yet):    {basic_stats['pending']:>6}")
    print(f"Trading days in period:      {basic_stats['trading_days']:>6}")
    print(f"Actionable (all signals):    {basic_stats['actionable_count']:>6} ({basic_stats['actionable_pct']:>5.1f}% of trading days)")

    print("\n" + "─"*100)
    print("🎯 SIGNAL BREAKDOWN")
    print("─"*100)

    print("\nBy Direction (All Signals):")
    for direction in ['UP', 'DOWN']:
        count = dir_all.get(direction, 0)
        pct = count/len(df)*100
        print(f"   {direction:>4}: {count:>5} ({pct:>5.1f}%)")

    print("\nBy Direction (Actionable Only):")
    for direction in ['UP', 'DOWN']:
        count = dir_actionable.get(direction, 0)
        pct = count/len(actionable)*100 if len(actionable) > 0 else 0
        print(f"   {direction:>4}: {count:>5} ({pct:>5.1f}%)")

    print("\nBy Calendar Cohort:")
    for cohort, count in cohort_counts.items():
        pct = count/len(df)*100
        print(f"   {cohort:.<20} {count:>5} ({pct:>5.1f}%)")

    print("\nBy Confidence Bucket:")
    for bucket, count in conf_counts.items():
        pct = count/len(df)*100
        print(f"   {bucket:.<15} {count:>5} ({pct:>5.1f}%)")

    if len(evaluated) > 0:
        print("\n" + "─"*100)
        print("✅ PERFORMANCE METRICS (Evaluated Signals Only)")
        print("─"*100)

        print(f"\nOverall Direction Accuracy:  {performance_data['overall_accuracy']*100:>6.2f}% ({performance_data['overall_correct']:.0f}/{performance_data['overall_total']})")

        print("\nAccuracy by Prediction:")
        if performance_data['up_accuracy'] is not None:
            print(f"   UP   predictions: {performance_data['up_accuracy']*100:>6.2f}% ({performance_data['up_correct']:.0f}/{performance_data['up_total']})")
        if performance_data['down_accuracy'] is not None:
            print(f"   DOWN predictions: {performance_data['down_accuracy']*100:>6.2f}% ({performance_data['down_correct']:.0f}/{performance_data['down_total']})")

        print(f"\nMagnitude Performance:")
        print(f"   Mean Absolute Error:     {performance_data['mae_mean']:>7.2f}")
        print(f"   Median Absolute Error:   {performance_data['mae_median']:>7.2f}")
        print(f"   Std Dev:                 {performance_data['mae_std']:>7.2f}")
        print(f"   Min Error:               {performance_data['mae_min']:>7.2f}")
        print(f"   Max Error:               {performance_data['mae_max']:>7.2f}")

        print(f"\nError Quartiles:")
        print(f"    25th percentile:        {performance_data['mae_q25']:>7.2f}")
        print(f"    50th percentile:        {performance_data['mae_q50']:>7.2f}")
        print(f"    75th percentile:        {performance_data['mae_q75']:>7.2f}")
        print(f"    90th percentile:        {performance_data['mae_q90']:>7.2f}")
        print(f"    95th percentile:        {performance_data['mae_q95']:>7.2f}")

        print("\n" + "─"*100)
        print("🚨 EXTREME EVENT DETECTION")
        print("─"*100)

        print(f"\nVIX Spikes (>20 point increases):")
        print(f"   Count:               {extreme_data['spikes_count']:>5}")
        if extreme_data['spikes_count'] > 0:
            print(f"   Correctly predicted: {extreme_data['spikes_caught']:.0f}/{extreme_data['spikes_count']} ({extreme_data['spikes_accuracy']*100:>5.1f}%)")
            print(f"   Mean error:          {extreme_data['spikes_mae']:>7.2f}")
            print(f"   Missed spikes:       {extreme_data['spikes_missed']:.0f} 🚨")

        print(f"\nVIX Crashes (<-20 point decreases):")
        print(f"   Count:               {extreme_data['crashes_count']:>5}")
        if extreme_data['crashes_count'] > 0:
            print(f"   Correctly predicted: {extreme_data['crashes_caught']:.0f}/{extreme_data['crashes_count']} ({extreme_data['crashes_accuracy']*100:>5.1f}%)")
            print(f"   Mean error:          {extreme_data['crashes_mae']:>7.2f}")

        print("\n" + "─"*100)
        print("📅 PERFORMANCE BY CALENDAR COHORT")
        print("─"*100)
        print(f"\n{'Cohort':<20} {'Count':>6} {'Accuracy':>9} {'MAE':>8} {'UP':>5} {'DOWN':>5}")
        print("─"*60)
        for row in cohort_perf:
            print(f"{row['cohort']:<20} {row['count']:>6.0f} {row['accuracy']*100:>8.1f}% {row['mae']:>8.2f} {row['up_count']:>5.0f} {row['down_count']:>5.0f}")

        print("\n" + "─"*100)
        print("🎚️  CONFIDENCE CALIBRATION")
        print("─"*100)
        print(f"\n{'Bucket':<15} {'Count':>6} {'Accuracy':>9} {'Avg Conf':>9} {'Cal Gap':>9} {'MAE':>8}")
        print("─"*70)
        for row in conf_perf:
            gap_status = "✓" if row['calibration_gap'] < 0.10 else "✗"
            print(f"{row['bucket']:<15} {row['count']:>6.0f} {row['accuracy']*100:>8.1f}% {row['mean_conf']*100:>8.1f}% {row['calibration_gap']*100:>8.1f}% {gap_status} {row['mae']:>7.2f}")

        print("\n" + "─"*100)
        print("📈 TEMPORAL PATTERNS")
        print("─"*100)

        print("\nBy Month:")
        print(f"\n{'Month':>6} {'Count':>6} {'Accuracy':>9} {'MAE':>8}")
        print("─"*35)
        for row in monthly_perf:
            print(f"{row['month']:>6} {row['count']:>6} {row['accuracy']*100:>8.1f}% {row['mae']:>8.2f}")

        print("\nBy Day of Week:")
        print(f"\n{'Day':>5} {'Count':>6} {'Accuracy':>9} {'MAE':>8}")
        print("─"*35)
        for row in dow_perf:
            print(f"{row['day']:>5} {row['count']:>6} {row['accuracy']*100:>8.1f}% {row['mae']:>8.2f}")

        print("\n" + "─"*100)
        print("🔢 CONFUSION MATRIX")
        print("─"*100)
        print(f"\n                  Actual UP    Actual DOWN")
        print(f"Predicted UP        {confusion_data['tp']:>5}          {confusion_data['fp']:>5}")
        print(f"Predicted DOWN      {confusion_data['fn']:>5}          {confusion_data['tn']:>5}")
        print(f"\nMetrics:")
        print(f"   Precision (UP):  {confusion_data['precision_up']*100:>6.2f}%")
        print(f"   Recall (UP):     {confusion_data['recall_up']*100:>6.2f}%")
        print(f"   F1 Score (UP):   {confusion_data['f1_up']*100:>6.2f}%")

        print("\n" + "─"*100)
        print("💡 KEY INSIGHTS")
        print("─"*100)

        print("\n✅ STRENGTHS:")
        for s in strengths:
            print(f"   • {s}")

        print("\n⚠️  WEAKNESSES:")
        for w in weaknesses:
            print(f"   • {w}")

    else:
        print("\n⚠️  No evaluated signals yet - all predictions are pending")

    if len(pending) > 0:
        print("\n" + "─"*100)
        print("⏳ PENDING SIGNALS (Not Yet Evaluated)")
        print("─"*100)
        print(f"\nTotal pending: {len(pending)}")
        print(f"\nBy Direction:")
        for direction in ['UP', 'DOWN']:
            count = (pending['direction_prediction'] == direction).sum()
            print(f"   {direction:>4}: {count:>5}")
        print(f"\nNext 5 to be evaluated:")
        next_5 = pending.nsmallest(5, 'observation_date')[['forecast_date', 'observation_date', 'direction_prediction', 'expected_vix', 'calendar_cohort']]
        print(next_5.to_string(index=False))

    print("\n" + "="*100)
    print("END OF REPORT")
    print("="*100 + "\n")

    # ==================== LLM-FRIENDLY OUTPUT (TEXT FILE) ====================

    llm_output = []
    llm_output.append(f"VIX SIGNAL ANALYSIS - LLM COMPACT FORMAT")
    llm_output.append(f"Generated: {basic_stats['generated']}")
    llm_output.append(f"File: {basic_stats['file']}")
    llm_output.append(f"Period: {basic_stats['period_start']} to {basic_stats['period_end']}")
    llm_output.append(f"")
    llm_output.append(f"OVERVIEW: total_predictions={basic_stats['total_predictions']}, evaluated={basic_stats['evaluated']}, pending={basic_stats['pending']}, trading_days={basic_stats['trading_days']}, actionable={basic_stats['actionable_count']} ({basic_stats['actionable_pct']:.1f}% of trading days)")
    llm_output.append(f"")
    llm_output.append(f"DIRECTION_ALL: UP={dir_all.get('UP', 0)} ({dir_all.get('UP', 0)/len(df)*100:.1f}%), DOWN={dir_all.get('DOWN', 0)} ({dir_all.get('DOWN', 0)/len(df)*100:.1f}%)")
    llm_output.append(f"DIRECTION_ACTIONABLE: UP={dir_actionable.get('UP', 0)} ({dir_actionable.get('UP', 0)/len(actionable)*100 if len(actionable) > 0 else 0:.1f}%), DOWN={dir_actionable.get('DOWN', 0)} ({dir_actionable.get('DOWN', 0)/len(actionable)*100 if len(actionable) > 0 else 0:.1f}%)")
    llm_output.append(f"")
    llm_output.append(f"COHORTS: " + ", ".join([f"{k}={v}" for k, v in cohort_counts.items()]))
    llm_output.append(f"CONFIDENCE_BUCKETS: " + ", ".join([f"{k}={v}" for k, v in conf_counts.items()]))
    llm_output.append(f"")

    if len(evaluated) > 0:
        llm_output.append(f"PERFORMANCE: overall_accuracy={performance_data['overall_accuracy']*100:.2f}% ({performance_data['overall_correct']:.0f}/{performance_data['overall_total']})")
        llm_output.append(f"ACCURACY_BY_DIRECTION: UP={performance_data['up_accuracy']*100 if performance_data['up_accuracy'] else 0:.2f}% ({performance_data['up_correct']:.0f}/{performance_data['up_total']}), DOWN={performance_data['down_accuracy']*100 if performance_data['down_accuracy'] else 0:.2f}% ({performance_data['down_correct']:.0f}/{performance_data['down_total']})")
        llm_output.append(f"MAGNITUDE: mean={performance_data['mae_mean']:.2f}, median={performance_data['mae_median']:.2f}, std={performance_data['mae_std']:.2f}, min={performance_data['mae_min']:.2f}, max={performance_data['mae_max']:.2f}")
        llm_output.append(f"QUARTILES: q25={performance_data['mae_q25']:.2f}, q50={performance_data['mae_q50']:.2f}, q75={performance_data['mae_q75']:.2f}, q90={performance_data['mae_q90']:.2f}, q95={performance_data['mae_q95']:.2f}")
        llm_output.append(f"")
        llm_output.append(f"SPIKES: count={extreme_data['spikes_count']}, caught={extreme_data['spikes_caught']:.0f}, accuracy={extreme_data['spikes_accuracy']*100 if extreme_data['spikes_accuracy'] else 0:.1f}%, mae={extreme_data['spikes_mae'] if extreme_data['spikes_mae'] else 0:.2f}, missed={extreme_data['spikes_missed']:.0f}")
        llm_output.append(f"CRASHES: count={extreme_data['crashes_count']}, caught={extreme_data['crashes_caught']:.0f}, accuracy={extreme_data['crashes_accuracy']*100 if extreme_data['crashes_accuracy'] else 0:.1f}%, mae={extreme_data['crashes_mae'] if extreme_data['crashes_mae'] else 0:.2f}")
        llm_output.append(f"")
        llm_output.append(f"COHORT_PERFORMANCE:")
        for row in cohort_perf:
            llm_output.append(f"  {row['cohort']}: count={row['count']}, accuracy={row['accuracy']*100:.1f}%, mae={row['mae']:.2f}, up={row['up_count']:.0f}, down={row['down_count']:.0f}")
        llm_output.append(f"")
        llm_output.append(f"CONFIDENCE_CALIBRATION:")
        for row in conf_perf:
            llm_output.append(f"  {row['bucket']}: count={row['count']}, accuracy={row['accuracy']*100:.1f}%, conf={row['mean_conf']*100:.1f}%, gap={row['calibration_gap']*100:.1f}%, mae={row['mae']:.2f}")
        llm_output.append(f"")
        llm_output.append(f"MONTHLY_PERFORMANCE: " + ", ".join([f"m{row['month']}={row['accuracy']*100:.1f}%" for row in monthly_perf]))
        llm_output.append(f"DOW_PERFORMANCE: " + ", ".join([f"{row['day']}={row['accuracy']*100:.1f}%" for row in dow_perf]))
        llm_output.append(f"")
        llm_output.append(f"CONFUSION_MATRIX: TP={confusion_data['tp']}, TN={confusion_data['tn']}, FP={confusion_data['fp']}, FN={confusion_data['fn']}, precision={confusion_data['precision_up']*100:.2f}%, recall={confusion_data['recall_up']*100:.2f}%, f1={confusion_data['f1_up']*100:.2f}%")
        llm_output.append(f"")
        llm_output.append(f"STRENGTHS: {len(strengths)}")
        for s in strengths:
            llm_output.append(f"  - {s}")
        llm_output.append(f"WEAKNESSES: {len(weaknesses)}")
        for w in weaknesses:
            llm_output.append(f"  - {w}")
    else:
        llm_output.append(f"NO_EVALUATION_DATA: All predictions are pending")

    if len(pending) > 0:
        llm_output.append(f"")
        llm_output.append(f"PENDING: total={len(pending)}, UP={(pending['direction_prediction'] == 'UP').sum()}, DOWN={(pending['direction_prediction'] == 'DOWN').sum()}")

    # Write LLM output to file
    llm_file = csv_path.replace('.csv', '_llm_report.txt')
    with open(llm_file, 'w') as f:
        f.write('\n'.join(llm_output))

    print(f"✅ LLM-friendly report saved to: {llm_file}\n")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyzer.py <csv_file>")
        sys.exit(1)

    analyze_signals(sys.argv[1])
