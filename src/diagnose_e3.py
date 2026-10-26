"""
diagnose_e3.py
==============

Investigates the AUC < 0.5 anomaly from E3 (CLAS -> WESAD, AUC = 0.223).
An AUC below 0.5 means the model's probability ranking is SYSTEMATICALLY
INVERTED, not just noisy -- this needs a specific diagnosis before trusting
any result built on top of it (e.g. the pooled pretraining encoder).

This reruns CLAS -> WESAD (same as E3) but additionally saves/prints:
    1. Confusion matrix -- is the model confidently wrong, or just uncertain?
    2. Per-subject AUC breakdown -- is the inversion uniform across all 15
       WESAD subjects, or concentrated in a few?
    3. Flip test -- does AUC(1 - pred_probs) recover a sensible number?
       If yes, the ranking information is real, just inverted somewhere
       (worth knowing whether that's a genuine semantic finding or a bug).

This does NOT change cross_dataset.py's leakage-safe training logic --
it reuses the exact same train_cross_dataset function and config, only
adding diagnostic output on top.
"""

import argparse

import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score

from dataset_loader import load_dataset, register_dataset
from cross_dataset import run_cross_dataset_experiment, register_known_datasets


def diagnose(train_name='clas', test_name='wesad', epochs=350, batch_size=512,
             early_stopping_patience=80, verbose=0, save_path='e3_diagnostic.npz'):

    register_known_datasets()

    test_data = load_dataset(test_name)
    subject_ids_test = test_data['subject_ids']
    y_test = test_data['labels']

    print(f"=== Rerunning {train_name} -> {test_name} with diagnostics ===\n")

    # Reuse the exact same leakage-safe training logic as cross_dataset.py --
    # we just need the model back so we can get predictions ourselves.
    metrics, model = run_cross_dataset_experiment(
        [train_name], test_name, epochs=epochs, batch_size=batch_size,
        early_stopping_patience=early_stopping_patience, verbose=verbose
    )

    pred_probs = model.predict(test_data['windows'], verbose=0)[:, 1]
    pred_classes = (pred_probs >= 0.5).astype(int)

    print("--- 1. Confusion matrix ---")
    cm = confusion_matrix(y_test, pred_classes)
    print(f"                 Predicted 0   Predicted 1")
    print(f"Actual 0 (non-stress)  {cm[0][0]:>6}       {cm[0][1]:>6}")
    print(f"Actual 1 (stress)      {cm[1][0]:>6}       {cm[1][1]:>6}")
    tn, fp, fn, tp = cm.ravel()
    print(f"\n-> Is the model CONFIDENTLY wrong (high FP+FN relative to TP+TN), "
          f"or just weak/uncertain (predictions near 50/50)?")
    print(f"   Correct: {tn + tp} / {len(y_test)} ({100*(tn+tp)/len(y_test):.1f}%)")
    print(f"   Wrong:   {fp + fn} / {len(y_test)} ({100*(fp+fn)/len(y_test):.1f}%)")

    print("\n--- 2. Per-subject AUC breakdown ---")
    unique_subjects = np.unique(subject_ids_test)
    per_subject_auc = {}
    for subj in unique_subjects:
        mask = subject_ids_test == subj
        y_subj = y_test[mask]
        if len(np.unique(y_subj)) < 2:
            print(f"  {subj}: only one class present, AUC undefined")
            continue
        auc_subj = roc_auc_score(y_subj, pred_probs[mask])
        per_subject_auc[str(subj)] = auc_subj
        print(f"  {subj}: AUC = {auc_subj:.3f}")

    aucs = list(per_subject_auc.values())
    if aucs:
        below_half = sum(1 for a in aucs if a < 0.5)
        print(f"\n-> {below_half} / {len(aucs)} subjects have AUC < 0.5")
        print(f"   If most/all subjects are below 0.5 -> UNIFORM inversion (suggests a bug)")
        print(f"   If only some are -> CONCENTRATED (suggests a genuine, if unusual, finding)")

    print("\n--- 3. Flip test ---")
    overall_auc = roc_auc_score(y_test, pred_probs)
    flipped_auc = roc_auc_score(y_test, 1 - pred_probs)
    print(f"  Original AUC:  {overall_auc:.4f}")
    print(f"  Flipped AUC:   {flipped_auc:.4f}  (computed as AUC(1 - pred_probs))")
    if flipped_auc > 0.65:
        print(f"\n  -> Flipping recovers a MEANINGFUL AUC ({flipped_auc:.3f}). The ranking")
        print(f"     information is real, just inverted. Check: label encoding direction")
        print(f"     (is CLAS's 'task'=1 possibly physiologically closer to WESAD's")
        print(f"     'baseline'=0 than to WESAD's 'stress'=1?), or a 1-y vs y swap")
        print(f"     somewhere in the pipeline.")
    else:
        print(f"\n  -> Flipping does NOT recover a strong AUC. This looks like genuinely")
        print(f"     poor/uninformative transfer, not a simple inversion bug.")

    np.savez(save_path,
             y_test=y_test, pred_probs=pred_probs, subject_ids=subject_ids_test,
             confusion_matrix=cm, per_subject_auc=per_subject_auc,
             overall_auc=overall_auc, flipped_auc=flipped_auc)
    print(f"\nSaved full diagnostic artifacts to {save_path}")

    return {
        'confusion_matrix': cm, 'per_subject_auc': per_subject_auc,
        'overall_auc': overall_auc, 'flipped_auc': flipped_auc
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Diagnose the E3 (CLAS->WESAD) AUC<0.5 anomaly")
    parser.add_argument("--train", default="clas")
    parser.add_argument("--test", default="wesad")
    parser.add_argument("--quick", action="store_true", help="Fast version for self-testing, not the real diagnostic")
    args = parser.parse_args()

    if args.quick:
        diagnose(args.train, args.test, epochs=5, batch_size=32,
                  early_stopping_patience=3, verbose=1, save_path='e3_diagnostic_quick.npz')
    else:
        diagnose(args.train, args.test, verbose=0)
