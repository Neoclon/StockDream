# ensemble_prediction.py
import pandas as pd

def ensemble_predictions(ae_csv, if_csv, weight_ae=0.5, weight_if=0.5, threshold=0.5):
    df_ae = pd.read_csv(ae_csv, parse_dates=["start_date"])
    df_if = pd.read_csv(if_csv, parse_dates=["start_date"])
    df_ens = pd.merge(df_ae, df_if, on=["symbol", "start_date", "digit_type"], how="inner")
    df_ens["ensemble_prob"] = weight_ae * df_ens["ae_prob"] + weight_if * df_ens["if_prob"]
    df_ens["anomaly"] = df_ens["ensemble_prob"] > threshold
    return df_ens

def main():
    ae_csv = "./공모전준비/모델 학습 과정/결과/autoencoder_results.csv"  # 결과 from lstm_autoencoder_test.py
    if_csv = "./공모전준비/모델 학습 과정/결과/if_results.csv"             # 결과 from isolation_forest_test.py
    df_ensemble = ensemble_predictions(ae_csv, if_csv, weight_ae=0.5, weight_if=0.5, threshold=0.3)
    df_ensemble.to_csv("ensemble_results.csv", index=False)
    print("Ensemble results saved to ensemble_results.csv")
    print(df_ensemble.head())
    
    # 최종 이상치 비율 출력
    anomaly_ratio = (df_ensemble["anomaly"]).mean() * 100
    print(f"Overall Ensemble anomaly ratio: {anomaly_ratio:.2f}%")

if __name__ == "__main__":
    main()
