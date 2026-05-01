# Healthcare CVE Classification - CI/CD Pipeline

![CI](https://github.com/Andi-IM/SMSML_Andi-Irham/actions/workflows/ci.yaml/badge.svg)

Repositori ini adalah lingkungan otomatisasi (CI/CD environment) untuk memastikan model dapat diproduksi secara konsisten dan siap dideploy.

## 🎯 Peran Environment
Lingkungan ini difokuskan untuk **Otomatisasi Pipeline**. Setiap perubahan kode yang di-push ke branch `main` akan memicu serangkaian proses otomatis untuk melatih ulang model, mencatat hasilnya, dan membungkusnya ke dalam Docker image.

## ⚙️ Komponen Utama

### 1. MLflow Projects (`MLProject`)
Lingkungan ini menggunakan standar `MLProject` untuk memastikan training dapat dijalankan di mana saja dengan parameter yang sama.
- **Entry Point:** `main`
- **Parameter:** `n_estimators`, `max_depth`, `dataset`.

### 2. GitHub Actions Workflow
Workflow terletak di `.github/workflows/ci.yaml` yang melakukan:
- Training model otomatis menggunakan `uv`.
- Pengiriman hasil (metrics & plots) ke DagsHub.
- Pembuatan Docker image dari model terbaik.
- Push image ke Docker Hub: `irham22ai/smsml_andi-irham`.

## 🔐 Konfigurasi Secrets
Untuk menjalankan pipeline ini di GitHub, Anda harus mengatur **Secrets** berikut:
- `DOCKERHUB_TOKEN`: Token akses untuk push image.
- `DAGSHUB_TOKEN`: Token akses untuk log eksperimen ke DagsHub.
- `DOCKERHUB_USERNAME`: (Variabel) Username Docker Hub Anda.

## 🚀 Cara Menjalankan Secara Manual
Jika Anda ingin mensimulasikan proses CI di lokal:
```bash
uv run mlflow run . --experiment-name "Healthcare CVE Classification" --env-manager=local
```

## 📦 Hasil Akhir (Deployment)
Model yang berhasil dilatih akan tersedia di Docker Hub:
`docker pull irham22ai/smsml_andi-irham:latest`
Image ini sudah siap untuk digunakan sebagai REST API untuk prediksi.
