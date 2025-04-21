ai-marketing-assistant/        # root folder
├── README.md                 # Giới thiệu, hướng dẫn cài đặt & chạy
├── requirements.txt          # Thư viện cần cài đặt
│
├── config/                   # Cấu hình chung (paths, parameters)
│   └── settings.py           # Các biến môi trường, config model, app
│
├── data/                     # Chứa dữ liệu nguyên thủy và đã xử lý
│   ├── raw/                  # Dữ liệu crawl hoặc tải về
│   └── processed/            # Dữ liệu sau tiền xử lý (.csv, .pkl,...)
│
├── notebooks/                # Jupyter notebooks (EDA, experiments)
│   ├── 01_EDA.ipynb
│   └── 02_Model_Training.ipynb
│
├── src/                      # Mã nguồn chính (package)
│   ├── __init__.py           # Đánh dấu là package Python
│   │
│   ├── data_processing/      # Module xử lý dữ liệu
│   │   ├── __init__.py
│   │   ├── loader.py         # Hàm load CSV/DB
│   │   └── cleaner.py        # Hàm xử lý missing, encode, scale
│   │
│   ├── modeling/             # Module training & prediction
│   │   ├── __init__.py
│   │   ├── train.py          # Script huấn luyện & lưu model
│   │   └── predict.py        # API phục vụ dự đoán (load model)
│   │
│   ├── analysis/             # Module phân cụm & sentiment
│   │   ├── clustering.py     # KMeans, gán cluster
│   │   └── sentiment.py      # Sentiment analysis từ review
│   │
│   ├── utils/                # Công cụ hỗ trợ chung
│   │   ├── __init__.py
│   │   └── helpers.py        # Hàm chung (logging, download)
│   │
│   └── webapp/               # Ứng dụng web (Streamlit)
│       ├── __init__.py
│       └── app.py            # Streamlit entry-point
│
└── tests/                    # Unit tests
    ├── __init__.py
    ├── test_cleaner.py
    └── test_predict.py