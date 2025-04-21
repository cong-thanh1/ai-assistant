import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle

# Must be the first Streamlit command
st.set_page_config(page_title="AI Marketing Assistant", layout="centered")

# Get the absolute path to the model files
current_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(current_dir, '..', 'model')

def load_model_and_encoders():
    try:
        model_path = os.path.join(model_dir, 'customer_churn_model.pkl')
        encoders_path = os.path.join(model_dir, 'encoders.pkl')
        
        # Using plain pickle instead of joblib
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        with open(encoders_path, 'rb') as f:
            encoders = pickle.load(f)
        return model, encoders
    except FileNotFoundError:
        st.error("Model files not found. Please check if the model files exist in the correct location.")
        return None, None
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None

# Load model and encoders
model, encoders = load_model_and_encoders()

# 2. Preprocessing function (applies same encoding as training)
def preprocess_input(df: pd.DataFrame) -> pd.DataFrame:
    df_out = df.copy()
    for col, encoder in encoders.items():
        df_out[col] = encoder.transform(df_out[col])
    return df_out

# 3. Rule-based strategy suggestion
def suggest_strategy(prob: float) -> str:
    if prob >= 0.8:
        return "🔥 Gọi chăm sóc + khuyến mãi VIP"
    elif prob >= 0.5:
        return "⚠️ Giảm giá nhẹ hoặc khuyến mãi thêm dịch vụ"
    else:
        return "✅ Khách hàng ổn định, duy trì dịch vụ tốt"

# 5. Title and divider
st.title("🧠 AI Dự Đoán Khách Hàng Rời Dịch Vụ (Churn)")
st.markdown("---")

# 6. Tabs for single and batch prediction
tab1, tab2 = st.tabs(["📋 Dự đoán 1 khách hàng", "📂 Dự đoán từ file CSV"] )

# Tab 1: Single customer prediction
with tab1:
    st.header("Thông tin khách hàng")

    # Input fields
    gender = st.selectbox("Giới tính", ["Male", "Female"])
    contract = st.selectbox("Loại hợp đồng", ["Month-to-month", "One year", "Two year"])
    tenure = st.slider("Số tháng sử dụng (tenure)", 0, 72, 12)
    monthly = st.number_input("Chi phí hàng tháng (Monthly Charges)", 0.0, 200.0, 70.0)
    total = st.number_input("Tổng chi phí (Total Charges)", 0.0, 10000.0, 2000.0)

    # Create DataFrame
    input_df = pd.DataFrame([{
        "gender": gender,
        "Contract": contract,
        "tenure": tenure,
        "MonthlyCharges": monthly,
        "TotalCharges": total
    }])

    # Predict when button clicked
    if st.button("🧠 Dự đoán"):
        try:
            df_enc = preprocess_input(input_df)
            proba = model.predict_proba(df_enc)[0][1]
            label = "🚨 Khách có khả năng rời bỏ!" if proba > 0.5 else "✅ Khách hàng ổn định"
            st.subheader(f"Kết quả: {label}")
            st.metric(label="Xác suất churn", value=f"{proba:.2%}")
            st.info(suggest_strategy(proba))
        except Exception as e:
            st.error(f"Lỗi xử lý dữ liệu: {e}")

# Tab 2: Batch prediction via CSV upload
with tab2:
    st.header("Upload file CSV")
    uploaded_file = st.file_uploader("Tải lên file CSV chứa danh sách khách hàng", type=["csv"])
    if uploaded_file is not None:
        try:
            df_csv = pd.read_csv(uploaded_file)
            df_enc = preprocess_input(df_csv)
            # Add prediction columns
            df_csv['Churn_Prob'] = model.predict_proba(df_enc)[:, 1]
            df_csv['Dự đoán'] = df_csv['Churn_Prob'].apply(lambda p: "Churn" if p > 0.5 else "Không Churn")
            df_csv['Chiến lược'] = df_csv['Churn_Prob'].apply(suggest_strategy)

            st.success("✅ Dự đoán thành công!")
            st.dataframe(df_csv)

            # Download results as CSV
            csv = df_csv.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Tải kết quả dự đoán",
                data=csv,
                file_name='ket_qua_du_doan.csv',
                mime='text/csv'
            )
        except Exception as e:
            st.error(f"Lỗi khi xử lý file CSV: {e}")
