from pathlib import Path
import json
import os

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Cổng thông tin tra cứu hạn mức và phê duyệt khoản vay",
    layout="centered",
)

if "page" not in st.session_state:
    st.session_state.page = "main"

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "credit_model.pkl"
SCALER_PATH = BASE_DIR / "scaler.pkl"
COLUMNS_PATH = BASE_DIR / "model_columns.pkl"
METRICS_PATH = BASE_DIR / "model_metrics.json"

CATEGORICAL_COLUMNS = [
    "person_home_ownership",
    "loan_intent",
    "cb_person_default_on_file",
]

HOME_OWNERSHIP_OPTIONS = {
    "Thuê nhà": "RENT",
    "Đang trả góp mua nhà": "MORTGAGE",
    "Sở hữu riêng": "OWN",
    "Khác": "OTHER",
}

LOAN_INTENT_OPTIONS = {
    "Tiêu dùng cá nhân": "PERSONAL",
    "Học tập / Giáo dục": "EDUCATION",
    "Y tế / Chăm sóc sức khỏe": "MEDICAL",
    "Đầu tư / Kinh doanh": "VENTURE",
    "Cải tạo nhà cửa": "HOMEIMPROVEMENT",
    "Hợp thức hóa / Gộp nợ": "DEBTCONSOLIDATION",
}

loan_type_descriptions = {
    "Vay tín chấp": "Hình thức vay không cần tài sản đảm bảo, dựa hoàn toàn vào uy tín cá nhân và thu nhập của người vay.",
    "Vay thế chấp": "Hình thức vay truyền thống yêu cầu người vay phải có tài sản có giá trị (nhà, đất, xe cộ) để làm tài sản đảm bảo.",
    "Vay thấu chi": "Hình thức cho phép người vay chi tiêu vượt quá số dư thực tế trong tài khoản thanh toán đến một hạn mức nhất định.",
    "Vay trả góp": "Hình thức trả nợ định kỳ (hàng tháng) trong đó số tiền gốc và lãi được chia đều để thanh toán theo thỏa thuận.",
}

ADVISORY_PAGES = {
    "reduce_loan": {
        "title": "Tư vấn giảm khoản vay",
        "eyebrow": "Điều chỉnh kế hoạch vay",
        "body": "Giảm số tiền đề xuất vay có thể giúp tỷ lệ khoản vay trên thu nhập phù hợp hơn với khả năng chi trả hàng tháng.",
        "steps": [
            "Xác định số tiền thực sự cần cho mục đích hiện tại.",
            "Ưu tiên khoản vay vừa với ngân sách sau khi trừ các chi phí cố định.",
            "Tra cứu lại với số tiền mới trước khi nộp hồ sơ chính thức.",
        ],
    },
    "collateral_loan": {
        "title": "Tư vấn khoản vay có tài sản đảm bảo",
        "eyebrow": "Giải pháp có tài sản đảm bảo",
        "body": "Khoản vay thế chấp có thể phù hợp khi Quý khách sở hữu nhà, đất hoặc phương tiện đủ điều kiện và cần hạn mức cao hơn.",
        "steps": [
            "Chuẩn bị thông tin cơ bản về tài sản dự kiến dùng để đảm bảo.",
            "Trao đổi với chuyên viên về hạn mức và thời hạn phù hợp.",
            "Chỉ cung cấp hồ sơ qua các kênh chính thức của ngân hàng.",
        ],
    },
    "credit_improvement": {
        "title": "Kế hoạch cải thiện hồ sơ",
        "eyebrow": "Sức khỏe tài chính",
        "body": "Một lịch sử thanh toán ổn định và thời gian làm việc lâu hơn có thể giúp hồ sơ được đánh giá tích cực hơn trong tương lai.",
        "steps": [
            "Theo dõi và thanh toán đúng hạn các nghĩa vụ hiện có.",
            "Hạn chế mở thêm nhiều khoản tín dụng trong thời gian ngắn.",
            "Duy trì thu nhập và công việc ổn định trước khi đăng ký lại.",
        ],
    },
    "contact_hotline": {
        "title": "Gặp chuyên viên tư vấn",
        "eyebrow": "Hỗ trợ trực tiếp",
        "body": "Chuyên viên ngân hàng có thể xem xét thêm hoàn cảnh tài chính và gợi ý sản phẩm phù hợp hơn với nhu cầu thực tế.",
        "steps": [
            "Hotline tư vấn: 1900 1234",
            "Thời gian hỗ trợ: 08:00 - 20:00, tất cả các ngày trong tuần.",
            "Không cung cấp mật khẩu, mã OTP hoặc thông tin bảo mật qua điện thoại.",
        ],
    },
}

PAGE_SEQUENCE = (
    "main",
    "support",
    "reduce_loan",
    "collateral_loan",
    "credit_improvement",
    "contact_hotline",
)

AGE_BANDS = {
    "18 - 24 tuổi": 21,
    "25 - 34 tuổi": 29,
    "35 - 44 tuổi": 39,
    "45 - 54 tuổi": 49,
    "55 - 64 tuổi": 59,
    "65 - 100 tuổi": 70,
}

EMPLOYMENT_BANDS = {
    "Chưa có / dưới 1 năm": 0,
    "1 - 3 năm": 2,
    "4 - 6 năm": 5,
    "7 - 10 năm": 8,
    "11 - 20 năm": 15,
    "Trên 20 năm": 25,
}

NUMBER_WORDS = (
    "không",
    "một",
    "hai",
    "ba",
    "bốn",
    "năm",
    "sáu",
    "bảy",
    "tám",
    "chín",
)

def _read_three_digits(number: int, full: bool = False) -> str:
    hundreds, remainder = divmod(number, 100)
    tens, ones = divmod(remainder, 10)
    words = []

    if hundreds or full:
        words.extend((NUMBER_WORDS[hundreds], "trăm"))
        if remainder == 0:
            return " ".join(words)
        if remainder < 10:
            return " ".join(words + ["lẻ", NUMBER_WORDS[ones]])

    if tens:
        words.append("mười" if tens == 1 else f"{NUMBER_WORDS[tens]} mươi")
        if ones:
            words.append("mốt" if ones == 1 and tens > 1 else NUMBER_WORDS[ones])
    elif ones:
        words.append(NUMBER_WORDS[ones])
    return " ".join(words)

def number_to_vietnamese_words(amount: float) -> str:
    """Convert a non-negative USD amount to Vietnamese words for display."""
    integer_amount = int(round(amount))
    if integer_amount < 0:
        raise ValueError("Số tiền không được âm.")
    if integer_amount == 0:
        return "Không đô la Mỹ"

    groups = []
    while integer_amount:
        groups.append(integer_amount % 1000)
        integer_amount //= 1000

    group_names = ("", "nghìn", "triệu", "tỷ", "nghìn tỷ", "triệu tỷ")
    words = []
    for index in range(len(groups) - 1, -1, -1):
        group = groups[index]
        if not group:
            continue
        words.append(_read_three_digits(group, full=index < len(groups) - 1 and group < 100))
        if index < len(group_names) and group_names[index]:
            words.append(group_names[index])
    return " ".join(words).capitalize() + " đô la Mỹ"

@st.cache_resource(show_spinner=False)
def load_artifacts():
    """Load the model artifacts created by the training notebook."""
    missing = [
        path.name
        for path in (MODEL_PATH, SCALER_PATH, COLUMNS_PATH, METRICS_PATH)
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Chưa tìm thấy tệp mô hình: " + ", ".join(missing)
        )

    return (
        joblib.load(MODEL_PATH),
        joblib.load(SCALER_PATH),
        joblib.load(COLUMNS_PATH),
        json.loads(METRICS_PATH.read_text(encoding="utf-8")),
    )

def prepare_features(values: dict, model_columns) -> pd.DataFrame:
    """Build one manually encoded row using the persisted training schema."""
    columns = list(model_columns)

    features = pd.DataFrame(0.0, index=[0], columns=columns)
    numerical_values = {
        "person_age": values["person_age"],
        "person_income": values["person_income"],
        "person_emp_length": values["person_emp_length"],
        "loan_amnt": values["loan_amnt"],
        "loan_percent_income": values["loan_percent_income"],
    }
    for column, value in numerical_values.items():
        if column in features.columns:
            features.at[0, column] = value

    for feature_name in CATEGORICAL_COLUMNS:
        selected_value = values[feature_name]
        matching_columns = [
            column
            for column in columns
            if column.startswith(f"{feature_name}_")
            and column.lower().endswith(f"_{selected_value.lower()}")
        ]
        if matching_columns:
            features.at[0, matching_columns[0]] = 1.0

    return features

def predict_risk(values: dict, model, scaler, model_columns) -> tuple[int, float, dict]:
    features = prepare_features(values, model_columns)
    scaled_features = scaler.transform(features)
    prediction = int(model.predict(scaled_features)[0])
    probabilities = model.predict_proba(scaled_features)[0]
    default_index = list(model.classes_).index(1)
    contributions = scaled_features[0] * model.coef_[0]
    feature_contributions = {
        column: float(contribution)
        for column, contribution in zip(model_columns, contributions)
        if abs(contribution) >= 0.05
    }
    return prediction, float(probabilities[default_index]), feature_contributions

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=Space+Grotesk:wght@500;700&display=swap');
    .stApp { background: linear-gradient(145deg, #eef6ff 0%, #f7fbff 58%, #e5f3f7 100%); color: #102a43; }
    .block-container { max-width: 900px; padding-top: 2.2rem; padding-bottom: 3rem; }
    .hero { position: relative; overflow: hidden; background: linear-gradient(120deg, #0a3161, #155e9e); border-radius: 8px; color: #f8fbff; padding: 2.2rem 2.3rem 2.5rem; margin-bottom: 1.4rem; box-shadow: 0 14px 32px rgba(10, 49, 97, .16); }
    .hero::after { content: ''; position: absolute; left: -20%; bottom: 0; width: 40%; height: 3px; background: #60a5fa; animation: bank-scan 5s ease-in-out infinite; }
    @keyframes bank-scan { 0%, 100% { transform: translateX(0); opacity: .35; } 50% { transform: translateX(350%); opacity: 1; } }
    .hero h1 { font-family: 'Space Grotesk', sans-serif; font-size: 2.15rem; line-height: 1.1; margin: 0; }
    .hero p { color: #d5e8fa; font-family: 'DM Sans', sans-serif; font-size: 1rem; margin: .8rem 0 0; }
    .trust-line { color: #2563a6; font-family: 'DM Sans', sans-serif; font-size: .9rem; margin: 0 0 1.3rem; }
    .section-label { color: #155e9e; font-family: 'DM Sans', sans-serif; font-size: .76rem; font-weight: 700; letter-spacing: .09em; margin: 1rem 0 .4rem; text-transform: uppercase; }
    .home-panel { background: rgba(255, 255, 255, .78); border: 1px solid #bfdbfe; border-left: 4px solid #2563eb; border-radius: 8px; padding: 1.5rem 1.7rem; margin: 1.2rem 0 1.4rem; }
    .home-panel h2 { color: #0a3161; font-family: 'Space Grotesk', sans-serif; margin: .35rem 0 .6rem; }
    .home-panel p { color: #486581; margin: 0; }
    .eyebrow { color: #2563eb; font-size: .75rem; font-weight: 700; letter-spacing: .1em; text-transform: uppercase; }
    div[data-testid="stHorizontalBlock"] button { border-color: #bfdbfe; color: #0a3161; }
    div[data-testid="stForm"] { background: rgba(255, 255, 255, .86); border: 1px solid #bfdbfe; border-radius: 8px; padding: 1.25rem 1.35rem; box-shadow: 0 8px 24px rgba(10, 49, 97, .06); }
    div[data-testid="stFormSubmitButton"] button { background: #0a3161; border: 0; color: white; font-family: 'DM Sans', sans-serif; font-size: 1rem; font-weight: 700; min-height: 3.1rem; }
    div[data-testid="stFormSubmitButton"] button:hover { background: #155e9e; color: white; }
    [data-testid="stAlert"] { border-radius: 8px; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <h1>Cổng thông tin Tra cứu Khoản vay</h1>
        <p>Kiểm tra khả năng phê duyệt khoản vay của Quý khách trong vài bước đơn giản.</p>
    </div>
    <p class="trust-line">Thông tin được bảo mật và chỉ dùng cho mục đích tra cứu sơ bộ.</p>
    """,
    unsafe_allow_html=True,
)

def navigate_to(page: str) -> None:
    st.session_state.page = page

FEATURE_LABELS = {
    "person_age": "Độ tuổi",
    "person_income": "Thu nhập",
    "person_emp_length": "Thâm niên công tác",
    "loan_amnt": "Số tiền vay",
    "loan_percent_income": "Tỷ lệ vay / thu nhập",
    "person_home_ownership_MORTGAGE": "Nhà ở: đang trả góp",
    "person_home_ownership_OTHER": "Nhà ở: khác",
    "person_home_ownership_OWN": "Nhà ở: sở hữu riêng",
    "person_home_ownership_RENT": "Nhà ở: thuê",
    "loan_intent_DEBTCONSOLIDATION": "Mục đích: gộp nợ",
    "loan_intent_EDUCATION": "Mục đích: giáo dục",
    "loan_intent_HOMEIMPROVEMENT": "Mục đích: cải tạo nhà",
    "loan_intent_MEDICAL": "Mục đích: y tế",
    "loan_intent_PERSONAL": "Mục đích: cá nhân",
    "loan_intent_VENTURE": "Mục đích: kinh doanh",
    "cb_person_default_on_file_N": "Không có lịch sử nợ xấu",
    "cb_person_default_on_file_Y": "Có lịch sử nợ xấu",
}

def render_assessment_result(
    prediction: int,
    risk_probability: float,
    feature_contributions: dict,
    metrics: dict,
) -> None:
    probability_percentage = risk_probability * 100
    if risk_probability < 0.35:
        risk_label = "Thấp"
        risk_message = "Hồ sơ đang nằm trong vùng rủi ro thấp theo dữ liệu mô hình."
    elif risk_probability < 0.55:
        risk_label = "Trung bình"
        risk_message = "Hồ sơ nằm gần vùng quyết định; nên rà soát khả năng trả nợ trước khi đăng ký."
    else:
        risk_label = "Cao"
        risk_message = "Hồ sơ có nhiều dấu hiệu rủi ro; nên điều chỉnh khoản vay hoặc trao đổi với chuyên viên."

    result_left, result_right = st.columns([1, 1.4])
    with result_left:
        st.metric("Xác suất rủi ro ước tính", f"{probability_percentage:.1f}%")
        st.progress(min(max(risk_probability, 0.0), 1.0), text=f"Mức rủi ro: {risk_label}")
    with result_right:
        if prediction == 0 and risk_probability < 0.5:
            st.success("Kết quả sơ bộ: hồ sơ có khả năng được chấp thuận cao hơn.")
        else:
            st.warning("Kết quả sơ bộ: hồ sơ cần được cân nhắc thêm.")
        st.caption(risk_message)

    st.caption(
        f"Mô hình {metrics.get('model', 'Logistic Regression')} | "
        f"ROC-AUC kiểm thử: {metrics.get('roc_auc', 0):.3f}"
    )

    positive_drivers = sorted(
        ((FEATURE_LABELS.get(name, name), value) for name, value in feature_contributions.items() if value > 0),
        key=lambda item: item[1],
        reverse=True,
    )[:3]
    negative_drivers = sorted(
        ((FEATURE_LABELS.get(name, name), value) for name, value in feature_contributions.items() if value < 0),
        key=lambda item: item[1],
    )[:3]
    if positive_drivers or negative_drivers:
        st.markdown("#### Vì sao mô hình đưa ra kết quả này?")
        driver_left, driver_right = st.columns(2)
        with driver_left:
            st.markdown("**Làm tăng rủi ro**")
            for label, _ in positive_drivers:
                st.write(f"• {label}")
        with driver_right:
            st.markdown("**Giảm rủi ro tương đối**")
            for label, _ in negative_drivers:
                st.write(f"• {label}")

    if prediction == 0 and risk_probability < 0.5:
        st.info(
            "Đây là kết quả sàng lọc tự động, không phải quyết định tín dụng cuối cùng. "
            "Ngân hàng vẫn cần thẩm định hồ sơ và khả năng trả nợ thực tế."
        )
        return

    st.error(
        "Hãy cân nhắc giảm số tiền vay, tăng mức dự phòng trả nợ hoặc trao đổi với chuyên viên "
        "trước khi nộp hồ sơ chính thức."
    )
    st.markdown('<div class="section-label">Dịch vụ phù hợp hơn</div>', unsafe_allow_html=True)
    service_left, service_right = st.columns(2)
    with service_left:
        if st.button("Tư vấn giảm khoản vay", key="service_lower_loan", use_container_width=True):
            navigate_to("reduce_loan")
            st.rerun()
        st.write("Cân nhắc khoản vay nhỏ hơn để giảm tỷ lệ nghĩa vụ trả nợ trên thu nhập.")
        if st.button("Kế hoạch cải thiện hồ sơ", key="service_credit_plan", use_container_width=True):
            navigate_to("credit_improvement")
            st.rerun()
        st.write("Theo dõi lịch sử tín dụng và duy trì công việc ổn định trước khi đăng ký lại.")
    with service_right:
        if st.button("Tư vấn khoản vay có tài sản đảm bảo", key="service_secured_loan", use_container_width=True):
            navigate_to("collateral_loan")
            st.rerun()
        st.write("Tìm hiểu phương án thế chấp nếu Quý khách có tài sản phù hợp và cần hạn mức cao hơn.")
        if st.button("Gặp chuyên viên tư vấn / Thông tin Hotline", key="service_specialist", use_container_width=True):
            navigate_to("contact_hotline")
            st.rerun()
        st.write("Liên hệ kênh chính thức của ngân hàng để được đánh giá hồ sơ chi tiết và bảo mật.")

nav_left, nav_center, nav_right = st.columns([1.2, 1.4, 1.2])
with nav_left:
    if st.button("Trang chủ tra cứu", use_container_width=True, key="nav_home"):
        navigate_to("home")
        st.rerun()
with nav_center:
    if st.button("Kiểm tra khả năng vay", use_container_width=True, key="nav_assessment"):
        navigate_to("main")
        st.rerun()
with nav_right:
    if st.button("Hỗ trợ khách hàng", use_container_width=True, key="nav_support"):
        navigate_to("support")
        st.rerun()

if st.session_state.page == "home":
    st.markdown(
        """
        <div class="home-panel">
            <div class="eyebrow">Dịch vụ ngân hàng số</div>
            <h2>Giải pháp tài chính phù hợp với kế hoạch của Quý khách</h2>
            <p>Tra cứu sơ bộ khả năng vay, tìm hiểu sản phẩm và nhận tư vấn minh bạch trước khi nộp hồ sơ chính thức.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    home_col_left, home_col_right = st.columns(2)
    with home_col_left:
        st.markdown("#### Tra cứu khoản vay")
        st.write("Kiểm tra nhanh hồ sơ với các thông tin cơ bản về thu nhập, công việc và khoản vay dự kiến.")
        if st.button("Bắt đầu tra cứu", type="primary", use_container_width=True, key="home_start_assessment"):
            navigate_to("main")
            st.rerun()
    with home_col_right:
        st.markdown("#### Hỗ trợ tài chính")
        st.write("Nhận định hướng về lựa chọn khoản vay và các bước chuẩn bị hồ sơ phù hợp.")
        if st.button("Xem hỗ trợ", use_container_width=True, key="home_support"):
            navigate_to("support")
            st.rerun()
    st.stop()

if st.session_state.page == "support":
    st.markdown('<div class="section-label">Hỗ trợ khách hàng</div>', unsafe_allow_html=True)
    st.markdown("### Đồng hành cùng kế hoạch tài chính của Quý khách")
    st.write("Chọn nội dung cần quan tâm để được định hướng trước khi đăng ký sản phẩm.")
    support_left, support_right = st.columns(2)
    with support_left:
        st.markdown("#### Tư vấn khoản vay phù hợp")
        st.write("Trao đổi với chuyên viên về hạn mức, thời hạn và phương án trả nợ phù hợp với ngân sách.")
        st.markdown("#### Hướng dẫn chuẩn bị hồ sơ")
        st.write("Tìm hiểu các giấy tờ thường cần thiết cho quy trình đăng ký chính thức.")
    with support_right:
        st.markdown("#### Kênh hỗ trợ chính thức")
        st.write("Liên hệ chi nhánh, hotline hoặc ứng dụng ngân hàng để được xác thực và tư vấn trực tiếp.")
        st.markdown("#### Tra cứu lại khả năng vay")
        if st.button("Mở công cụ tra cứu", type="primary", use_container_width=True, key="support_assessment"):
            navigate_to("main")
            st.rerun()
    st.stop()

if st.session_state.page in ADVISORY_PAGES:
    advisory = ADVISORY_PAGES[st.session_state.page]
    if st.button("⬅ Quay lại trang tra cứu", key="advisory_back_top", use_container_width=True):
        navigate_to("main")
        st.rerun()
    st.markdown(
        f'<div class="section-label">{advisory["eyebrow"]}</div>',
        unsafe_allow_html=True,
    )
    st.markdown(f"### {advisory['title']}")
    st.write(advisory["body"])
    st.markdown("#### Các bước nên cân nhắc")
    for step in advisory["steps"]:
        st.write(f"- {step}")
    if st.session_state.page == "contact_hotline":
        st.info("Hotline 1900 1234")
    advisory_back, advisory_assessment = st.columns(2)
    with advisory_back:
        if st.button("Quay lại kết quả tra cứu", key="advisory_back_result", use_container_width=True):
            navigate_to("main")
            st.rerun()
    with advisory_assessment:
        if st.button("Mở lại công cụ tra cứu", type="primary", use_container_width=True, key="advisory_open_assessment"):
            navigate_to("main")
            st.rerun()
    st.stop()

try:
    model, scaler, model_columns, model_metrics = load_artifacts()
except Exception as error:
    st.error(f"Không thể tải hệ thống tra cứu: {error}")
    st.info(
        "Vui lòng chạy notebook huấn luyện và đặt ba tệp "
        "`credit_model.pkl`, `scaler.pkl`, `model_columns.pkl` trong thư mục dự án."
    )
    st.stop()

with st.form("loan_assessment_form"):
    st.markdown('<div class="section-label">Thông tin cá nhân cơ bản</div>', unsafe_allow_html=True)
    personal_left, personal_right = st.columns(2)
    with personal_left:
        age_band = st.selectbox("Độ tuổi", ["Chọn độ tuổi"] + list(AGE_BANDS))
        person_age = AGE_BANDS.get(age_band)
        person_income = st.number_input(
            "Thu nhập hàng năm (USD)",
            min_value=0.0,
            value=None,
            step=1000.0,
        )
        if person_income is not None and person_income > 0:
            st.caption(number_to_vietnamese_words(person_income))
    with personal_right:
        employment_band = st.selectbox(
            "Thâm niên công tác hiện tại",
            ["Chọn thâm niên"] + list(EMPLOYMENT_BANDS),
        )
        person_emp_length = EMPLOYMENT_BANDS.get(employment_band)
        person_home_ownership = st.selectbox(
            "Tình trạng nhà ở hiện tại",
            ["Chọn tình trạng nhà ở"] + list(HOME_OWNERSHIP_OPTIONS),
        )

    st.markdown('<div class="section-label">Thông tin khoản vay mong muốn</div>', unsafe_allow_html=True)
    loan_left, loan_right = st.columns(2)
    with loan_left:
        loan_amnt = st.number_input(
            "Số tiền đề xuất vay (USD)",
            min_value=0.0,
            value=None,
            step=500.0,
        )
        if loan_amnt is not None and loan_amnt > 0:
            st.caption(number_to_vietnamese_words(loan_amnt))
        loan_intent = st.selectbox(
            "Mục đích sử dụng vốn",
            ["Chọn mục đích sử dụng vốn"] + list(LOAN_INTENT_OPTIONS),
        )
    with loan_right:
        cb_person_default_on_file = st.selectbox(
            "Quý khách đã từng có lịch sử nợ quá hạn/nợ xấu trước đây không?",
            ["Chọn lịch sử tín dụng", "Chưa từng có nợ quá hạn", "Đã từng có nợ quá hạn / nợ xấu"],
        )
        st.caption("Thông tin này được dùng trực tiếp để đánh giá rủi ro tín dụng sơ bộ.")

    submitted = st.form_submit_button(
        "Tra cứu khả năng phê duyệt khoản vay",
        use_container_width=True,
    )

if submitted:
    missing_fields = []
    if person_age is None:
        missing_fields.append("độ tuổi")
    if person_emp_length is None:
        missing_fields.append("thâm niên công tác")
    if person_income is None or person_income <= 0:
        missing_fields.append("thu nhập hàng năm")
    if loan_amnt is None or loan_amnt <= 0:
        missing_fields.append("số tiền đề xuất vay")
    if person_home_ownership not in HOME_OWNERSHIP_OPTIONS:
        missing_fields.append("tình trạng nhà ở")
    if loan_intent not in LOAN_INTENT_OPTIONS:
        missing_fields.append("mục đích sử dụng vốn")
    if cb_person_default_on_file == "Chọn lịch sử tín dụng":
        missing_fields.append("lịch sử tín dụng")

    if missing_fields:
        st.warning("Vui lòng hoàn tất: " + ", ".join(missing_fields) + ".")
        st.stop()

    default_history = "Y" if cb_person_default_on_file.startswith("Đã từng") else "N"
    loan_percent_income = loan_amnt / person_income
    model_values = {
        "person_age": person_age,
        "person_income": person_income,
        "person_home_ownership": HOME_OWNERSHIP_OPTIONS[person_home_ownership],
        "person_emp_length": person_emp_length,
        "loan_intent": LOAN_INTENT_OPTIONS[loan_intent],
        "loan_amnt": loan_amnt,
        "loan_percent_income": loan_percent_income,
        "cb_person_default_on_file": default_history,
    }

    try:
        prediction, default_probability, feature_contributions = predict_risk(
            model_values, model, scaler, model_columns
        )
        st.session_state.last_assessment = (
            prediction,
            default_probability,
            feature_contributions,
            model_metrics,
        )
        render_assessment_result(
            prediction,
            default_probability,
            feature_contributions,
            model_metrics,
        )
    except Exception as error:
        st.error(f"Không thể hoàn tất tra cứu. Vui lòng thử lại sau: {error}")

if not submitted and st.session_state.page == "main" and "last_assessment" in st.session_state:
    render_assessment_result(*st.session_state.last_assessment)
