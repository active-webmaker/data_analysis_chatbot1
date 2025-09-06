import os
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
import streamlit as st

# Optional plotting libs
import matplotlib.pyplot as plt
import seaborn as sns


# -----------------------------
# Config
# -----------------------------
st.set_page_config(page_title="Titanic EDA", page_icon="🚢", layout="wide")
sns.set_theme(style="whitegrid")


# -----------------------------
# Data Loading
# -----------------------------
@st.cache_data(show_spinner=False)
def load_seaborn_titanic() -> Optional[pd.DataFrame]:
    try:
        df = sns.load_dataset("titanic")
        # Align column names to common conventions (capitalize first letter for some)
        return df
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def load_local_csv(path: str) -> Optional[pd.DataFrame]:
    try:
        return pd.read_csv(path)
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def read_uploaded_csv(file) -> Optional[pd.DataFrame]:
    try:
        return pd.read_csv(file)
    except Exception:
        try:
            file.seek(0)
            return pd.read_excel(file)
        except Exception:
            return None


def coerce_common_titanic_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Try to standardize likely Titanic columns to a common set if present."""
    df = df.copy()
    # common alternative column names
    rename_map = {
        "survived": "survived",
        "Survived": "survived",
        "pclass": "pclass",
        "Pclass": "pclass",
        "sex": "sex",
        "Sex": "sex",
        "age": "age",
        "Age": "age",
        "sibsp": "sibsp",
        "SibSp": "sibsp",
        "parch": "parch",
        "Parch": "parch",
        "fare": "fare",
        "Fare": "fare",
        "embarked": "embarked",
        "Embarked": "embarked",
        "class": "class",
        "Class": "class",
        "who": "who",
        "adult_male": "adult_male",
        "deck": "deck",
        "embark_town": "embark_town",
        "alone": "alone",
        "name": "name",
        "Name": "name",
        "ticket": "ticket",
        "Ticket": "ticket",
        "cabin": "cabin",
        "Cabin": "cabin",
        "boat": "boat",
        "body": "body",
        "home.dest": "home_dest",
    }

    # Only rename those that exist
    rename_actual = {k: v for k, v in rename_map.items() if k in df.columns}
    if rename_actual:
        df = df.rename(columns=rename_actual)

    # Ensure categorical types where appropriate if columns exist
    for cat_col in ["survived", "sex", "embarked", "class", "who", "deck", "embark_town", "alone"]:
        if cat_col in df.columns:
            df[cat_col] = df[cat_col].astype("category")

    return df


def load_data() -> Tuple[Optional[pd.DataFrame], str]:
    """Load data by priority: upload > local csv > seaborn."""
    st.sidebar.subheader("데이터 불러오기")
    file = st.sidebar.file_uploader("CSV 또는 Excel 파일 업로드", type=["csv", "xlsx", "xls"])  # type: ignore

    if file is not None:
        df = read_uploaded_csv(file)
        if df is not None and not df.empty:
            return coerce_common_titanic_columns(df), "uploaded"
        st.sidebar.error("파일을 읽지 못했습니다. CSV 또는 Excel 형식인지 확인하세요.")

    # Try local file in common places
    for candidate in [
        "titanic.csv",
        os.path.join(os.getcwd(), "titanic.csv"),
    ]:
        if os.path.exists(candidate):
            df = load_local_csv(candidate)
            if df is not None and not df.empty:
                return coerce_common_titanic_columns(df), f"local:{candidate}"

    # Fallback to seaborn dataset
    df = load_seaborn_titanic()
    if df is not None and not df.empty:
        return coerce_common_titanic_columns(df), "seaborn"

    return None, "none"


# -----------------------------
# Sidebar Filters
# -----------------------------

def sidebar_filters(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.subheader("필터")

    work_df = df.copy()

    # Numeric filters
    if "age" in work_df.columns:
        min_age = int(np.floor(work_df["age"].dropna().min())) if work_df["age"].notna().any() else 0
        max_age = int(np.ceil(work_df["age"].dropna().max())) if work_df["age"].notna().any() else 80
        age_range = st.sidebar.slider("나이 (age)", min_value=min_age, max_value=max_age, value=(min_age, max_age))
        work_df = work_df[(work_df["age"].isna()) | ((work_df["age"] >= age_range[0]) & (work_df["age"] <= age_range[1]))]

    # Categorical filters
    def cat_filter(col: str, label: str):
        nonlocal work_df
        if col in work_df.columns:
            cats = [x for x in work_df[col].dropna().unique().tolist()]
            cats = [str(x) for x in cats]
            if cats:
                selected = st.sidebar.multiselect(label, options=sorted(cats))
                if selected:
                    work_df = work_df[work_df[col].astype(str).isin(selected)]

    cat_filter("pclass", "객실 등급 (pclass)")
    cat_filter("sex", "성별 (sex)")
    cat_filter("embarked", "승선항 (embarked)")
    cat_filter("class", "객실 등급 (class)")

    return work_df


# -----------------------------
# Basic Stats Section
# -----------------------------

def show_basic_stats(df: pd.DataFrame):
    st.header("기본 통계량 및 데이터 미리보기")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("행 수", df.shape[0])
    with c2:
        st.metric("열 수", df.shape[1])
    with c3:
        st.metric("결측치 비율(%)", round(df.isna().sum().sum() / (df.shape[0]*max(df.shape[1],1)) * 100, 2))
    with c4:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        st.metric("수치형 열 수", len(num_cols))

    st.subheader("데이터 미리보기")
    st.dataframe(df.head(20), use_container_width=True)

    st.subheader("데이터 타입")
    dtypes_df = pd.DataFrame({"column": df.columns, "dtype": df.dtypes.astype(str)})
    st.dataframe(dtypes_df, use_container_width=True)

    st.subheader("결측치 요약")
    na_df = df.isna().sum().reset_index()
    na_df.columns = ["column", "n_missing"]
    na_df["pct_missing"] = (na_df["n_missing"] / len(df) * 100).round(2)
    st.dataframe(na_df.sort_values("n_missing", ascending=False), use_container_width=True)

    # Describe numerical and categorical separately
    with st.expander("수치형 기술통계(describe)", expanded=False):
        if len(num_cols) > 0:
            st.dataframe(df[num_cols].describe().T, use_container_width=True)
        else:
            st.info("수치형 열이 없습니다.")

    cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    with st.expander("범주형 기술통계(describe)", expanded=False):
        if len(cat_cols) > 0:
            st.dataframe(df[cat_cols].describe().T, use_container_width=True)
        else:
            st.info("범주형 열이 없습니다.")


# -----------------------------
# Visualization Section
# -----------------------------

def show_visualizations(df: pd.DataFrame):
    st.header("시각화")

    # Target variable selection (default to 'survived' if exists)
    target_col = None
    if "survived" in df.columns:
        target_col = "survived"
    else:
        # Allow user to select a target if desired
        cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
        target_col = st.selectbox("타겟(분류) 컬럼 선택 (선택 사항)", options=["<선택 안 함>"] + cat_cols)
        if target_col == "<선택 안 함>":
            target_col = None

    # Distribution of numeric columns
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if num_cols:
        st.subheader("수치형 분포 (히스토그램)")
        selected_nums = st.multiselect("히스토그램을 그릴 수치형 컬럼 선택", options=num_cols, default=num_cols[:2])
        n = max(1, len(selected_nums))
        cols = st.columns(min(3, n))
        for i, col in enumerate(selected_nums):
            with cols[i % len(cols)]:
                fig, ax = plt.subplots(figsize=(4, 3))
                sns.histplot(data=df, x=col, hue=target_col if target_col else None, kde=True, ax=ax)
                ax.set_title(f"{col} 분포")
                st.pyplot(fig, use_container_width=True)

    # Count plots for categorical columns
    cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    if cat_cols:
        st.subheader("범주형 분포 (countplot)")
        selected_cats = st.multiselect("카운트플롯을 그릴 범주형 컬럼 선택", options=cat_cols, default=[c for c in ["sex", "pclass", "embarked"] if c in cat_cols])
        n = max(1, len(selected_cats))
        cols = st.columns(min(3, n))
        for i, col in enumerate(selected_cats):
            with cols[i % len(cols)]:
                fig, ax = plt.subplots(figsize=(4, 3))
                if target_col and col != target_col:
                    sns.countplot(data=df, x=col, hue=target_col, ax=ax)
                else:
                    sns.countplot(data=df, x=col, ax=ax)
                ax.set_title(f"{col} 분포")
                ax.tick_params(axis='x', rotation=30)
                st.pyplot(fig, use_container_width=True)

    # Box plots for numeric vs category
    if num_cols and cat_cols:
        st.subheader("박스플롯 (수치형 vs 범주형)")
        x_col = st.selectbox("범주형 (x)", options=cat_cols, index=min(cat_cols.index("sex") if "sex" in cat_cols else 0, len(cat_cols)-1))
        y_col = st.selectbox("수치형 (y)", options=num_cols, index=min(num_cols.index("age") if "age" in num_cols else 0, len(num_cols)-1))
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.boxplot(data=df, x=x_col, y=y_col, hue=target_col if (target_col and target_col != x_col) else None, ax=ax)
        ax.tick_params(axis='x', rotation=30)
        st.pyplot(fig, use_container_width=True)

    # Correlation heatmap (numeric)
    if len(num_cols) >= 2:
        st.subheader("상관관계 히트맵 (수치형)")
        corr = df[num_cols].corr(numeric_only=True)
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
        st.pyplot(fig, use_container_width=True)


# -----------------------------
# Main App
# -----------------------------

def main():
    st.title("🚢 Titanic 데이터 EDA")
    st.write("업로드한 데이터 또는 예시 데이터를 사용해 탐색적 데이터 분석을 수행합니다.")

    df, source = load_data()
    if df is None:
        st.error("데이터를 불러올 수 없습니다. 좌측 사이드바에서 CSV/Excel을 업로드하거나 인터넷 연결 후 다시 시도하세요.")
        st.stop()

    st.caption(f"데이터 소스: {source}")

    # Sidebar filters
    filtered_df = sidebar_filters(df)

    # Column selector for working view
    st.sidebar.subheader("열 선택")
    selected_cols = st.sidebar.multiselect("분석에 사용할 열 선택", options=list(filtered_df.columns), default=list(filtered_df.columns))
    if selected_cols:
        work_df = filtered_df[selected_cols]
    else:
        st.sidebar.warning("최소 1개의 열을 선택하세요. 모든 열을 사용합니다.")
        work_df = filtered_df

    show_basic_stats(work_df)
    st.markdown("---")
    show_visualizations(work_df)


if __name__ == "__main__":
    main()
