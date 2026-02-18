import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import tempfile
import os
from io import BytesIO
# NOTE: All necessary ML imports are placed here
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures 
from sklearn.pipeline import make_pipeline 

# --- PAGE CONFIG ---
st.set_page_config(page_title="Cummins Engine Data Dashboard", page_icon="🚛", layout="wide")

# --- UI ENHANCEMENTS: CUSTOM CSS ---
def apply_custom_css():
    st.markdown("""
        <style>
        :root {
            --primary-bg: #FFFFFF;
            --secondary-bg: #F0F2F6;
            --accent-color: #0E77B5;
            --text-color: #333333;
        }
        .stApp { background-color: var(--primary-bg); color: var(--text-color); }
        .st-emotion-cache-1cypcdb, .st-emotion-cache-p5m9b9, .st-emotion-cache-17z5936, .st-emotion-cache-1r6dm1 {
             background-color: var(--secondary-bg); 
        }
        h1, h2, h3 { color: var(--accent-color) !important; }
        .stButton>button { background-color: var(--accent-color); color: white !important; border-radius: 8px; }
        </style>
    """, unsafe_allow_html=True)

apply_custom_css()

# --- Helper Functions ---
def find_header_row(file_path):
    for encoding in ['ISO-8859-1', 'utf-8', 'latin-1']:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                for i, line in enumerate(f):
                    if line.strip().startswith('"Date","Time"') or line.strip().startswith('Date,Time') or ('ID' in line and 'Frame' in line):
                        return i
            return 0 
        except: continue
    return -1

@st.cache_data
def load_data(uploaded_file):
    try:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=f"_{uploaded_file.name}")
        tmp.write(uploaded_file.getbuffer())
        file_path = tmp.name
        tmp.close()
        header_row = find_header_row(file_path)
        if header_row == -1: return None, "File reading failed."
        
        df = pd.read_csv(file_path, low_memory=False, encoding='ISO-8859-1', skiprows=header_row)
        if 'ID' in df.columns and 'Frame' in df.columns:
            df['ID'] = df['ID'].astype(str).str.replace('"', '').str.strip()

            # --- Unify hex data into the 'data' column ---
            # Some CSV formats (e.g. 500kbps logs) put hex data directly in 'Frame'
            # (like 'x| FC FF FA FA FF FF FF FF') with 'data' column empty/NaN,
            # while others put 'Extended Frame' in Frame and hex in 'data'.
            if 'data' in df.columns:
                mask = df['data'].isna() & df['Frame'].astype(str).str.startswith('x|')
                df.loc[mask, 'data'] = df.loc[mask, 'Frame']
            else:
                # If there's no 'data' column at all, create one from Frame
                df['data'] = df['Frame'].where(
                    df['Frame'].astype(str).str.startswith('x|')
                )

            # --- Fill missing type, DLC, CAN-FD for short-format rows ---
            # Short-format rows only have 7 fields; type/DLC/CAN-FD are NaN.
            # We infer them from the hex data.
            if 'type' in df.columns:
                df.loc[df['type'].isna() & df['data'].notna(), 'type'] = 'data frame'
            if 'DLC' in df.columns:
                dlc_mask = df['DLC'].isna() & df['data'].notna()
                if dlc_mask.any():
                    df.loc[dlc_mask, 'DLC'] = df.loc[dlc_mask, 'data'].apply(
                        lambda x: len(str(x).replace('x|', '').replace('|', '').strip().split())
                    )
            if 'CAN-FD' in df.columns:
                df.loc[df['CAN-FD'].isna() & df['data'].notna(), 'CAN-FD'] = 'CAN'

            # --- Find and parse the time column ---
            time_col = None
            if 'system time' in df.columns:
                time_col = 'system time'
            elif 'Timestamp' in df.columns:
                time_col = 'Timestamp'

            if time_col:
                raw_time = df[time_col].astype(str).str.replace('="', '').str.replace('"', '').str.strip()

                # Try parsing as datetime (handles 'HH:MM:SS.fff', 'YYYY-MM-DD HH:MM:SS', etc.)
                parsed_dt = pd.to_datetime(raw_time, errors='coerce', format='mixed')

                # If pure time strings like '13:43:26.718', pd.to_datetime sets date to today
                # Convert to total milliseconds from midnight for interval calculation
                if parsed_dt.notna().sum() > 0:
                    df['Timestamp_dt'] = parsed_dt
                    # Total ms from midnight = hours*3600000 + minutes*60000 + seconds*1000 + microseconds/1000
                    df['Timestamp_ms'] = (
                        parsed_dt.dt.hour * 3600000 +
                        parsed_dt.dt.minute * 60000 +
                        parsed_dt.dt.second * 1000 +
                        parsed_dt.dt.microsecond / 1000
                    )
                else:
                    # Fallback: try as pure numeric
                    df['Timestamp_ms'] = pd.to_numeric(raw_time, errors='coerce')

                # Rename original column to 'Timestamp' for display
                if time_col != 'Timestamp':
                    df.rename(columns={time_col: 'Timestamp'}, inplace=True)

            return df, "Raw CAN Log"
        elif 'Engine Speed (RPM)' in df.columns:
            # Create a map to store original column names for look-up during encoding
            original_col_map = {} 
            
            # Combine Date and Time into a single datetime column
            if 'Date' in df.columns and 'Time' in df.columns:
                try:
                    df['DateTimeStr'] = df['Date'].astype(str) + ' ' + df['Time'].astype(str)
                    df['Timestamp'] = pd.to_datetime(df['DateTimeStr'], errors='coerce')
                    df.drop(columns=['DateTimeStr'], inplace=True)
                except:
                    pass
            
            # Selective Coercion and Label Encoding
            for col in df.columns:
                if col not in ['Timestamp', 'Date', 'Time', 'Log Index']:
                    series = pd.to_numeric(df[col], errors='coerce')
                    numeric_percentage = series.notna().sum() / len(df)
                    
                    if numeric_percentage > 0.5:
                        df[col] = series
                    else:
                        if df[col].dtype == 'object':
                            original_col_map[col] = df[col] 
                            encoded_col_name = f"{col} (Encoded)"
                            df[encoded_col_name] = df[col].astype('category').cat.codes
            
            st.session_state['original_category_map'] = original_col_map
            return df, "INSITE Sensor Data"
        return df, "Unknown"
    except Exception as e: return None, str(e)

# --- Helper: Normalization ---
def normalize_column(df, col_name):
    """Scales a single column between 0 and 1."""
    max_val = df[col_name].max()
    min_val = df[col_name].min()
    if max_val != min_val:
        return (df[col_name] - min_val) / (max_val - min_val)
    return df[col_name]

# --- SIDEBAR ---
st.sidebar.header("1. Data Source")
uploaded_files = st.sidebar.file_uploader("Upload CSV File(s)", type=["csv"], accept_multiple_files=True)

if 'original_category_map' not in st.session_state:
    st.session_state['original_category_map'] = {}

if uploaded_files:
    primary_file = uploaded_files[0]
    df_result, file_type = load_data(primary_file)

    if df_result is not None and len(df_result) > 0:
        df_global = df_result.copy() 
        st.success(f"✅ Baseline File: **{primary_file.name}** loaded.")

        if file_type == "Raw CAN Log" and len(uploaded_files) > 1:
            st.title("📡 Multi-File CAN Differential Analysis")
            can_tab1, can_tab2, can_tab3, can_tab4 = st.tabs(["📊 Traffic Overview", "🔍 Manual ID Comparison", "🧪 Advanced Re-Analysis", "📂 Advanced Excel Export"])
            
            with can_tab1:
                id_counts = df_global['ID'].value_counts().reset_index()
                id_counts.columns = ['CAN ID', 'Count']
                st.plotly_chart(px.bar(id_counts.head(20), x='CAN ID', y='Count', title="Top 20 Active IDs"), width="stretch")

            with can_tab2:
                file_names = [f.name for f in uploaded_files]
                f1_name = st.selectbox("Select File 1 (Master Log):", file_names, index=0)
                f2_name = st.selectbox("Select File 2 (WITHOUT Sensor Log):", file_names, index=1 if len(file_names)>1 else 0)
                df1, _ = load_data(uploaded_files[file_names.index(f1_name)])
                df2, _ = load_data(uploaded_files[file_names.index(f2_name)])
                ids1, ids2 = set(df1['ID'].unique()), set(df2['ID'].unique())
                
                diff_ids = sorted(list(ids1 - ids2))
                st.markdown(f"**Isolated Unique IDs ({len(diff_ids)})**")
                st.table(pd.DataFrame([diff_ids[i:i+10] for i in range(0, len(diff_ids), 10)]))

            with can_tab4:
                st.header("📂 Advanced Controller-Linked Export")
                st.write("Step-by-step sensor isolation & multi-sheet Excel report generator.")

                # --- Initialize session state for multi-sheet support ---
                if 'analysis_sheets' not in st.session_state:
                    st.session_state.analysis_sheets = []
                if 'current_diff_ids' not in st.session_state:
                    st.session_state.current_diff_ids = None
                if 'current_master_count' not in st.session_state:
                    st.session_state.current_master_count = 0
                if 'current_filtered_count' not in st.session_state:
                    st.session_state.current_filtered_count = 0

                file_names = [f.name for f in uploaded_files]

                # ============================================
                # STEP 1: File Selection & Unique ID Counts
                # ============================================
                st.subheader("Step 1: Select Files")
                col1, col2 = st.columns(2)
                with col1:
                    master_sel = st.selectbox("📁 Master File (All Sensors):", file_names, key='m_sel')
                with col2:
                    filtered_sel = st.selectbox("📁 Filtered File (Missing Sensor):", file_names, key='f_sel')

                # Load both files and show unique ID counts
                df_m, _ = load_data(uploaded_files[file_names.index(master_sel)])
                df_f, _ = load_data(uploaded_files[file_names.index(filtered_sel)])

                if df_m is not None and df_f is not None:
                    master_ids = set(df_m['ID'].unique())
                    filtered_ids = set(df_f['ID'].unique())
                    master_count = len(master_ids)
                    filtered_count = len(filtered_ids)

                    mc1, mc2 = st.columns(2)
                    with mc1:
                        st.metric(label=f"🟢 Master: {master_sel}", value=f"{master_count} Unique IDs")
                        with st.expander(f"📝 View all {master_count} Master IDs"):
                            master_id_list = sorted(list(master_ids))
                            id_rows_m = [master_id_list[i:i+4] for i in range(0, len(master_id_list), 4)]
                            st.table(pd.DataFrame(id_rows_m))
                    with mc2:
                        st.metric(label=f"🔴 Filtered: {filtered_sel}", value=f"{filtered_count} Unique IDs")
                        with st.expander(f"📝 View all {filtered_count} Filtered IDs"):
                            filtered_id_list = sorted(list(filtered_ids))
                            id_rows_f = [filtered_id_list[i:i+4] for i in range(0, len(filtered_id_list), 4)]
                            st.table(pd.DataFrame(id_rows_f))

                    st.divider()

                    # ============================================
                    # STEP 2: Isolate Sensor IDs
                    # ============================================
                    st.subheader("Step 2: Isolate Sensor IDs")

                    if st.button("🔬 Isolate Sensor IDs", key="isolate_btn"):
                        diff_ids = sorted(list(master_ids - filtered_ids))
                        st.session_state.current_diff_ids = diff_ids
                        st.session_state.current_master_count = master_count
                        st.session_state.current_filtered_count = filtered_count

                    # Display results if isolation has been performed
                    if st.session_state.current_diff_ids is not None:
                        diff_ids = st.session_state.current_diff_ids
                        m_count = st.session_state.current_master_count
                        f_count = st.session_state.current_filtered_count
                        diff_count = len(diff_ids)

                        # Show the mathematical operation
                        st.markdown("### 📐 Set Subtraction Operation")
                        st.info(
                            f"**Master IDs ({m_count})** − **Filtered IDs ({f_count})** = **{diff_count} Isolated Sensor IDs**"
                        )

                        if diff_count > 0:
                            # Show isolated IDs in a clean table
                            st.markdown(f"**Isolated CAN IDs ({diff_count}):**")
                            # Display in rows of 5
                            id_rows = [diff_ids[i:i+5] for i in range(0, len(diff_ids), 5)]
                            st.table(pd.DataFrame(id_rows))

                            st.divider()

                            # ============================================
                            # STEP 3: Generate Analysis Sheet
                            # ============================================
                            st.subheader("Step 3: Generate Analysis Sheet")
                            significance_input = st.text_input(
                                "🏷️ Sensor Significance Label (e.g., NOx, Urea Tank):",
                                value="Target Sensor",
                                key="sig_input"
                            )
                            summary_sheet_label = st.text_input(
                                "📋 Summary Sheet Name (in Excel):",
                                value=f"{significance_input}_Summary",
                                key="sum_sheet_name"
                            )

                            if st.button("⚙️ Generate Analysis", key="gen_analysis_btn"):
                                with st.spinner("Analyzing CAN frames..."):
                                    report_rows = []
                                    for target_id in diff_ids:
                                        id_df = df_m[df_m['ID'] == target_id].sort_values(by='Timestamp_ms')
                                        occurrence_count = len(id_df)
                                        id_df = id_df.copy()

                                        # --- Interval calculation (Timestamp_ms is now proper ms from midnight) ---
                                        id_df['interval'] = id_df['Timestamp_ms'].diff()

                                        non_zero_intervals = id_df['interval'].dropna()
                                        non_zero_intervals = non_zero_intervals[non_zero_intervals > 0]
                                        avg_interval = round(non_zero_intervals.mean(), 2) if len(non_zero_intervals) > 0 else 0

                                        unique_data_vals = id_df['data'].dropna().unique()
                                        # Filter out empty/whitespace-only values that produce { } in Arduino format
                                        unique_data_vals = [v for v in unique_data_vals if str(v).replace('x|', '').replace('|', '').strip()]
                                        num_unique = len(unique_data_vals)

                                        # --- Helper: convert hex string to Arduino format ---
                                        def hex_to_arduino(hex_str):
                                            """Convert 'x| 43 26 39 26 FF FF 00 FC' to '{ 0x43, 0x26, 0x39, 0x26, 0xFF, 0xFF, 0x00, 0xFC }'"""
                                            cleaned = str(hex_str).replace('x|', '').replace('|', '').strip()
                                            bytes_list = cleaned.split()
                                            return '{ ' + ', '.join(f'0x{b.upper()}' for b in bytes_list if len(b) == 2) + ' }'

                                        # --- Determine overall pattern for this ID ---
                                        if occurrence_count == 1:
                                            id_pattern = "SINGLE_TRIGGER"
                                        elif num_unique == 1:
                                            id_pattern = "CONSTANT"
                                        elif num_unique == 2:
                                            id_pattern = "TOGGLED"
                                        else:
                                            id_pattern = "UNIQUE_DATA"

                                        for idx, row in id_df.iterrows():
                                            interval_val = round(row['interval'], 2) if not pd.isna(row['interval']) else 0

                                            # --- SMART COMMENT LOGIC (Arduino Controller Ready) ---
                                            if id_pattern == "SINGLE_TRIGGER":
                                                arduino_data = hex_to_arduino(row['data'])
                                                comment_str = f"Single trigger → TimedFrame | {arduino_data}"

                                            elif id_pattern == "CONSTANT":
                                                comment_str = f"Constant data @ {avg_interval}ms → CANMessage (count={occurrence_count if occurrence_count < 500 else -1})"

                                            elif id_pattern == "TOGGLED":
                                                arduino_a = hex_to_arduino(unique_data_vals[0])
                                                arduino_b = hex_to_arduino(unique_data_vals[1])
                                                comment_str = f"Toggled dataA/dataB @ {avg_interval}ms → AltCANMessage | A={arduino_a} B={arduino_b}"

                                            else:  # UNIQUE_DATA — list ALL unique values
                                                all_arduino = [hex_to_arduino(v) for v in unique_data_vals]
                                                data_list = '\n'.join(all_arduino)
                                                comment_str = f"unique data ({num_unique} patterns) → TimedFrame\n{data_list}"

                                            # Preserve hex data as string
                                            frame_data = str(row['data']).strip() if pd.notna(row['data']) else ""

                                            report_rows.append({
                                                "ID": target_id,
                                                "Data (Hex)": frame_data,
                                                "Time_Interval (ms)": interval_val,
                                                "Avg_Interval (ms)": avg_interval,
                                                "No. of Occurrence": occurrence_count,
                                                "Significance": significance_input,
                                                "Comment": comment_str
                                            })

                                    report_df = pd.DataFrame(report_rows)
                                    # Ensure Data column stays as string
                                    report_df["Data (Hex)"] = report_df["Data (Hex)"].astype(str)

                                    # --- Build Arduino summary for Excel ---
                                    def hex_to_ard(hex_str):
                                        cleaned = str(hex_str).replace('x|', '').replace('|', '').strip()
                                        bl = cleaned.split()
                                        return '{ ' + ', '.join(f'0x{b.upper()}' for b in bl if len(b) == 2) + ' }'

                                    summary_excel_rows = []
                                    for sid in report_df['ID'].unique():
                                        sid_data = report_df[report_df['ID'] == sid]
                                        u_hex = sid_data['Data (Hex)'].unique()
                                        # Filter out empty/whitespace-only values
                                        u_hex = [v for v in u_hex if str(v).replace('x|', '').replace('|', '').strip()]
                                        t_occ = sid_data['No. of Occurrence'].iloc[0]
                                        ivs = sid_data['Time_Interval (ms)']
                                        nz_iv = ivs[ivs > 0]
                                        a_iv = round(nz_iv.mean(), 2) if len(nz_iv) > 0 else 0
                                        n_u = len(u_hex)

                                        if t_occ == 1: s_type = "TimedFrame"
                                        elif n_u == 1: s_type = "CANMessage"
                                        elif n_u == 2: s_type = "AltCANMessage"
                                        else: s_type = "TimedFrame"

                                        all_frames_str = '\n'.join(hex_to_ard(v) for v in u_hex)

                                        summary_excel_rows.append({
                                            "ID": sid,
                                            "Controller Struct": s_type,
                                            "Unique Patterns": n_u,
                                            "Total Occurrences": t_occ,
                                            "Avg Interval (ms)": a_iv,
                                            "Significance": significance_input,
                                            "All Unique Data (Arduino Format)": all_frames_str
                                        })
                                    summary_excel_df = pd.DataFrame(summary_excel_rows)

                                    # Generate sheet name from significance
                                    sheet_name = significance_input.replace(" ", "_")[:31]  # Excel max 31 chars
                                    # Avoid duplicate sheet names
                                    existing_names = [s[0] for s in st.session_state.analysis_sheets]
                                    if sheet_name in existing_names:
                                        sheet_name = f"{sheet_name}_{len(existing_names)+1}"

                                    # Store detail, summary DataFrames, and custom summary sheet name
                                    st.session_state.analysis_sheets.append((sheet_name, report_df, summary_excel_df, summary_sheet_label))
                                    st.success(f"✅ Sheet **'{sheet_name}'** generated with {len(report_df)} rows!")

                            # ============================================
                            # STEP 4: Display All Generated Sheets
                            # ============================================
                            if st.session_state.analysis_sheets:
                                st.divider()
                                st.subheader("Step 4: Review Generated Sheets")

                                for i, (sname, sdf, *_rest) in enumerate(st.session_state.analysis_sheets):
                                    with st.expander(f"📋 Sheet {i+1}: {sname} ({len(sdf)} rows)", expanded=(i == len(st.session_state.analysis_sheets) - 1)):
                                        view_mode = st.radio(
                                            "View Mode:",
                                            ["📊 Detailed (All Occurrences)", "📝 Summary (Unique Patterns per ID)"],
                                            key=f"view_mode_{i}",
                                            horizontal=True
                                        )

                                        if view_mode == "📊 Detailed (All Occurrences)":
                                            st.dataframe(sdf, use_container_width=True)
                                        else:
                                            # Build summary: one row per ID with Arduino-ready comments
                                            def hex_to_arduino_fmt(hex_str):
                                                cleaned = str(hex_str).replace('x|', '').replace('|', '').strip()
                                                bytes_list = cleaned.split()
                                                return '{ ' + ', '.join(f'0x{b.upper()}' for b in bytes_list if len(b) == 2) + ' }'

                                            summary_rows = []
                                            for can_id in sdf['ID'].unique():
                                                id_data = sdf[sdf['ID'] == can_id]
                                                unique_hex = id_data['Data (Hex)'].unique()
                                                total_occ = id_data['No. of Occurrence'].iloc[0]
                                                intervals = id_data['Time_Interval (ms)']
                                                non_zero_intervals = intervals[intervals > 0]
                                                avg_interval = round(non_zero_intervals.mean(), 2) if len(non_zero_intervals) > 0 else 0
                                                num_unique = len(unique_hex)

                                                if total_occ == 1:
                                                    struct_type = "TimedFrame"
                                                    brief = f"Single trigger (1 frame)"
                                                elif num_unique == 1:
                                                    count_val = total_occ if total_occ < 500 else -1
                                                    struct_type = "CANMessage"
                                                    brief = f"Constant data @ {avg_interval}ms, count={count_val}"
                                                elif num_unique == 2:
                                                    struct_type = "AltCANMessage"
                                                    brief = f"Toggled dataA/dataB @ {avg_interval}ms"
                                                else:
                                                    struct_type = "TimedFrame"
                                                    brief = f"unique data ({num_unique} frames)"

                                                summary_rows.append({
                                                    "ID": can_id,
                                                    "Controller Struct": struct_type,
                                                    "Unique Patterns": num_unique,
                                                    "Total Occurrences": total_occ,
                                                    "Avg Interval (ms)": avg_interval,
                                                    "Significance": id_data['Significance'].iloc[0],
                                                    "Arduino Reference": brief
                                                })

                                            summary_df = pd.DataFrame(summary_rows)
                                            st.dataframe(summary_df, use_container_width=True)
                                            st.caption(f"📌 {len(summary_df)} unique IDs → ready to translate into Arduino controller structs")

                                    # --- Arduino Data Reference OUTSIDE the parent expander ---
                                    def hex_to_arduino_ref(hex_str):
                                        cleaned = str(hex_str).replace('x|', '').replace('|', '').strip()
                                        bytes_list = cleaned.split()
                                        return '{ ' + ', '.join(f'0x{b.upper()}' for b in bytes_list if len(b) == 2) + ' }'

                                    st.markdown(f"#### 🔧 Full Arduino Data Reference (per ID) — {sname}")
                                    for can_id in sdf['ID'].unique():
                                        id_data = sdf[sdf['ID'] == can_id]
                                        unique_hex = id_data['Data (Hex)'].unique()
                                        total_occ = id_data['No. of Occurrence'].iloc[0]
                                        intervals = id_data['Time_Interval (ms)']
                                        non_zero_intervals = intervals[intervals > 0]
                                        avg_interval = round(non_zero_intervals.mean(), 2) if len(non_zero_intervals) > 0 else 0
                                        num_unique = len(unique_hex)

                                        if total_occ == 1: struct_type = "TimedFrame"
                                        elif num_unique == 1: struct_type = "CANMessage"
                                        elif num_unique == 2: struct_type = "AltCANMessage"
                                        else: struct_type = "TimedFrame"

                                        all_arduino_lines = [hex_to_arduino_ref(v) for v in unique_hex]

                                        st.markdown(f"**🔹 {can_id}** — `{struct_type}` ({num_unique} unique frames)")
                                        if struct_type == "CANMessage":
                                            code = f'{{ {can_id}, {all_arduino_lines[0]}, {int(avg_interval)}, 0, {total_occ if total_occ < 500 else -1} }}'
                                            st.code(code, language="cpp")
                                        elif struct_type == "AltCANMessage":
                                            code = f'{{ {can_id},\n  /* dataA */ {all_arduino_lines[0]},\n  /* dataB */ {all_arduino_lines[1]},\n  false, {int(avg_interval)}, 0 }}'
                                            st.code(code, language="cpp")
                                        else:
                                            lines = []
                                            for j, frame in enumerate(all_arduino_lines):
                                                lines.append(f'  {{ {can_id}, {frame}, {int(avg_interval)}, false }},  // frame {j+1}')
                                            code = '\n'.join(lines)
                                            st.code(code, language="cpp")
                                            st.caption(f"Total: {len(all_arduino_lines)} unique data frames for {can_id}")

                                st.divider()

                                # ============================================
                                # STEP 5: Add More / Download / Manage
                                # ============================================
                                st.subheader("Step 5: Add More or Download")

                                # --- Per-sheet controls ---
                                st.markdown(f"**📊 Total Sheets: {len(st.session_state.analysis_sheets)}**")
                                for i, (sname, sdf, *_rest) in enumerate(st.session_state.analysis_sheets):
                                    scol1, scol2, scol3 = st.columns([3, 1, 1])
                                    with scol1:
                                        st.markdown(f" {i+1}. `{sname}` — {len(sdf)} rows")
                                    with scol2:
                                        # Individual sheet download (detail + summary)
                                        ind_buffer = BytesIO()
                                        with pd.ExcelWriter(ind_buffer, engine='xlsxwriter') as ind_writer:
                                            sdf.to_excel(ind_writer, index=False, sheet_name=sname)
                                            wb = ind_writer.book
                                            ws = ind_writer.sheets[sname]
                                            txt_fmt = wb.add_format({'num_format': '@'})
                                            ws.set_column(1, 1, 30, txt_fmt)  # Data (Hex) column
                                            comment_fmt = wb.add_format({'text_wrap': True, 'num_format': '@'})
                                            ws.set_column(6, 6, 60, comment_fmt)  # Comment column (shifted by Avg_Interval)
                                            # Write summary sheet if available
                                            if _rest and len(_rest) >= 1 and _rest[0] is not None:
                                                # Use custom summary sheet name if provided (index 1 in _rest)
                                                if len(_rest) >= 2 and _rest[1]:
                                                    sum_name = str(_rest[1]).replace(" ", "_")[:31]
                                                else:
                                                    sum_name = f"{sname[:23]}_Summary" if len(sname) > 23 else f"{sname}_Sum"
                                                _rest[0].to_excel(ind_writer, index=False, sheet_name=sum_name)
                                                sw = ind_writer.sheets[sum_name]
                                                wrap_fmt = wb.add_format({'text_wrap': True, 'num_format': '@'})
                                                sw.set_column(6, 6, 60, wrap_fmt)  # Arduino data column
                                        st.download_button(
                                            label="⬇️ Download",
                                            data=ind_buffer.getvalue(),
                                            file_name=f"{sname}_Analysis.xlsx",
                                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                            key=f"dl_sheet_{i}"
                                        )
                                    with scol3:
                                        if st.button("🗑️ Delete", key=f"del_sheet_{i}"):
                                            st.session_state.analysis_sheets.pop(i)
                                            st.rerun()

                                st.divider()

                                # --- Custom file name ---
                                excel_filename = st.text_input(
                                    "📝 Name your Excel file:",
                                    value="CAN_Sensor_Analysis",
                                    key="excel_filename"
                                )

                                # --- Action buttons row ---
                                acol1, acol2, acol3 = st.columns(3)

                                with acol1:
                                    if st.button("➕ Add Another Sensor Analysis", key="add_more_btn"):
                                        st.session_state.current_diff_ids = None
                                        st.rerun()

                                with acol2:
                                    # Combined multi-sheet download (detail + summary per analysis)
                                    buffer = BytesIO()
                                    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                                        for sname, sdf, *rest in st.session_state.analysis_sheets:
                                            sdf.to_excel(writer, index=False, sheet_name=sname)
                                            workbook = writer.book
                                            worksheet = writer.sheets[sname]
                                            text_format = workbook.add_format({'num_format': '@'})
                                            worksheet.set_column(1, 1, 30, text_format)
                                            comment_format = workbook.add_format({'text_wrap': True, 'num_format': '@'})
                                            worksheet.set_column(6, 6, 60, comment_format)  # Comment column (shifted by Avg_Interval)
                                            # Write companion summary sheet
                                            if rest and len(rest) >= 1 and rest[0] is not None:
                                                if len(rest) >= 2 and rest[1]:
                                                    sum_name = str(rest[1]).replace(" ", "_")[:31]
                                                else:
                                                    sum_name = f"{sname[:23]}_Summary" if len(sname) > 23 else f"{sname}_Sum"
                                                rest[0].to_excel(writer, index=False, sheet_name=sum_name)
                                                sw = writer.sheets[sum_name]
                                                wrap_fmt = workbook.add_format({'text_wrap': True, 'num_format': '@'})
                                                sw.set_column(6, 6, 60, wrap_fmt)

                                    safe_name = excel_filename.strip().replace(" ", "_") if excel_filename.strip() else "CAN_Sensor_Analysis"
                                    st.download_button(
                                        label="📥 Download All Sheets (Combined)",
                                        data=buffer.getvalue(),
                                        file_name=f"{safe_name}.xlsx",
                                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                        key="download_excel_btn"
                                    )

                                with acol3:
                                    if st.button("🗑️ Clear All Sheets & Start Over", key="clear_btn"):
                                        st.session_state.analysis_sheets = []
                                        st.session_state.current_diff_ids = None
                                        st.rerun()

                                # --- FINAL REPORT: Per-ID + Combined + Time Summary ---
                                st.divider()
                                st.markdown("#### 📑 Final Report — Per-ID Sheets + Combined + Time Summary")
                                st.caption("All master file columns, per-ID sheets, a combined sheet, and total time analysis.")

                                if st.button("📥 Generate Final Report Excel", key="final_report_btn"):
                                    final_buffer = BytesIO()
                                    with pd.ExcelWriter(final_buffer, engine='xlsxwriter') as fw:
                                        # Collect all analyzed IDs
                                        all_analyzed_ids = set()
                                        for sname, sdf, *rest in st.session_state.analysis_sheets:
                                            all_analyzed_ids.update(sdf['ID'].unique())
                                        all_analyzed_ids = sorted(all_analyzed_ids)

                                        combined_frames = []
                                        time_summary_rows = []

                                        # --- Master file total time ---
                                        master_sorted = df_m.sort_values(by='Timestamp_ms')
                                        master_first_ms = master_sorted['Timestamp_ms'].iloc[0]
                                        master_last_ms = master_sorted['Timestamp_ms'].iloc[-1]
                                        master_total_ms = round(master_last_ms - master_first_ms, 2)
                                        # Convert ms to readable time
                                        def ms_to_readable(ms_val):
                                            secs = ms_val / 1000
                                            mins = int(secs // 60)
                                            remaining_secs = round(secs % 60, 3)
                                            if mins > 0:
                                                return f"{mins}m {remaining_secs}s"
                                            return f"{remaining_secs}s"

                                        master_first_ts = master_sorted['Timestamp'].iloc[0] if 'Timestamp' in master_sorted.columns else ""
                                        master_last_ts = master_sorted['Timestamp'].iloc[-1] if 'Timestamp' in master_sorted.columns else ""

                                        time_summary_rows.append({
                                            "Source": "Master File (All IDs)",
                                            "First Timestamp": str(master_first_ts),
                                            "Last Timestamp": str(master_last_ts),
                                            "Total Time (ms)": master_total_ms,
                                            "Total Time (Readable)": ms_to_readable(master_total_ms),
                                            "Row Count": len(master_sorted)
                                        })

                                        # --- Per-ID sheets ---
                                        for uid in all_analyzed_ids:
                                            uid_df = df_m[df_m['ID'] == uid].copy()
                                            uid_df = uid_df.sort_values(by='Timestamp_ms')

                                            # Time interval per row
                                            uid_df['Time_Interval (ms)'] = uid_df['Timestamp_ms'].diff()
                                            nz = uid_df['Time_Interval (ms)'].dropna()
                                            nz = nz[nz > 0]
                                            uid_avg = round(nz.mean(), 2) if len(nz) > 0 else 0
                                            uid_df['Avg_Interval (ms)'] = uid_avg

                                            # Total time for this ID
                                            uid_first_ms = uid_df['Timestamp_ms'].iloc[0]
                                            uid_last_ms = uid_df['Timestamp_ms'].iloc[-1]
                                            uid_total_ms = round(uid_last_ms - uid_first_ms, 2)
                                            uid_df['Total_Time (ms)'] = uid_total_ms

                                            uid_first_ts = uid_df['Timestamp'].iloc[0] if 'Timestamp' in uid_df.columns else ""
                                            uid_last_ts = uid_df['Timestamp'].iloc[-1] if 'Timestamp' in uid_df.columns else ""

                                            time_summary_rows.append({
                                                "Source": f"ID: {uid}",
                                                "First Timestamp": str(uid_first_ts),
                                                "Last Timestamp": str(uid_last_ts),
                                                "Total Time (ms)": uid_total_ms,
                                                "Total Time (Readable)": ms_to_readable(uid_total_ms),
                                                "Row Count": len(uid_df)
                                            })

                                            combined_frames.append(uid_df)

                                            # Write individual ID sheet — drop internal helper columns
                                            clean_id = str(uid).replace('0x', '').replace(' ', '')[:20]
                                            id_sheet = clean_id[:31]
                                            existing = list(fw.sheets.keys()) if hasattr(fw, 'sheets') else []
                                            if id_sheet in existing:
                                                id_sheet = f"{id_sheet[:22]}_{len(existing)}"

                                            # Drop internal columns that can corrupt xlsx output
                                            drop_cols = [c for c in ['Timestamp_dt', 'Timestamp_ms'] if c in uid_df.columns]
                                            uid_export = uid_df.drop(columns=drop_cols)
                                            uid_export.to_excel(fw, index=False, sheet_name=id_sheet)
                                            wb = fw.book
                                            ws = fw.sheets[id_sheet]
                                            txt_fmt = wb.add_format({'num_format': '@'})
                                            for ci, cn in enumerate(uid_export.columns):
                                                if cn in ['ID', 'Frame', 'data', 'Timestamp', 'Data (Hex)']:
                                                    ws.set_column(ci, ci, 25, txt_fmt)

                                        # --- Combined sheet (all IDs together, sorted by timestamp) ---
                                        if combined_frames:
                                            combined_df = pd.concat(combined_frames, ignore_index=True)
                                            combined_df = combined_df.sort_values(by='Timestamp_ms')

                                            # Combined total time
                                            comb_first_ms = combined_df['Timestamp_ms'].iloc[0]
                                            comb_last_ms = combined_df['Timestamp_ms'].iloc[-1]
                                            comb_total_ms = round(comb_last_ms - comb_first_ms, 2)
                                            combined_df['Total_Time (ms)'] = comb_total_ms

                                            comb_first_ts = combined_df['Timestamp'].iloc[0] if 'Timestamp' in combined_df.columns else ""
                                            comb_last_ts = combined_df['Timestamp'].iloc[-1] if 'Timestamp' in combined_df.columns else ""

                                            time_summary_rows.append({
                                                "Source": f"Combined ({len(all_analyzed_ids)} IDs)",
                                                "First Timestamp": str(comb_first_ts),
                                                "Last Timestamp": str(comb_last_ts),
                                                "Total Time (ms)": comb_total_ms,
                                                "Total Time (Readable)": ms_to_readable(comb_total_ms),
                                                "Row Count": len(combined_df)
                                            })

                                            # Drop internal columns before writing
                                            drop_cols_c = [c for c in ['Timestamp_dt', 'Timestamp_ms'] if c in combined_df.columns]
                                            combined_export = combined_df.drop(columns=drop_cols_c)
                                            combined_export.to_excel(fw, index=False, sheet_name="All_IDs_Combined")
                                            ws_c = fw.sheets["All_IDs_Combined"]
                                            for ci, cn in enumerate(combined_export.columns):
                                                if cn in ['ID', 'Frame', 'data', 'Timestamp', 'Data (Hex)']:
                                                    ws_c.set_column(ci, ci, 25, txt_fmt)

                                        # --- Time Summary sheet ---
                                        time_df = pd.DataFrame(time_summary_rows)
                                        time_df.to_excel(fw, index=False, sheet_name="Time_Summary")
                                        ws_t = fw.sheets["Time_Summary"]
                                        ws_t.set_column(0, 0, 30)   # Source
                                        ws_t.set_column(1, 2, 22)   # Timestamps
                                        ws_t.set_column(3, 3, 18)   # Total Time ms
                                        ws_t.set_column(4, 4, 20)   # Readable
                                        ws_t.set_column(5, 5, 12)   # Row Count

                                    safe_name = excel_filename.strip().replace(" ", "_") if excel_filename.strip() else "CAN_Sensor_Analysis"
                                    st.download_button(
                                        label="⬇️ Download Final Report Excel",
                                        data=final_buffer.getvalue(),
                                        file_name=f"{safe_name}_Final_Report.xlsx",
                                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                        key="download_final_btn"
                                    )
                        else:
                            st.warning("⚠️ No unique IDs isolated. Both files contain the same CAN IDs.")
        
        else: # Standard INSITE UI
            st.sidebar.divider()
            st.sidebar.header("2. Global Filters")
            total_rows = len(df_global)
            df_filtered = df_global 
            
            numeric_cols = df_global.select_dtypes(include='number').columns.tolist()
            time_cols = ['Log Index']
            if 'Timestamp' in df_global.columns: time_cols.append('Timestamp')
            if 'Date' in df_global.columns: time_cols.append('Date')
            if 'Time' in df_global.columns: time_cols.append('Time')
            status_cols = [col for col in df_global.columns if df_global[col].dtype == 'object' and col not in time_cols]
            x_options = time_cols + status_cols + numeric_cols
            default_x_option = 'Timestamp' if 'Timestamp' in x_options else 'Log Index'
            
            if total_rows > 1:
                start_row, end_row = st.sidebar.slider("Select Data Range:", 0, total_rows, (0, total_rows))
                df_filtered = df_global.iloc[start_row:end_row].copy()
            
            tab1, tab2, tab3, tab4 = st.tabs(["📈 Custom Plotter & Regression", "🧪 ML Analysis", "📄 Data & Export", "⚙️ System Info"])

            with tab1: # Custom Plotter Tab
                if file_type == "INSITE Sensor Data":
                    st.header("📈 Custom Plotting & Regression Analysis")
                    y_numeric_cols = [c for c in df_filtered.select_dtypes(include='number').columns.tolist() if c != 'Log Index']
                    
                    st.sidebar.divider()
                    st.sidebar.header("3. Plot Settings")
                    y_axis_val_list = st.sidebar.multiselect("Y-Axis:", y_numeric_cols, default=[y_numeric_cols[0]] if y_numeric_cols else None, key='plot_y_multi')
                    x_axis_val = st.sidebar.selectbox("X-Axis:", options=x_options, index=x_options.index(default_x_option) if default_x_option in x_options else 0)
                    chart_type = st.sidebar.radio("Chart Type:", ["Line Chart", "Scatter Plot"])
                    norm_x = st.sidebar.checkbox("Normalize X-Axis (0-1)", value=False)
                    norm_y = st.sidebar.checkbox("Normalize Y-Axis (0-1)", value=False)
                    reg_enabled = st.sidebar.checkbox("Show Regression Plot Below", value=False)

                    if y_axis_val_list:
                        plot_df = df_filtered.copy()
                        hover_data_list = []
                        
                        for col in y_axis_val_list:
                            if " (Encoded)" in col:
                                original_name = col.replace(" (Encoded)", "")
                                plot_df[f"Label_{original_name}"] = df_global[original_name].iloc[plot_df.index]
                                hover_data_list.append(f"Label_{original_name}")
                            
                            if norm_y:
                                plot_df[f"Original_{col}"] = plot_df[col]
                                plot_df[col] = normalize_column(plot_df, col)
                                hover_data_list.append(f"Original_{col}")

                        if chart_type == "Line Chart": fig = px.line(plot_df, x=x_axis_val, y=y_axis_val_list, hover_data=hover_data_list)
                        else: fig = px.scatter(plot_df, x=x_axis_val, y=y_axis_val_list, hover_data=hover_data_list)
                        st.plotly_chart(fig, use_container_width=True)

                        if reg_enabled:
                            st.divider(); st.subheader("🧪 Regression Model Training")
                            col_y, col_x, col_model = st.columns(3)
                            with col_y: reg_target_col = st.selectbox("Target (Y):", y_numeric_cols, key='reg_target_tab1')
                            with col_x:
                                reg_feature_options = [c for c in y_numeric_cols if c != reg_target_col]
                                reg_feature_cols = st.multiselect("Features (X):", reg_feature_options, key='reg_feature_tab1')
                            
                            with col_model:
                                if len(reg_feature_cols) == 1: m_opts = ["Simple Linear", "Polynomial"]
                                elif len(reg_feature_cols) > 1: m_opts = ["Multiple Linear", "Polynomial"]
                                else: m_opts = ["Linear", "Polynomial"]
                                model_selection = st.selectbox("Model Type:", options=m_opts, key='reg_model_type')
                                poly_degree = st.number_input("Degree:", 2, 5, 2) if model_selection == "Polynomial" else 1

                            if st.button("Run Regression Model", key='run_reg_tab1_button', use_container_width=True):
                                if not reg_feature_cols: st.warning("Select Feature (X).")
                                else:
                                    df_reg = plot_df[[reg_target_col] + reg_feature_cols].dropna()
                                    X_train, X_test, Y_train, Y_test = train_test_split(df_reg[reg_feature_cols], df_reg[reg_target_col], test_size=0.3, random_state=42)
                                    
                                    if "Polynomial" in model_selection: model = make_pipeline(PolynomialFeatures(poly_degree, include_bias=False), LinearRegression())
                                    else: model = LinearRegression()
                                    
                                    model.fit(X_train, Y_train); Y_pred = model.predict(X_test)
                                    st.metric("R-squared Score", f"{model.score(X_test, Y_test):.4f}")
                                    
                                    lr = model.named_steps['linearregression'] if "Polynomial" in model_selection else model
                                    feats = model.named_steps['polynomialfeatures'].get_feature_names_out(reg_feature_cols) if "Polynomial" in model_selection else reg_feature_cols
                                    eq = f"{reg_target_col.replace(' ','')} = {lr.intercept_:.4f}"
                                    for c, f_name in zip(lr.coef_, feats): eq += f" {'+' if c > 0 else '-'} {abs(c):.4f} * {f_name.replace(' ', '*')}"
                                    st.code(eq)

                                    st.write("#### Actual vs. Predicted Value Plot")
                                    plot_data = pd.DataFrame({"Actual Value": Y_test, "Predicted Value": Y_pred})
                                    min_v, max_v = plot_data[['Actual Value', 'Predicted Value']].min().min(), plot_data[['Actual Value', 'Predicted Value']].max().max()
                                    fig_reg = px.scatter(plot_data, x="Actual Value", y="Predicted Value", title="Regression Performance")
                                    fig_reg.add_shape(type="line", x0=min_v, y0=min_v, x1=max_v, y1=max_v, line=dict(color="Red", width=2, dash="dash"))
                                    st.plotly_chart(fig_reg, use_container_width=True)

                elif file_type == "Raw CAN Log":
                    st.header("📡 CAN Message Frequency Analysis")
                    id_counts = df_filtered['ID'].value_counts().reset_index()
                    id_counts.columns = ['CAN ID', 'Count']
                    fig = px.bar(id_counts.head(20), x='Count', y='CAN ID', orientation='h', title="Top 20 IDs")
                    st.plotly_chart(fig, use_container_width=True)

            with tab2: # ML Analysis
                if file_type == "INSITE Sensor Data":
                    st.header("🧪 Feature Correlation Analysis")
                    ml_num = [c for c in df_global.select_dtypes(include='number').columns if c != 'Log Index']
                    feat_corr = st.multiselect("Select Features:", ml_num, default=ml_num[:5])
                    if feat_corr:
                        fig_corr = px.imshow(df_global[feat_corr].corr(), text_auto=".2f", color_continuous_scale='RdBu_r')
                        st.plotly_chart(fig_corr, use_container_width=True)

            with tab3: # Data Export
                st.subheader("📄 Filtered Data Inspector")
                st.dataframe(df_filtered, use_container_width=True)
                st.download_button("📥 Download Filtered CSV", df_filtered.to_csv(index=False).encode('utf-8'), "filtered_data.csv")

            with tab4: # Info
                st.subheader("⚙️ File Metadata")
                st.metric("Detected File Type", file_type)
                st.metric("Total Rows", f"{total_rows:,}")

else:
    st.info("👋 Please upload a file to begin.")