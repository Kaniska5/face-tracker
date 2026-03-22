import streamlit as st
import sqlite3
import pandas as pd
import os
from PIL import Image

st.set_page_config(page_title='Face Tracker Dashboard', layout='wide')
st.title('Intelligent Face Tracker Dashboard')
st.markdown('YOLO + InsightFace + Centroid Tracker | Unique Visitor Counter')
st.divider()

DB_PATH = '/content/face_tracker/visitors.db'

def load_data():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query('SELECT * FROM visitors ORDER BY timestamp ASC', conn)
    uv = pd.read_sql_query('SELECT * FROM unique_visitors ORDER BY first_seen', conn)
    conn.close()
    return df, uv

df, uv = load_data()

c1, c2, c3, c4 = st.columns(4)
c1.metric('Unique Visitors', len(uv))
c2.metric('Videos Processed', df['video_file'].nunique())
c3.metric('Entry Events', len(df[df['event_type'] == 'entry']))
c4.metric('Exit Events', len(df[df['event_type'] == 'exit']))
st.divider()

st.subheader('Filter by Video')
vids = ['All Videos'] + sorted(df['video_file'].unique().tolist())
sel_vid = st.selectbox('Select Video', vids)
fdf = df.copy()
if sel_vid != 'All Videos':
    fdf = fdf[fdf['video_file'] == sel_vid]
st.divider()

st.subheader('Detected Faces Gallery')
visible_faces = (
    fdf[fdf['event_type'] == 'entry']
    .sort_values('timestamp')
    .drop_duplicates(subset='face_id', keep='first')
    .head(12)
)
cols = st.columns(6)
shown = 0
for _, row in visible_faces.iterrows():
    path = str(row['image_path']) if row['image_path'] else ''
    if path and os.path.exists(path):
        with cols[shown % 6]:
            img = Image.open(path)
            fid = row['face_id'][:6]
            st.image(img, caption='ID: ' + fid, use_container_width=True)
        shown += 1
    if shown >= 12:
        break
if shown == 0:
    st.info('No face images found.')
st.divider()

st.subheader('Visitor Timeline by Hour')
if not fdf.empty:
    plot_df = fdf.copy()
    plot_df['hour'] = pd.to_datetime(plot_df['timestamp'], errors='coerce').dt.hour
    tl = (
        plot_df[plot_df['event_type'] == 'entry']
        .drop_duplicates(subset='face_id')
        .groupby('hour')
        .size()
        .reset_index(name='visitors')
    )
    if not tl.empty:
        st.bar_chart(tl.set_index('hour'))
    else:
        st.info('No data to plot.')
st.divider()

st.subheader('Event Log')
st.dataframe(
    fdf[fdf['event_type'] == 'entry'][['face_id', 'timestamp', 'video_file']],
    use_container_width=True,
    height=350
)

st.subheader('Unique Visitors Registry')
st.dataframe(uv, use_container_width=True, height=300)
st.divider()

csv = fdf.to_csv(index=False)
st.download_button('Download Event Log as CSV', csv, 'visitor_log.csv', 'text/csv')
st.markdown('---')
st.caption('This project is a part of a hackathon run by https://katomaran.com')
