from dash import Dash, html, dcc, Input, Output, ctx, State
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
'''
import plotly.graph_objects as go; import numpy as np
from plotly_resampler import FigureResampler, FigureWidgetResampler
'''
import numpy as np
import pandas as pd
from datetime import timedelta, datetime
import mne
import base64
import io
import pdb

class EDFInfo():

    def __init__(self, file_name, eeg_data):

        self.file_name = file_name
        self.eeg_data = eeg_data
        self.eeg_channels = self.eeg_data.info['ch_names']
        self.eeg_data_raw = self.eeg_data.get_data(picks=self.eeg_channels)
        self.eeg_start_time = pd.to_datetime(self.eeg_data.info['meas_date'] if 'meas_date' in self.eeg_data.info else datetime.today())
        self.eeg_sampling_freq = float(self.eeg_data.info['sfreq'])
        self.df = pd.DataFrame(self.eeg_data_raw.T, columns=self.eeg_channels)
        self.df = self.df.reset_index()
        self.df = self.df.rename(columns={'index': 'Time'})
        self.chunk_in_sec = 30

        self.setChunkStart(0)
        self.setChunkSize(30)

    def setChunkSize(self, interval):
        
        if self.chunk_in_sec != interval: self.setChunkStart(0)

        self.chunk_in_sec = interval
        
        if interval == 300: self.chunk_in_time_text = '5 min'
        elif interval == 600: self.chunk_size_text = '10 min'
        else:
            self.chunk_in_time_text = f'{interval} sec'

        self.chunk = self.chunk_in_sec * self.eeg_sampling_freq

    def setChunkStartByTime(self, time_start):

        try:
            hh, mm, ss = time_start.split(":")
            hh, mm, ss = int(hh), int(mm), int(ss)
            
            if 0 <= hh <= 23 and 0 <= mm <= 59 and 0 <= ss <= 59:
                day = datetime.today()
                time1 = datetime.combine(day, pd.to_datetime(time_start).time())
                time2 = datetime.combine(day, self.eeg_start_time.time())
                chunk_start = (time1-time2).total_seconds() * self.eeg_sampling_freq

        except Exception as e:
            print(f"Error: {e}")
            return
        
        self.setChunkStart(chunk_start)

    def setChunkStart(self, chunk_start):
        self.start_point = chunk_start

    def decChunk(self):
        self.start_point = max(self.start_point-self.chunk, 0)

    def incChunk(self):
        self.start_point = min(self.start_point+self.chunk, self.df.shape[0]-self.chunk)

    def firstChunk(self):
        self.setChunkStart(0)

    def lastChunk(self):
        self.setChunkStart(self.df.shape[0]-self.chunk)

    def extractData(self):
        df_current = self.df.loc[self.start_point : self.start_point + self.chunk].copy()
        df_current['Time'] = df_current['Time'].apply(lambda x: self.eeg_start_time + timedelta(seconds=x/self.eeg_sampling_freq))
        return df_current
    
    def hasPrevData(self):
        return self.start_point > 0
    
    def hasNextData(self):
        return (self.start_point + self.chunk) < self.df.shape[0]

def processLayoutUpdate():

    df_current = edf_info.extractData()
    df_current = df_current.melt(['Time']).rename(columns={'variable': 'Channel', 'value':'Amp'})
    has_prev_data, has_next_data = edf_info.hasPrevData(), edf_info.hasNextData()

    fig = px.line(df_current, x="Time", y="Amp", color="Channel", facet_row="Channel", height=100*len(edf_info.eeg_channels))
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1], textangle=0))
    fig.for_each_yaxis(lambda y: y.update(title = ''))
    fig['layout']['xaxis']['showticklabels'] = True
    fig['layout']['xaxis']['side'] = 'bottom'
    fig['layout']['xaxis8']['showticklabels'] = True
    fig['layout']['xaxis8']['side'] = 'top'
    fig.add_annotation(x=-0.03,y=0.5, text="Amplitude (db)", textangle=-90, xref="paper", yref="paper")
    fig.update_layout(showlegend=False)
    
    '''    
    fig = make_subplots(rows=len(eeg_channels), cols=1,                        shared_xaxes=True,
                        vertical_spacing=0.02)

    for i, ch in enumerate(eeg_channels):
        fig.add_trace(go.Line(x=df_current['Time'], y=df_current[ch]), row=i+1, col=1)

    fig.update_layout(height=1000, width=1000, showlegend=False)
    fig['layout']['xaxis']['title'] = 'Time'
    fig['layout']['xaxis']['showticklabels'] = True
    fig['layout']['xaxis']['side'] = 'top'
    fig.add_annotation(x=-0.04, y=0.5, text="Amplitude (db)", textangle=-90, xref="paper", yref="paper")
    '''
    '''
    fig = FigureResampler(go.Figure())
    fig.add_trace(go.Scattergl(name='F3', showlegend=True), hf_x=df['Time'], hf_y=df['F3'])
    fig.show_dash(mode='inline')
    '''

    return (
        dcc.Graph(
            figure=fig,
            style={
                'width': '100%',
            }
    ), 
    f'< Previous {edf_info.chunk_in_time_text}', 
    f'Next {edf_info.chunk_in_time_text} >', 
    (not has_prev_data), 
    (not has_next_data), 
    f'First {edf_info.chunk_in_time_text}', 
    f'Last {edf_info.chunk_in_time_text}', 
    (not has_prev_data), 
    (not has_next_data))

def parse_contents(contents, file_name):

    global edf_info

    content_type, content_string = contents.split(',')
    content_bytes = base64.b64decode(content_string)
    try:
        if file_name.endswith('.edf'):
            edf_file_path = f'data/{file_name}'
            with open(edf_file_path, 'wb') as f:
                f.write(content_bytes)
            eeg_data = mne.io.read_raw_edf(edf_file_path, encoding='latin1', verbose=False)
            edf_info = EDFInfo(file_name, eeg_data)
    except Exception as e:
        print(f"Error: {e}")
        return (f"Shows channel-wise EEG signals in {chunk_in_time_text} interval", "There was an error processing this file", '')
    return ([html.Span("Shows channel-wise EEG signals for "), html.Strong(edf_info.file_name), html.Span(f" in {edf_info.chunk_in_time_text} interval")], '', '')
    
app = Dash(__name__)
edf_info, chunk_in_time_text = None, "30 sec"

#edf_file_path = 'data/19-0261_F_9.3_1_di_al.edf'
#eeg_data = mne.io.read_raw_edf(edf_file_path, verbose=False)
#edf_info = EDFInfo('sample_file', eeg_data)
#fig_container, btns = processLayoutUpdate()

app.layout = html.Div(
        children=[
            html.Div(
                children=[
                    html.Div(
                        style={
                            "flex": "1"
                        }
                    ),
                    html.Div(
                        children=[
                            html.H1(
                                children='EEG Data Visualization',
                                style = {
                                    'textAlign': 'center',
                                }
                            ),
                            html.P(
                                children=f'Shows EEG signals for selected channels in {chunk_in_time_text} interval',
                                style = {
                                    'textAlign': 'center',
                                },
                                id='file-info'
                            )
                        ],
                        style={
                            "flex": "2"
                        }
                    ),
                    html.Div(
                        children=[
                            html.Div(
                                children=[
                                    html.Label('Change time interval: '),
                                    dcc.Dropdown(
                                        options=[
                                            {'label': '30 sec', 'value': 30},
                                            {'label': '5 min', 'value': 300},
                                            {'label': '10 min', 'value': 600},
                                        ],
                                        value=30,
                                        clearable=False,
                                        style={
                                            "width": "100px",
                                            "textAlign": "center"
                                        },
                                        id="menu-interval"
                                    )
                                ],
                                style={
                                    "display": "flex",
                                    "alignItems": "center"
                                }
                            ),
                            html.Div(
                                children=[
                                    html.Label('Show data from: '),
                                    dcc.Input(
                                        placeholder="hh:mm:ss",
                                        type="text",
                                        n_submit=0,
                                        pattern=u"^[0-2][0-3]:[0-5][0-9]:[0-5][0-9]$",
                                        style={
                                            "width": "100px",
                                            "textAlign": "center"
                                        },
                                        id="start-interval"
                                    ),
                                ],
                                style={
                                    "display": "flex",
                                    "alignItems": "center"
                                }
                            )
                        ],
                        style={
                            "flex": "1",
                            "display": "flex",
                            "flexDirection": "column",
                            "alignItems": "end",
                            "justifyContent": "space-around"
                        }
                    )
                ],
                style={
                    "display": 'flex',
                    "marginBottom": "10px"
                }
            ),
            html.Div(
                children=[
                    dcc.Upload(
                        children=[
                            'Drag and Drop or ',
                            html.A(
                                'Select a File',
                                style={
                                    'color': 'blue',
                                    'cursor': 'pointer'
                                }
                            )
                        ], 
                        style={
                            'height': '60px',
                            'lineHeight': '60px',
                            'borderWidth': '1px',
                            'borderStyle': 'dashed',
                            'borderRadius': '5px',
                            'textAlign': 'center'
                        },
                        id='upload-data'
                    ),
                    html.Span(
                        '',
                        style={
                            'textAlign': 'center',
                            'color': 'red'
                        },
                        id='upload-status'
                    )
                ],
                style={
                    'width': '100%',
                    'marginBottom': '10px',
                    'display': 'flex',
                    'flexDirection': 'column'
                }
            ),

            html.Div(
                children=[
                    html.Button(
                        children=f'First {chunk_in_time_text}', 
                        id='first-btn', 
                        n_clicks=0,
                        disabled=True, 
                        className='button'
                    ),
                    
                    html.Button(
                        children=f'< Previous {chunk_in_time_text}', 
                        id='prev-btn', 
                        n_clicks=0,
                        disabled=True, 
                        className='button'
                    ),

                    dcc.Loading(
                        type="default",
                        children=[
                            html.Div(id="loading-output"),
                            html.Div(id="loading-output1")
                        ]
                    ),

                    html.Button(
                        children=f'Next {chunk_in_time_text} >', 
                        id='next-btn', 
                        n_clicks=0,
                        disabled=True, 
                        className='button'
                    ),

                    html.Button(
                        children=f'Last {chunk_in_time_text}', 
                        id='last-btn', 
                        n_clicks=0,
                        disabled=True, 
                        className='button'
                    ),
                ], 
                style={
                    'display': 'flex', 
                    'alignItems': 'center', 
                    'justifyContent': 'space-between',
                    'marginBottom': '10px'
                },
                id='div-btns'
            ),

            html.Div(
                children=[],
                style={
                    'height': '80%',
                    'display': 'flex', 
                    'alignItems': 'center',
                },
                id='eeg-graph-container'
            ),
    ],
    style={
        'display': 'flex',
        'flexDirection': 'column',
        'marginLeft': '40px',
        'marginRight': '40px',
        'overflow': 'hidden'
    }
)

@app.callback(
        Output('file-info', 'children'),
        Output('upload-status', 'children'),
        Output("loading-output1", "children"),
        Input('upload-data', 'contents'),
        State('upload-data', 'filename'),
        prevent_initial_call=True
)
def upload_data(contents, file_name):
    return parse_contents(contents, file_name) if contents else (f"Shows channel-wise EEG signals in {(edf_info.chunk_in_time_text if edf_info else chunk_in_time_text)} interval", '', '')

@app.callback(
    Output('eeg-graph-container', 'children'),
    Output('prev-btn', 'children'),
    Output('next-btn', 'children'),
    Output('prev-btn', 'disabled'),
    Output('next-btn', 'disabled'),
    Output('first-btn', 'children'),
    Output('last-btn', 'children'),
    Output('first-btn', 'disabled'),
    Output('last-btn', 'disabled'),
    Output("loading-output", "children"),
    Input('prev-btn', 'n_clicks'),
    Input('next-btn', 'n_clicks'),
    Input('first-btn', 'n_clicks'),
    Input('last-btn', 'n_clicks'),
    Input('menu-interval', 'value'),
    Input('start-interval', 'n_submit'),
    State('start-interval', 'value'),
    Input("loading-output1", "children"), 
    prevent_initial_call=True
)
def update_figure(prev_btn, next_btn, first_btn, last_btn, interval, n_submit, interval_start, temp):

    if not edf_info: return '', f'< Previous {chunk_in_time_text}', f'Next {chunk_in_time_text} >', True, True, True, True, ''

    if ctx.triggered_id == "prev-btn": edf_info.decChunk()
    elif ctx.triggered_id == "next-btn": edf_info.incChunk()
    elif ctx.triggered_id == "first-btn": edf_info.firstChunk()
    elif ctx.triggered_id == "last-btn": edf_info.lastChunk()
    
    if interval: edf_info.setChunkSize(interval)
    if ctx.triggered_id == "start-interval" and interval_start: edf_info.setChunkStartByTime(interval_start)
    
    return *processLayoutUpdate(), ''


if __name__ == '__main__':
    app.run_server(debug=True)