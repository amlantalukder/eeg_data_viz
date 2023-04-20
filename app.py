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
from datetime import timedelta
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
        self.eeg_start_time =  pd.to_datetime(self.eeg_data.info['meas_date'])
        self.eeg_sampling_freq = float(self.eeg_data.info['sfreq'])
        self.df = pd.DataFrame(self.eeg_data_raw.T, columns=self.eeg_channels)
        self.df = self.df.reset_index()
        self.df = self.df.rename(columns={'index': 'Time'})
        self.epoch_index = 1
        self.window_time = 30

        self.setWindowTime(30)

    def setWindowTime(self, interval):
        
        if self.window_time != interval: self.epoch_index = 1

        self.window_time = interval
        
        if interval == 300: self.window_time_text = '5 min'
        elif interval == 600: self.window_time_text = '10 min'
        else:
            self.window_time_text = f'{interval} sec'

        self.window = self.window_time * self.eeg_sampling_freq
        self.num_epochs = self.df.shape[0]//self.window + int(self.df.shape[0]%self.window > 0)

def processLayoutUpdate():

    df_current = edf_info.df.loc[int(edf_info.window * (edf_info.epoch_index - 1)):int(edf_info.window * edf_info.epoch_index)].copy()
    df_current['Time'] = df_current['Time'].apply(lambda x: edf_info.eeg_start_time + timedelta(seconds=x/edf_info.eeg_sampling_freq))
    df_current = df_current.melt(['Time']).rename(columns={'variable': 'Channel', 'value':'Amp'})
    is_first_epoch, is_last_epoch = edf_info.epoch_index == 1, edf_info.epoch_index == edf_info.num_epochs-1

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

    return dcc.Graph(
        figure=fig,
        style={
            'width': '100%',
        }
    ), f'< Previous {edf_info.window_time_text}', f'Next {edf_info.window_time_text} >', is_first_epoch, is_last_epoch

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
        print(e)
        return (f"Shows channel-wise EEG signals in {window_time_text} interval", "There was an error processing this file", '')
    return ([html.Span("Shows channel-wise EEG signals for "), html.Strong(edf_info.file_name), html.Span(f" in {edf_info.window_time_text} interval")], '', '')
    
app = Dash(__name__)
edf_info, window_time_text = None, "30 sec"

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
                                children=f'Shows EEG signals for selected channels in {window_time_text} interval',
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
                            )
                        ],
                        style={
                            "flex": "1",
                            "display": "flex",
                            "alignItems": "end",
                            "justifyContent": "end"
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
                            'width': '99%',
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
                        children=f'< Previous {window_time_text}', 
                        id='prev-btn', 
                        n_clicks=0,
                        disabled=True, 
                        style={
                            'height': '50px',
                            'cursor': 'pointer'
                        }
                    ),

                    dcc.Loading(
                        type="default",
                        children=[
                            html.Div(id="loading-output"),
                            html.Div(id="loading-output1")
                        ]
                    ),

                    html.Button(
                        children=f'Next {window_time_text} >', 
                        id='next-btn', 
                        n_clicks=0,
                        disabled=True, 
                        style={
                            'height': '50px',
                            'cursor': 'pointer'
                        }
                    )
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
        State('upload-data', 'filename')
)
def upload_data(contents, file_name):
    return parse_contents(contents, file_name) if contents else (f"Shows channel-wise EEG signals in {(edf_info.window_time_text if edf_info else window_time_text)} interval", '', '')

@app.callback(
    Output('eeg-graph-container', 'children'),
    Output('prev-btn', 'children'),
    Output('next-btn', 'children'),
    Output('prev-btn', 'disabled'),
    Output('next-btn', 'disabled'),
    Output("loading-output", "children"),
    Input('prev-btn', 'n_clicks'),
    Input('next-btn', 'n_clicks'),
    Input('menu-interval', 'value'),
    Input("loading-output1", "children")
)
def update_figure(prev_btn, next_btn, interval, temp):
    if not edf_info: return '', f'< Previous {window_time_text}', f'Next {window_time_text} >', True, True, ''
    if "prev-btn" == ctx.triggered_id and edf_info.epoch_index > 1: edf_info.epoch_index -= 1
    elif "next-btn" == ctx.triggered_id and edf_info.epoch_index < edf_info.num_epochs-1: edf_info.epoch_index += 1
    
    edf_info.setWindowTime(interval)
    
    return *processLayoutUpdate(), ''


if __name__ == '__main__':
    app.run_server(debug=True)