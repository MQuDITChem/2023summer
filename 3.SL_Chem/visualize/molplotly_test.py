# code reference: https://www.valencekjell.com/posts/2022-08-13-interactive/
#                 https://github.com/wjm41/molplotly

"""
use plotly & molplotly & dash to visualize molecules 
"""

import pandas as pd
import plotly.express as px

import molplotly

# load a DataFrame with smiles
df_esol = pd.read_csv(
    'https://raw.githubusercontent.com/deepchem/deepchem/master/datasets/delaney-processed.csv')
df_esol = df_esol.reset_index()
df_esol['y_pred'] = df_esol['ESOL predicted log solubility in mols per litre']
df_esol['y_true'] = df_esol['measured log solubility in mols per litre']

# generate a scatter plot
fig = px.scatter(df_esol, x="y_true", y="y_pred", )
fig.update_layout(
    autosize=False,
    width=800,
    height=800,
    template='plotly_white',
)

fig.add_shape(
    type="line", line=dict(dash="dash"), 
    x0=df_esol['y_true'].min(), y0=df_esol['y_true'].min(),
    x1=df_esol['y_true'].max(), y1=df_esol['y_true'].max(),
)

# add molecules to the plotly graph - returns a Dash app
app = molplotly.add_molecules(fig=fig,
                            df=df_esol,
                            smiles_col='smiles',
                            title_col='Compound ID',
                            caption_cols=['index', 'Number of Rings'],
                            )

# run Dash app inline in notebook (or in an external server)
app.run_server(debug=True, mode='inline', port=8700, height=1000)