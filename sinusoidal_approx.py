import numpy as np
from bokeh.plotting import figure, curdoc
from bokeh.models import ColumnDataSource
from models import MLP, Layer, Linear, ReLU, MSE, Tanh

x = np.linspace(-10, 10, 1000)

source = ColumnDataSource(data=dict(x=x, y=np.zeros_like(x)))

p = figure(width=1000, height=500)
p.line(x, 1/x * np.sin(x), line_width=3)
p.line('x', 'y', source=source, line_width=3, line_color="red")


mlp = MLP(1, MSE)
mlp.add(Layer(5, Tanh))
mlp.add(Layer(10, Tanh))
mlp.add(Layer(10, Tanh))
mlp.add(Layer(10, Tanh))
mlp.add(Layer(5, Tanh))
mlp.add(Layer(1, Tanh))

def update():
    x_input = x[:,np.newaxis]

    y_target = (1/x * np.sin(x))[:,np.newaxis]
    mlp.training_step(x_input, y_target, 0.01)
    out = mlp.forward(x.reshape(-1,1))
    source.data = dict(x=x, y=out[:,0])

curdoc().add_root(p)
curdoc().add_periodic_callback(update, 50)