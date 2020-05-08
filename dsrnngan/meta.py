import h5py
from tensorflow.keras import backend as K


class Nontrainable(object):
    
    def __init__(self, models):
        if not isinstance(models, list):
            models = [models]
        self.models = models

    def __enter__(self):
        self.trainable_status = [m.trainable for m in self.models]
        for m in self.models:
            m.trainable = False
        return self.models

    def __exit__(self, type, value, traceback):
        for (m,t) in zip(self.models,self.trainable_status):
            m.trainable = t


def save_opt_weights(model, filepath):
    with h5py.File(filepath, 'w') as f:
        # Save optimizer weights.
        symbolic_weights = getattr(model.optimizer, 'weights')
        if symbolic_weights:
            optimizer_weights_group = f.create_group('optimizer_weights')
            weight_values = K.batch_get_value(symbolic_weights)
            weight_names = []
            for i, (w, val) in enumerate(zip(symbolic_weights, 
                weight_values)):
                # Default values of symbolic_weights is /variable for theano
                if K.backend() == 'theano':
                    if hasattr(w, 'name') and w.name != "/variable":
                        name = str(w.name)
                    else:
                        name = 'param_' + str(i)
                else:
                    if hasattr(w, 'name') and w.name:
                        name = str(w.name)
                    else:
                        name = 'param_' + str(i)
                weight_names.append(name.encode('utf8'))
            optimizer_weights_group.attrs['weight_names'] = weight_names
            for name, val in zip(weight_names, weight_values):
                param_dset = optimizer_weights_group.create_dataset(
                    name,
                    val.shape,
                    dtype=val.dtype)
                if not val.shape:
                    # scalar
                    param_dset[()] = val
                else:
                    param_dset[:] = val


def load_opt_weights(model, filepath):
    with h5py.File(filepath, mode='r') as f:        
        optimizer_weights_group = f['optimizer_weights']
        optimizer_weight_names = [n.decode('utf8') for n in
            optimizer_weights_group.attrs['weight_names']]
        optimizer_weight_values = [optimizer_weights_group[n] for n in
            optimizer_weight_names]
        model.optimizer.set_weights(optimizer_weight_values)


def ensure_list(x):
    if type(x) != list:
        x = [x]
    return x


def input_shapes(model, prefix):
    shapes = [il.shape[1:] for il in 
        model.inputs if il.name.startswith(prefix)]
    shapes = [tuple([d for d in dims]) for dims in shapes]
    return shapes
