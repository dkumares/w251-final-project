import torch
import io

def encode_weights(model):
    buff = io.BytesIO()
    torch.save(model.state_dict(), buff)
    buff.seek(0)

    # Convert model to string for transmission
    model_str = buff.getvalue()
    return model_str

def average_weights(w):
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg
