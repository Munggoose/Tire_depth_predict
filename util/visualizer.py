import wandb
from util.box_ops import rescale_bboxes
import matplotlib.pyplot as plt
import torchvision.transforms as T
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import numpy as np

class Visualizer:
    
    def __init__(self,model,args=None):
        wandb.init(project="tire", entity="munpany",reinit=True)
        wandb.run.name = 'DETR'
        if args:
            wandb.config.update(args)
        wandb.watch(model)
        
    
    def visual_attention_weight(self, model,img):
        # use lists to store the outputs via up-values
        model.eval()
        conv_features, enc_attn_weights, dec_attn_weights = [], [], []

        hooks = [
            model.backbone[-2].register_forward_hook(
                lambda self, input, output: conv_features.append(output)
            ),
            model.transformer.encoder.layers[-1].self_attn.register_forward_hook(
                lambda self, input, output: enc_attn_weights.append(output[1])
            ),
            model.transformer.decoder.layers[-1].multihead_attn.register_forward_hook(
                lambda self, input, output: dec_attn_weights.append(output[1])
            ),
        ]

        
        # propagate through the model
        outputs = model(img)

        for hook in hooks:
            hook.remove()

        # don't need the list anymore
        conv_features = conv_features[0]
        enc_attn_weights = enc_attn_weights[0]
        dec_attn_weights = dec_attn_weights[0]
        # get the HxW shape of the feature maps of the CNN
        f_map = conv_features['0']
        shape = f_map.tensors.shape[-2:]
        # and reshape the self-attention to a more interpretable shape
        sattn = enc_attn_weights[0].reshape(shape + shape)
        print("Reshaped self-attention:", sattn.shape)
    
        h, w = conv_features['0'].tensors.shape[-2:]

        probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
        keep = probas.max(-1).values > 0.9

        # downsampling factor for the CNN, is 32 for DETR and 16 for DETR DC5
        fact = 32
        # let's select 4 reference points for visualization
        idxs = [(200, 200), (280, 400), (200, 600), (440, 800),]

        # here we create the canvas
        fig = plt.figure(constrained_layout=True, figsize=(25 * 0.7, 8.5 * 0.7))
        canvas = FigureCanvas(fig)
        # and we add one plot per reference point
        gs = fig.add_gridspec(2, 4)
        axs = [
            fig.add_subplot(gs[0, 0]),
            fig.add_subplot(gs[1, 0]),
            fig.add_subplot(gs[0, -1]),
            fig.add_subplot(gs[1, -1]),
        ]

        # for each one of the reference points, let's plot the self-attention
        # for that point
        for idx_o, ax in zip(idxs, axs):
            idx = (idx_o[0] // fact, idx_o[1] // fact)
            ax.imshow(sattn[..., idx[0], idx[1]], cmap='cividis', interpolation='nearest')
            ax.axis('off')
            ax.set_title(f'self-attention{idx_o}')

        # and now let's add the central image, with the reference points as red circles
        fcenter_ax = fig.add_subplot(gs[:, 1:-1])
        im = T.ToPILImage(img)
        fcenter_ax.imshow(im)
        for (y, x) in idxs:
            scale = im.height / img.shape[-2]
            x = ((x // fact) + 0.5) * fact
            y = ((y // fact) + 0.5) * fact
            fcenter_ax.add_patch(plt.Circle((x * scale, y * scale), fact // 2, color='r'))
            fcenter_ax.axis('off')
        canvas.draw()
        image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
        plt.imshow(image)
        plt.show()
        
        


        # fig, axs = plt.subplots(ncols=len(bboxes_scaled), nrows=2, figsize=(22, 7))

        # for idx, ax_i, (xmin, ymin, xmax, ymax) in zip(keep.nonzero(), axs.T, bboxes_scaled):
        #     ax = ax_i[0]
        #     ax.imshow(dec_attn_weights[0, idx].view(h, w))
        #     ax.axis('off')
        #     ax.set_title(f'query id: {idx.item()}')
        #     ax = ax_i[1]
        #     ax.imshow(im)
        #     ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
        #                             fill=False, color='blue', linewidth=3))
        #     ax.axis('off')
        #     ax.set_title(CLASSES[probas[idx].argmax()])
        # fig.tight_layout()
        model.train()
    
    def visual_log(self,log_dict):
       
        wandb.log(log_dict)