from models.SuperMRI import *
from models.SkiDwithSkipUnet import *
from models.SuperMRI import *
import torchvision
from utility.utils import *


class SaveFeatures():
    features = None
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
    def hook_fn(self, module, input, output):
        self.features = output
    def close(self):
        self.hook.remove()


train_loader, val_loader, test_loader = loadData("data/", 1, test_size=0.5, color='gray', noise=True)

# ------------------------------ Load model ----------------------------- #
model = UNet(use_attention_gate=True, max_blocks=6)
model.load_state_dict(torch.load('saved_models/Unet_3.pth'))
model.eval()

# Evaluate model 
print("The number of parameters in the model: ", sum(p.numel() for p in model.parameters()))
generated_images, original_images = [], []
with torch.no_grad():
    for i, data in enumerate(test_loader):
        img, _ = data
        img = img.to("cpu")
        output = model(img)
        original_images.append(img[0])
        generated_images.append(output[0])
        break


# ------------------------ Attention gate output ------------------------ #
final_conv_layer = model.up_blocks[1].attention_gate.norm
activated_features = SaveFeatures(final_conv_layer)

denoised_img = model(img)

# ------------------------- Calculate gradients ------------------------- #
model.zero_grad()
criterion = nn.MSELoss()
loss = criterion(denoised_img, original_images[0].cpu().squeeze())
loss.backward()
activations = activated_features.features

heatmap = torch.mean(activations, dim=1).squeeze()
heatmap = np.maximum(heatmap.cpu().detach().numpy(), 0)
res_hm = cv2.resize(heatmap, (256, 256))

plt.imshow(original_images[0].cpu().squeeze(), cmap='gray')
plt.imshow(res_hm, alpha=0.2, cmap='jet')
plt.axis('off')
plt.show()