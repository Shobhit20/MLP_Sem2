from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from models.SuperMRI import *
from models.SkiD import *
from models.SkiDwithSkip import *
from models.SkiDwithSkipUnet import *
from models.SuperMRI import *
import torchvision
from utility.utils import *

train_loader, test_loader, train_original, test_original = loadData("data/", 1, test_size=0.2, color='gray', noise=True)

model = UNet(use_attention_gate=True, max_blocks=6)
model.load_state_dict(torch.load('saved_models/model_unet_5_gaussnoise_lre-3.pth'))
model.eval()
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


class SaveFeatures():
    features = None
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
    def hook_fn(self, module, input, output):
        self.features = output
    def close(self):
        self.hook.remove()

print(model.up_blocks)
final_conv_layer = model.up_blocks[2].attention_gate.conv
activated_features = SaveFeatures(final_conv_layer)

denoised_img = model(img)

# Calculate gradients
model.zero_grad()
criterion = nn.MSELoss()
loss = criterion(denoised_img, original_images[0].cpu().squeeze())
loss.backward()
activations = activated_features.features

heatmap = torch.mean(activations, dim=1).squeeze()
heatmap = np.maximum(heatmap.cpu().detach().numpy(), 0)
res_hm = cv2.resize(heatmap, (256, 256))
# # Plot heatmap
# plt.imshow(heatmap)
# plt.axis('off')
# plt.show()
print(type(heatmap))
# Overlay heatmap on original noisy image
plt.imshow(original_images[0].cpu().squeeze(), cmap='gray')
plt.imshow(res_hm, alpha=0.2, cmap='jet')
plt.axis('off')
plt.show()