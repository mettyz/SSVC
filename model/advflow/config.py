#####################
# Which experiment: #
#####################

# Dataset to train
dataset = 'ImageNet'
mode    = ''

#########
# Data: #
#########
img_dims   = (3, 64, 64)
output_dim      = img_dims[0] * img_dims[1] * img_dims[2]
add_image_noise = 0.02

#################
# Architecture: #
#################

# Flow-based model architecture
high_res_blocks = 4     # Number of high-resolution, convolutional blocks
low_res_blocks  = 6     # Number of low-resolution, convolutional blocks
channels_hidden = 128   # Number of hidden channels for the convolutional blocks
batch_norm      = False # Batch normalization?

n_blocks        = 6     # Number of fully-connected blocks
internal_width  = 128   # Internal width of the FC blocks
fc_dropout      = 0.0   # Dropout for FC blocks

clamping        = 1.5   # Clamping parameter for avoiding exploding exponentiation

num_classes     = 1000  # Number of classes
org_size        = 299   # Image-size to re-shape the ImageNet data

####################
# Logging/preview: #
####################

loss_names           = ['L', 'L_rev']
preview_upscale      = 3                    # Scale up the images for preview
sampling_temperature = 1.0                  # Sample at a reduced temperature for the preview
progress_bar         = True                 # Show a progress bar of each epoch


train_from_scratch = False
init_scale = 0.03
pre_low_lr = 1
latent_noise = 0.1