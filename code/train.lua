-- a script for texture synthesis With Markovian Decovolutional Adversarial Networks (MDAN)
require 'torch'
require 'nn'
require 'optim'
require 'cutorch'
require 'cunn'
require 'image'
require 'signal' -- ifft for 1/f noise generation

disp           = require 'display'
pl             = require('pl.import_into')()
loadcaffe_wrap = paths.dofile('lib/loadcaffe_wrapper.lua')
util           = paths.dofile('lib/util.lua')

paths.dofile('lib/helper.lua')
paths.dofile('lib/tv.lua')
paths.dofile('lib/tv2.lua')
cutorch.setDevice(1)
MDAN_wrapper = require 'MDAN_wrapper'
AG_wrapper   = require 'AG_wrapper'
MGAN_wrapper = require 'MGAN_wrapper'

cutorch.setDevice(1)

-- Run: nohup th train.lua -dataset_name Picasso_CelebA100 &
-- Run: nohup th train.lua -dataset_name ../Dataset/Picasso_CelebA100/ -tv_weight 5e-4  >> nohup.out &

local cmd = torch.CmdLine()
cmd:option('-dataset_name'         , '../Dataset/Delaunay_ImageNet100/', 'Path to image data set')
cmd:option('-train_img_size'       , '128', 'Image size used during training')
cmd:option('-training_noise_weight', '0', 'Degree of noise added to the patches before forwarding them through NetEnco (VGG19 ReLU4_1)')
cmd:option('-cropped_inputs'       , '0', 'Set to 1 to use inputs of larger size that are then cropped after forwarding through netEnco (inputs size must be +64 larger)')
cmd:option('-num_noise_FM'         , '0', 'Number of noise slices to be stacked to the feature maps of each image after VGG conv4_1 (output of netEnco)')
cmd:option('-pixel_loss_OFF'       , '15', 'Turn OFF pixel loss after n epochs')
cmd:option('-style_weight'         , '1e-2', '')
cmd:option('-tv_weight_ON'         , '1', 'Turn ON total variation loss')
cmd:option('-multi_step_ON'        , '15', 'Turn ON multi step mode after 15 epoch. (Output of generator is fed again to vgg then generator as input: 2 forward passes)')
cmd:option('-learning_rate'        , '0.02', 'Initial learning rate for adam (same for generator and discriminator)')
cmd:option('-save_interval_image'  , '50', 'Save visualization of training and generator every 50 images')
cmd:option('-batch_size'           , '64', '')
cmd:option('-tv_weight'            , '1e-4', 'Total variation weight used when computing the generator gradient')
cmd:option('-ulyanov_loss'         , '0', 'Set to 1 to use Dmitry Ulyanov s loss function.')
local params = cmd:parse(arg)

-- general parameters
dataset_name            = params.dataset_name
stand_imageSize_syn     = 512 -- the largest dimension for the synthesized image
stand_imageSize_example = 512 -- the largest dimension for the style image
stand_atom              = 8 -- make sure the image size can be divided by stand_atom
train_imageSize         = params.train_img_size -- the actual size for training images. 
flag_MDAN               = false
flag_AG                 = false
flag_MGAN               = true

-- MDAN parameters
MDAN_numEpoch        = 5 -- number of epoch 
MDAN_numIterPerEpoch = 25 -- number of iterations per epoch
MDAN_contrast_std    = 2 -- cut off value for controling contrast in the output image. Anyvalue that is outside of (-std_im_out * contrast_std, std_im_out * contrast_std) will be set to -std_im_out * contrast_std or std_im_out * contrast_std
local MDAN_experiments = {
    -- MDAN_experiments[1]: input folder for content images 
    -- MDAN_experiments[2]: output folder for stylized images
    -- MDAN_experiments[3]: flag to use pre-trained discriminative network. In practice we need time to build up the discriminator, so it is better to re-use the dicriminator that has already been trained for a while.
    {'ContentInitial/', 'StyleInitial/', false}, -- run with randomly initialized discriminator
    {'ContentTrain/', 'StyleTrain/', true}, -- run with previously saved discriminator
    {'ContentTest/', 'StyleTest/', true}, -- run with previously saved discriminator
}

-- Augment parameters
AG_sampleStep    = 64 --train_imageSize-1 -- the sampling step for training images
AG_step_rotation = math.pi/18 -- the step for rotation
AG_step_scale    = 1.1 -- the step for scaling
AG_num_rotation  = 0 -- number of rotations
AG_num_scale     = 0 -- number of scalings
AG_flag_flip     = true -- flip image or not

-- MGAN parameters
MGAN_netS_weight = params.style_weight  --higher weight for discriminator gives sharper texture, but might deviate from image content
local MGAN_experiments = {
    -- MGAN_experiments[1]: starting epoch. Use value larger than one to load a previously saved model (start_epoch - 1)
    -- MGAN_experiments[2]: ending epoch.
    {1, 15}, -- learn five epochs
}


---*********************************************************************************************************************
-- DO NOT CHANGE AFTER THIS LINE
---*********************************************************************************************************************

metrics = io.open("metrics.txt", "w")

local start_MDAN = os.time()

------------------------------------------------------------------------------------------------------------------------
-- RUN MDAN
------------------------------------------------------------------------------------------------------------------------
if flag_MDAN then
    for i_test = 1, #MDAN_experiments do
        local MDAN_params = {}
        MDAN_params.dataset_name            = dataset_name
        MDAN_params.stand_imageSize_syn     = stand_imageSize_syn
        MDAN_params.stand_imageSize_example = stand_imageSize_example
        MDAN_params.stand_atom              = stand_atom
        MDAN_params.input_content_folder    = MDAN_experiments[i_test][1] 
        MDAN_params.output_style_folder     = MDAN_experiments[i_test][2]
        MDAN_params.output_model_folder     = 'MDAN/'
        MDAN_params.numEpoch                = MDAN_numEpoch
        MDAN_params.numIterPerEpoch         = MDAN_numIterPerEpoch
        MDAN_params.contrast_std            = MDAN_contrast_std
        MDAN_params.flag_pretrained         = MDAN_experiments[i_test][3]
        local MDAN_state                    = MDAN_wrapper.state(MDAN_params)
        collectgarbage()
    end
end

local end_MDAN=os.time()
local MDAN_duration=end_MDAN-start_MDAN


metrics:write("MDAN duration: " .. MDAN_duration/60 .. " (training not finished)\n")


local start_AG = os.time()
-- ------------------------------------------------------------------------------------------------------------------------
-- -- RUN Data Augmentation
-- ------------------------------------------------------------------------------------------------------------------------
if flag_AG then
    local AG_params = {}
    AG_params.dataset_name        = dataset_name
    AG_params.stand_imageSize_syn = stand_imageSize_syn
    AG_params.stand_atom          = stand_atom
    AG_params.AG_imageSize        = train_imageSize
    AG_params.AG_sampleStep       = AG_sampleStep 
    AG_params.AG_step_rotation    = AG_step_rotation 
    AG_params.AG_step_scale       = AG_step_scale 
    AG_params.AG_num_rotation     = AG_num_rotation 
    AG_params.AG_num_scale        = AG_num_scale 
    AG_params.AG_flag_flip        = AG_flag_flip 
    local AG_state                = AG_wrapper.state(AG_params)
    collectgarbage()
end

local end_AG=os.time()
local AG_duration=end_AG-start_AG
metrics:write("AG   duration: " .. AG_duration/60 .. " (training not finished)\n")


local style_img_name=pl.dir.getallfiles(dataset_name .. "Style/", '*.png')[1]
print("style_img" .. style_img_name)

-- For compatibility with Dmitry Ulyanov code
local ulyanov_params = {}
ulyanov_params.content_layers = 'relu4_2'
ulyanov_params.style_layers = 'relu1_1,relu2_1,relu3_1,relu4_1'
ulyanov_params.learning_rate = 1e-3
ulyanov_params.num_iterations = 50000
ulyanov_params.save_every = 1000
ulyanov_params.batch_size = 1
ulyanov_params.image_size = 256
ulyanov_params.content_weight = 1
ulyanov_params.style_weight = 1
ulyanov_params.tv_weight = 0
ulyanov_params.style_image = style_img_name
ulyanov_params.style_size = 256
ulyanov_params.mode = 'style'
ulyanov_params.checkpoints_path = 'data/checkpoints/'
ulyanov_params.model = 'pyramid'
ulyanov_params.normalize_gradients = 'false'
ulyanov_params.vgg_no_pad = 'false'
ulyanov_params.normalization = 'instance'
ulyanov_params.proto_file = 'data/pretrained/VGG_ILSVRC_19_layers_deploy.prototxt'
ulyanov_params.model_file = 'data/pretrained/VGG_ILSVRC_19_layers.caffemodel'
ulyanov_params.backend = 'cudnn'

-- For compatibility with Justin Johnsons code
ulyanov_params.texture_weight = ulyanov_params.style_weight
ulyanov_params.texture_layers = ulyanov_params.style_layers
ulyanov_params.texture = ulyanov_params.style_image

ulyanov_params.style_weight = params.style_weight
ulyanov_params.image_size = train_imageSize

local start_MGAN = os.time()
-- ------------------------------------------------------------------------------------------------------------------------
-- -- RUN MGAN
-- ------------------------------------------------------------------------------------------------------------------------
if flag_MGAN then
    for i_test = 1, #MGAN_experiments do
        local MGAN_params = {}
        MGAN_params.dataset_name          = dataset_name
        MGAN_params.start_epoch           = MGAN_experiments[i_test][1]
        MGAN_params.numEpoch              = MGAN_experiments[i_test][2]
        MGAN_params.stand_atom            = stand_atom
        MGAN_params.pixel_blockSize       = train_imageSize
        MGAN_params.netS_weight           = MGAN_netS_weight

	MGAN_params.training_noise_weight = params.training_noise_weight
	MGAN_params.cropped_inputs        = tonumber(params.cropped_inputs)
        MGAN_params.num_noise_FM          = params.num_noise_FM
        MGAN_params.pixel_loss_OFF        = params.pixel_loss_OFF
        MGAN_params.tv_weight_ON          = params.tv_weight_ON
        MGAN_params.multi_step_ON         = params.multi_step_ON
        MGAN_params.save_interval_image   = params.save_interval_image
        MGAN_params.learning_rate         = params.learning_rate
        MGAN_params.batch_size            = params.batch_size
        MGAN_params.tv_weight             = params.tv_weight
        MGAN_params.ulyanov_loss          = params.ulyanov_loss
        print("ulyanov_params.texture = " .. ulyanov_params.texture)
        print("MGAN_params.pixel_blockSize = " .. MGAN_params.pixel_blockSize)
        local MGAN_state = MGAN_wrapper.state(MGAN_params, ulyanov_params)
        collectgarbage()    
    end
end

local end_MGAN=os.time()
local MGAN_duration=end_MGAN-start_MGAN

metrics:write("MGAN duration: " .. MGAN_duration/60 .. " (training not finished)\n")
metrics:write("Training finished")
metrics:close()

do return end

