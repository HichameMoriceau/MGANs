-- a script for synthesising images with MGAN
require 'torch'
require 'nn'
require 'optim'
require 'cutorch'
require 'cunn'
require 'image'
loadcaffe_wrap = paths.dofile('lib/loadcaffe_wrapper.lua')
util = paths.dofile('lib/util.lua')
pl = require('pl.import_into')()

local function read_file(path)
    local file = io.open(path, "rb") -- r read mode and b binary mode
    if not file then return nil end
    local content = file:read "*a" -- *a or *all reads the whole file
    file:close()
    return content
end

local opt = {}

local cmd = torch.CmdLine()
cmd:option('-model_name'        ,'../Dataset/Picasso_CelebA100/MGAN/epoch_5_100_netG.t7'  , 'Path/to/release.t7 (released model).')
cmd:option('-input_folder'      ,'~/style_transfer/image_bench/', 'Path/to/testing/images/ (within \'../Dataset/\').')
cmd:option('-output_folder'     ,'~/style_transfer/LiAndWand_results/', 'Path/to/stylized/images (will be automatically generated).')
cmd:option('-input_noise_weight', '0', 'Noise coefficient used when adding noise to the input image.')
cmd:option('-num_noise_FM', '0' , 'Number of noise feature maps to stack before net input.')
cmd:option('-cropped_inputs'    , '0', 'Set to 1 to use inputs of larger size that are then cropped after forwarding through netEnco (inputs size must be +64 larger)')
local params = cmd:parse(arg)

opt.model_name         = params.model_name
opt.input_folder_name  = params.input_folder
opt.output_folder_name = params.output_folder

opt.max_length     = 512 -- change this value for image size. Larger images needs more gpu memory.
opt.stand_atom     = 8 -- make sure the image size can be divided by stand_atom
opt.noise_weight   = params.input_noise_weight -- change this weight for balancing between style and content. Stronger noise makes the synthesis more stylish. 
opt.noise_name     = 'noise.jpg' -- low frequency noise image
opt.gpu            = 1
opt.nnc            = tonumber(params.num_noise_FM)
opt.cropped_inputs = tonumber(params.cropped_inputs)
-- vgg 
opt.vgg_proto_file = '../Dataset/model/VGG_ILSVRC_19_layers_deploy.prototxt' 
opt.vgg_model_file = '../Dataset/model/VGG_ILSVRC_19_layers.caffemodel'
opt.vgg_backend    = 'nn'
opt.vgg_num_layer  = 36

-- encoder that produce vgg feature map of an input image. This feature map is used as the input for netG
opt.netEnco_vgg_Outputlayer = 21
opt.netEnco_vgg_nOutputPlane = 512+opt.nnc -- this value is decided by netEnco_vgg_Outputlayer
opt.netEnco_vgg_Outputblocksize = opt.max_length / 8 -- the denominator s value is decided by netEnco_vgg_Outputlayer


local BlockInterface = torch.Tensor(opt.netEnco_vgg_nOutputPlane, opt.netEnco_vgg_Outputblocksize, opt.netEnco_vgg_Outputblocksize)
BlockInterface = BlockInterface:cuda()

local settings_fname = opt.output_folder_name .. "settings.txt"
print("settings filename " .. settings_fname)

local settings_file = io.open(settings_fname, "w")
settings_file:write("\nmodel name              = " .. opt.model_name)
settings_file:write("\nimage generated in size = " .. opt.max_length)
settings_file:write("\ninput noise weight      = " .. params.input_noise_weight)
settings_file:write("\n")
settings_file:close()

local function round(x)
  if x%2 ~= 0.5 then
    return math.floor(x+0.5)
  end
  return x-0.5
end

local function resize_to_ratio(img_in, maxSize)
  -- find the larger dimension, and resize it to maxSize,
  -- keeps aspect ratio and returns even values for x and y
  local iW = img_in:size(3)
  local iH = img_in:size(2)
  if iW > iH then
    y = round(maxSize * iH / iW)
    if (y%2) ~= 0 then
      y = y-1
    end
    img_out = image.scale(img_in, maxSize, y)
  else
    x = round(maxSize * iW / iH)
    if (x%2) ~= 0 then
      x = x-1
    end
    img_out = image.scale(img_in, x, maxSize)
  end
  return img_out
end

signal = require 'signal'  -- ifft for noise generation

local function meshgrid(vec)
  local uu = torch.repeatTensor(vec, vec:size(1), 1)
  local vv = uu:t():clone()
  return uu, vv
end

local function generate_noise(image_size)
  n    = image_size -- opt.pixel_blockSize   -- image size (must be even)
  beta = -2.5 -- exponent of Power vs. frequency
  u, v = meshgrid( torch.cat( torch.range(0, n/2), torch.range(n/2-1,1,-1)) )
  f = torch.sqrt(torch.pow(u,2):add(torch.pow(v,2)))
  P = torch.pow(f,beta)
  P[1][1] = 0

  local randoms = torch.rand(n,n)
  local exp_2i_pi_rand_n = torch.rand(n,n):mul(2*math.pi):exp()
  local res = P:sqrt():cmul( exp_2i_pi_rand_n )

  zeros = torch.zeros(n,n)
  res   = torch.cat(zeros,res,3) 
  local ifft_res  = signal.ifft2(res)
  local noise_img = ifft_res:select(3,1):mul(n)--:cmul(torch.Tensor(n,n):fill(torch.pow(n,2)))
  noise_img = torch.div( torch.csub(noise_img, torch.min(noise_img)), torch.max(noise_img) - torch.min(noise_img))

  return noise_img:clone()
end

-- returns <fm> with adequate number of concatenated <noise> tensors
-- <fm> must be a KxMxNxN tensor where K is the batch size, M is the nb of
-- feature maps per processed image and N the width and length of a single feature map
local function stack_noise_channels(fm, nnc)
  if nnc > 0 then
    local fm_n  = torch.Tensor(fm:size(1), fm:size(2)+nnc, fm:size(3), fm:size(4))
    -- for each img s feature maps
    for i_img_fm = 1, fm:size(1) do
      -- produce noise
      local noise = image.crop(generate_noise(math.max(fm:size(3), fm:size(4))), 0,0, fm:size(3), fm:size(4))
      local n     = torch.reshape(noise,1, fm:size(3), fm:size(4))
      -- concatenate <nnc> additional noise feature maps
      local n_maps = torch.repeatTensor(n:clone(), nnc,1,1)
      fm_n[i_img_fm] = torch.cat(fm[i_img_fm]:double(), n_maps,1)
    end
    return fm_n
  else 
    return fm
  end
end

local vgg_net = loadcaffe_wrap.load(opt.vgg_proto_file, opt.vgg_model_file, opt.vgg_backend, opt.vggD_num_layer)
-- build encoder
local netEnco = nn.Sequential()
for i_layer = 1, opt.netEnco_vgg_Outputlayer do
  netEnco:add(vgg_net:get(i_layer))
end

print(string.format('netEnco has been built'))
print(netEnco)
netEnco = util.cudnn(netEnco)
netEnco:cuda()

-- load data
os.execute('mkdir ' .. opt.output_folder_name)
local input_images_names = pl.dir.getallfiles(opt.input_folder_name, '*.png')
local num_input_images = #input_images_names
local noise_image = image.load(opt.noise_name, 3)
local net = util.load(opt.model_name, opt.gpu)
net:cuda()
print(net)

print('*****************************************************')
print('Testing: ');
print('*****************************************************') 
local counter = 0
for i_img = 1, num_input_images  do
    -- resize the image image
    local image_input = image.load(input_images_names[i_img], 3)
--[[
    local max_dim = math.max(image_input:size()[2], image_input:size()[3])
    local scale = opt.max_length / max_dim
    local new_dim_x = math.floor((image_input:size()[3] * scale) / opt.stand_atom) * opt.stand_atom
    local new_dim_y = math.floor((image_input:size()[2] * scale) / opt.stand_atom) * opt.stand_atom
    image_input = image.scale(image_input, new_dim_x, new_dim_y, 'bilinear')
]]
    -- add noise to the image (improve flat surfaces quality)
    noise_image = image.scale(noise_image, image_input:size(3), image_input:size(2), 'bilinear') 
    image_input:add(noise_image:mul(opt.noise_weight))
    
    image_input = resize_to_ratio(image_input, opt.max_length)
    image_input:resize(1, image_input:size()[1], image_input:size()[2], image_input:size()[3])

    image_input:mul(2):add(-1)
    image_input = image_input:cuda()
    
    -- decode image with a single forward prop
    local tm = torch.Timer()
    vgg_relu4_1 = netEnco:forward(image_input):clone()

    -- if chosen by CLI, add extra noise feature maps
    vgg_relu4_1 = stack_noise_channels(vgg_relu4_1, opt.nnc):cuda()

    BlockInterface_cropped = vgg_relu4_1:clone()

    if opt.cropped_inputs == 1 then
      local block_h = vgg_relu4_1:size(3)
      local block_w = vgg_relu4_1:size(4)
      local range   = 16+1
      local h_bound = math.ceil((block_h-(math.ceil(image_input:size(3)/8)))/2)+1
      local w_bound = math.ceil((block_w-(math.ceil(image_input:size(4)/8)))/2)+1
      BlockInterface_cropped = vgg_relu4_1[{{}, {}, {h_bound, -h_bound-1}, {w_bound, -w_bound-1}}]
    end

    local image_syn = net:forward(BlockInterface_cropped)
    cutorch.synchronize()
    print(string.format('Image size: %d by %d, time: %f', image_input:size()[3], image_input:size()[4], tm:time().real))
    image_syn = image.toDisplayTensor{input = image_syn, nrow = math.ceil(math.sqrt(image_syn:size(1)))}

    -- save image
    local image_name = input_images_names[i_img]:match("([^/]+)$")
    image_name = string.sub(image_name, 1, string.len(image_name) - 4)
    image.save(opt.output_folder_name .. image_name .. '_MGANs.jpg', image_syn)

    -- clear memory
    image_input = nil
    image_syn = nil
    noise_image_ = nil
    collectgarbage()
end

net = nil
collectgarbage()

