-- a script for texture synthesis With Markovian Generative Adversarial Networks

signal = require 'signal'

require 'src/utils'
require 'src/descriptor_net'

function meshgrid(vec)
  local uu = torch.repeatTensor(vec, vec:size(1), 1)
  --local vv = torch.repeatTensor(vec:view(-1,1), 1, vec:size(1))
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
  res = torch.cat(zeros,res,3) 
  local ifft_res = signal.ifft2(res)
  local noise_img = ifft_res:select(3,1):mul(n)--:cmul(torch.Tensor(n,n):fill(torch.pow(n,2)))
  noise_img = torch.div( torch.csub(noise_img, torch.min(noise_img)), torch.max(noise_img) - torch.min(noise_img))

  return noise_img:clone()
end

local function run_MGAN(params, ulyanov_params)
  local flag_state = 1
  local opt = {}
  opt.dataset_name          = params.dataset_name
  opt.start_epoch           = params.start_epoch
  opt.numEpoch              = params.numEpoch
  opt.stand_atom            = params.stand_atom
  opt.pixel_blockSize       = params.pixel_blockSize
  opt.training_noise_weight = params.training_noise_weight
  opt.num_noise_FM          = tonumber(params.num_noise_FM)
  opt.multi_step_ON         = tonumber(params.multi_step_ON)
  opt.cropped_inputs        = tonumber(params.cropped_inputs)

  print("pixel_blockSize_source = " .. params.pixel_blockSize)
  pixel_blockSize_source = tonumber(params.pixel_blockSize)
  pixel_blockSize_target = tonumber(params.pixel_blockSize)
  
  print("ulyanov_params.texture = " .. ulyanov_params.texture)
  opt.ulyanov_loss = params.ulyanov_loss
  local crit       = nn.ArtisticCriterion(ulyanov_params)

  if opt.cropped_inputs == 1 then
    pixel_blockSize_source = pixel_blockSize_source+68
    pixel_blockSize_target = pixel_blockSize_target+68
  end

  -- data
  opt.source_folder_name     = 'ContentTrainPatch' .. pixel_blockSize_source
  opt.testsource_folder_name = 'ContentTestPatch'  .. pixel_blockSize_source
  opt.target_folder_name     = 'StyleTrainPatch' .. pixel_blockSize_target
  opt.testtarget_folder_name = 'StyleTestPatch'  .. pixel_blockSize_target
  opt.experiment_name        = 'MGAN/'

  opt.nc            = 3    -- number of channels for color image (fixed to 3)
  opt.nf            = 64   -- multiplier for number of feaures at each layer
  opt.pixel_weights = 1    -- higher weight preserves image content, but blurs the results.
  opt.tv_weight     = tonumber(params.tv_weight) -- higher weight reduces optical noise, but may over-smooth the results.

  -- encoder that produce vgg feature map of an input image. This feature map is used as the input for netG
  opt.netEnco_vgg_Outputlayer     = 21
  opt.netEnco_vgg_nOutputPlane    = 512 -- this value is decided by netEnco_vgg_Outputlayer
  opt.netEnco_vgg_Outputblocksize = opt.pixel_blockSize / 8 -- the denominator s value is decided by netEnco_vgg_Outputlayer

  -- discriminator
  opt.netS_num              = 1
  opt.netS_weights          = {params.netS_weight}
  opt.netS_vgg_Outputlayer  = {12}
  opt.netS_vgg_nOutputPlane = {256} -- this value is decided by netS_vgg_Outputlayer
  opt.netS_blocksize        = {(pixel_blockSize_target) / 16} -- the denominator s is decided by the design of netS
  if opt.cropped_inputs == 1 then
    opt.netS_blocksize      = {(pixel_blockSize_target-68) / 16}
  end
  opt.netS_flag_mask        = 1 -- flag to mask out patches at the border. This reduce the artefacts of padding.
  opt.netS_border           = 1 -- margin to be mask. So only patches between (netS_border + 1, netS_blocksize - netS_border) will be used for back propogation.

  -- optimization
  opt.batchSize  = tonumber(params.batch_size) -- number of patches in a batch
  opt.optimizer  = 'adam'
  opt.netD_lr    = tonumber(params.learning_rate) -- netD initial learning rate for adam
  opt.netG_lr    = tonumber(params.learning_rate) -- netG initial learning rate for adam
  opt.netD_beta1 = 0.5 -- netD first momentum of adam
  opt.netG_beta1 = 0.5 -- netG first momentum of adam
  opt.real_label = 1 -- value of real label (fixed to 1)
  opt.fake_label = -1 -- value of fake label (fixed to -1)

  -- vgg 
  opt.vgg_proto_file = '../Dataset/model/VGG_ILSVRC_19_layers_deploy.prototxt' 
  opt.vgg_model_file = '../Dataset/model/VGG_ILSVRC_19_layers.caffemodel'
  opt.vgg_backend    = 'nn'
  opt.vgg_num_layer  = 36

  -- misc
  opt.save_interval_image = params.save_interval_image   -- save iterval for image
  opt.display             = 1 -- display samples while training. 0 = false
  opt.gpu                 = 1
  cutorch.setDevice(opt.gpu)

  local weight_sum = 0
  weight_sum = weight_sum + opt.pixel_weights
  for i_netS = 1, opt.netS_num do
    weight_sum = weight_sum + opt.netS_weights[i_netS]
  end

  opt.pixel_weights = opt.pixel_weights / weight_sum
  for i_netS = 1, opt.netS_num do
    opt.netS_weights[i_netS] = opt.netS_weights[i_netS] / weight_sum
  end

  os.execute('mkdir ' .. opt.dataset_name .. opt.experiment_name)
  opt.target_folder     = opt.dataset_name .. opt.target_folder_name
  opt.source_folder     = opt.dataset_name .. opt.source_folder_name
  opt.testsource_folder = opt.dataset_name .. opt.testsource_folder_name
  opt.testtarget_folder = opt.dataset_name .. opt.testtarget_folder_name
  opt.manualSeed        = torch.random(1, 10000) -- fix seed
  torch.manualSeed(opt.manualSeed)

  --------------------------------
  -- build networks
  --------------------------------
  -- build netVGG
  local net_ = loadcaffe_wrap.load(opt.vgg_proto_file, opt.vgg_model_file, opt.vgg_backend, opt.vggD_num_layer)
  local netVGG = nn.Sequential()
  for i_layer = 1, opt.vgg_num_layer do
    netVGG:add(net_:get(i_layer))
  end
  net_ = nil
  collectgarbage()
  print('netVGG has been built')
  print(netVGG)
  netVGG = util.cudnn(netVGG)
  netVGG:cuda()

  -- build encoder
  local netEnco = nn.Sequential()
  for i_layer = 1, opt.netEnco_vgg_Outputlayer do
   netEnco:add(netVGG:get(i_layer))
  end  
  print(string.format('netEnco has been built'))
  print(netEnco)   
  netEnco = util.cudnn(netEnco)
  netEnco:cuda()

  -- build generator
  local netG = nn.Sequential()
  if opt.start_epoch > 1 then
    netG = util.load(opt.dataset_name .. opt.experiment_name .. 'epoch_' .. opt.start_epoch - 1 .. '_netG.t7', opt.gpu)
    print(string.format('netG has been loaded'))
    print(netG)
  else
    print("building netG")
    -- layer 21
--[[

    netG:add(nn.SpatialFullConvolution(opt.netEnco_vgg_nOutputPlane+opt.num_noise_FM, opt.nf * 8, 3, 3, 1, 1, 1, 1)) -- x 1
    netG:add(nn.SpatialBatchNormalization(opt.nf * 8)):add(nn.ReLU(true))

    netG:add(nn.SpatialFullConvolution(opt.nf * 8, opt.nf * 4, 4, 4, 2, 2, 1, 1)) -- x 2
    netG:add(nn.SpatialBatchNormalization(opt.nf * 4)):add(nn.ReLU(true))

    netG:add(nn.SpatialFullConvolution(opt.nf * 4, opt.nf * 2, 4, 4, 2, 2, 1, 1)) -- x 4
    netG:add(nn.SpatialBatchNormalization(opt.nf * 2)):add(nn.ReLU(true))

    netG:add(nn.SpatialFullConvolution(opt.nf * 2, opt.nc, 4, 4, 2, 2, 1, 1)) -- x 8

    netG:add(nn.SpatialFullConvolution(opt.nc, opt.nc, 4, 4, 2, 2, 1, 1)) -- x 16

    netG:add(nn.SpatialConvolution(opt.nc, opt.nc, 3, 3, 2, 2, 1, 1)) -- / 2
    netG:add(nn.Tanh())

]]

--[[
    netG:add(nn.SpatialFullConvolution(opt.netEnco_vgg_nOutputPlane+opt.num_noise_FM, opt.nf * 8, 3, 3)) -- x 1
    netG:add(nn.SpatialBatchNormalization(opt.nf * 8)):add(nn.ReLU(true))

    netG:add(nn.SpatialFullConvolution(opt.nf * 8, opt.nf * 6, 3, 3, 2, 2))
    netG:add(nn.SpatialBatchNormalization(opt.nf * 6)):add(nn.ReLU(true))

    netG:add(nn.SpatialFullConvolution(opt.nf * 6, opt.nf * 4, 3, 3, 2, 2))
    netG:add(nn.SpatialBatchNormalization(opt.nf * 4)):add(nn.ReLU(true))

    netG:add(nn.SpatialFullConvolution(opt.nf * 4, opt.nf * 2, 2, 2, 2, 2))
    netG:add(nn.SpatialBatchNormalization(opt.nf * 2)):add(nn.ReLU(true))

    netG:add(nn.SpatialConvolution(opt.nf * 2, opt.nc, 4, 4))
    netG:add(nn.SpatialBatchNormalization(opt.nc)):add(nn.ReLU(true))

    netG:add(nn.SpatialConvolution(opt.nc, opt.nc, 5, 5))
    netG:add(nn.SpatialBatchNormalization(opt.nf * 2)):add(nn.ReLU(true))

    netG:add(nn.SpatialConvolution(opt.nc, opt.nc, 5, 5))
    netG:add(nn.SpatialBatchNormalization(opt.nf * 2)):add(nn.ReLU(true))

    netG:add(nn.SpatialConvolution(opt.nc, opt.nc, 5, 5))
    netG:add(nn.SpatialBatchNormalization(opt.nf * 2)):add(nn.ReLU(true))

    netG:add(nn.SpatialConvolution(opt.nc, opt.nc, 5, 5))
    netG:add(nn.SpatialBatchNormalization(opt.nf * 2)):add(nn.ReLU(true))

    netG:add(nn.SpatialConvolution(opt.nc, opt.nc, 4, 4))
    netG:add(nn.SpatialBatchNormalization(opt.nf * 2)):add(nn.ReLU(true))
    netG:add(nn.Tanh())
    netG:apply(weights_init)
]]

    -- layer 21
    netG:add(nn.SpatialFullConvolution(opt.netEnco_vgg_nOutputPlane, opt.nf * 8, 3, 3, 1, 1, 1, 1)) -- x 1
    netG:add(nn.SpatialBatchNormalization(opt.nf * 8)):add(nn.ReLU(true))
    netG:add(nn.SpatialFullConvolution(opt.nf * 8, opt.nf * 4, 4, 4, 2, 2, 1, 1)) -- x 2
    netG:add(nn.SpatialBatchNormalization(opt.nf * 4)):add(nn.ReLU(true))
    netG:add(nn.SpatialFullConvolution(opt.nf * 4, opt.nf * 2, 4, 4, 2, 2, 1, 1)) -- x 4
    netG:add(nn.SpatialBatchNormalization(opt.nf * 2)):add(nn.ReLU(true))
    netG:add(nn.SpatialFullConvolution(opt.nf * 2, opt.nc, 4, 4, 2, 2, 1, 1)) -- x 8
    netG:add(nn.Tanh())
    netG:apply(weights_init)

    print(string.format('netG has been built'))
    print(netG)  
  end
 
  netG = util.cudnn(netG)
  netG:cuda()

  -- build discriminator
  local netSVGG = {}
  for i_netS = 1, opt.netS_num do 
    table.insert(netSVGG, nn.Sequential())
    netSVGG[i_netS]:add(nn.TVLoss2(opt.tv_weight))
    for i_layer = 1, opt.netS_vgg_Outputlayer[i_netS] do
      netSVGG[i_netS]:add(netVGG:get(i_layer))
    end   
    print(string.format('netSVGG[%d] has been built', i_netS))
    print(netSVGG[i_netS])     
  end
  for i_netS = 1, opt.netS_num do 
    netSVGG[i_netS] = util.cudnn(netSVGG[i_netS])
    netSVGG[i_netS]:cuda()  
  end

  local netS = {}
  if opt.start_epoch > 1 then
    for i_netS = 1, opt.netS_num do 
      table.insert(netS, util.load(opt.dataset_name .. opt.experiment_name .. 'epoch_' .. opt.start_epoch - 1 .. '_netS_' .. i_netS .. '.t7', opt.gpu))
      print(string.format('netS[%d] has been loaded', i_netS))
      print(netS[i_netS]) 
    end
  else
    for i_netS = 1, opt.netS_num do
      table.insert(netS, nn.Sequential())
      netS[i_netS]:add(nn.LeakyReLU(0.2, true))
      netS[i_netS]:add(nn.SpatialConvolution(opt.netS_vgg_nOutputPlane[i_netS], opt.nf * 4, 4, 4, 2, 2, 1, 1)) -- x 1/2
      netS[i_netS]:add(nn.SpatialBatchNormalization(opt.nf * 4)):add(nn.LeakyReLU(0.2, true))
      netS[i_netS]:add(nn.SpatialConvolution(opt.nf * 4, opt.nf * 8, 4, 4, 2, 2, 1, 1)) -- x 1/4
      netS[i_netS]:add(nn.SpatialBatchNormalization(opt.nf * 8)):add(nn.LeakyReLU(0.2, true))     
      netS[i_netS]:add(nn.SpatialConvolution(opt.nf * 8, 1, 1, 1)) -- classify each neural patch using convolutional operation
      netS[i_netS]:add(nn.Reshape(opt.batchSize * opt.netS_blocksize[i_netS] * opt.netS_blocksize[i_netS], 1, 1, 1,false))--reshape the classification result for computing loss
      netS[i_netS]:add(nn.View(1):setNumInputDims(3))
      netS[i_netS]:apply(weights_init)
      print(string.format('netS[%d] has been built', i_netS))
      print(netS[i_netS]) 
    end
  end
  

  for i_netS = 1, opt.netS_num do 
    netS[i_netS] = util.cudnn(netS[i_netS])
    netS[i_netS]:cuda()  
  end

  local criterion_Pixel = nn.MSECriterion()
  criterion_Pixel:cuda()

  local criterion_netS = {}
  for i_netS = 1, opt.netS_num do
    table.insert(criterion_netS, nn.MarginCriterion(1))
  end
  for i_netS = 1, opt.netS_num do
    criterion_netS[i_netS] = criterion_netS[i_netS]:cuda()
  end

  --------------------------------
  -- build data
  --------------------------------
  print("in " .. opt.source_folder .. ": ")
  local source_images_names = pl.dir.getallfiles(opt.source_folder, '*.png')
  local num_source_images = #source_images_names
  print(string.format('num of source images: %d from %s', num_source_images, opt.source_folder))

  local target_images_names = pl.dir.getallfiles(opt.target_folder, '*.png')
  local num_target_images = #target_images_names
  print(string.format('num of target images: %d from %s', num_target_images, opt.target_folder))

  local test_images_names = pl.dir.getallfiles(opt.testsource_folder, '*.png')
  local num_test_images = #test_images_names
  print(string.format('num of test images: %d from %s', num_test_images, opt.testsource_folder))


  local BlockPixel_target = torch.Tensor(opt.batchSize, opt.nc, pixel_blockSize_target, pixel_blockSize_target)
  BlockPixel_target = BlockPixel_target:cuda();

  local BlockPixel_target_crop = torch.Tensor(opt.batchSize, opt.nc, pixel_blockSize_target-68, pixel_blockSize_target-68)
  BlockPixel_target_crop = BlockPixel_target:cuda()

  local BlockPixel_source     = torch.Tensor(opt.batchSize, opt.nc, pixel_blockSize_source, pixel_blockSize_source)
  BlockPixel_source           = BlockPixel_source:cuda()
  local BlockPixel_testsource = torch.Tensor(opt.batchSize, opt.nc, pixel_blockSize_source, pixel_blockSize_source)
  BlockPixel_testsource       = BlockPixel_testsource:cuda()

  local BlockPixel_G = torch.Tensor(opt.batchSize, opt.nc, opt.pixel_blockSize, opt.pixel_blockSize)
  BlockPixel_G       = BlockPixel_G:cuda()

  local BlockPixel_testtarget = torch.Tensor(opt.batchSize, opt.nc, pixel_blockSize_target, pixel_blockSize_target)
  BlockPixel_testtarget       = BlockPixel_testtarget:cuda()

  local BlockPixel_testtarget_crop = torch.Tensor(opt.batchSize, opt.nc, pixel_blockSize_target-68, pixel_blockSize_target-68)
  BlockPixel_testtarget_crop       = BlockPixel_testtarget_crop:cuda()

  local BlockPixel_Gtest = torch.Tensor(opt.batchSize, opt.nc, opt.pixel_blockSize, opt.pixel_blockSize)
  BlockPixel_Gtest       = BlockPixel_Gtest:cuda()

  local BlockInterface = torch.Tensor(opt.batchSize, opt.netEnco_vgg_nOutputPlane + opt.num_noise_FM, opt.netEnco_vgg_Outputblocksize, opt.netEnco_vgg_Outputblocksize)
  BlockInterface       = BlockInterface:cuda()

  local BlockInterface_cropped = BlockInterface:clone()

  local BlockInterfacetest = torch.Tensor(opt.batchSize, opt.netEnco_vgg_nOutputPlane + opt.num_noise_FM, opt.netEnco_vgg_Outputblocksize, opt.netEnco_vgg_Outputblocksize)
  BlockInterfacetest       = BlockInterfacetest:cuda()

  print('blocks declared')

  print("forward through netSVGG")
  local BlockVGG_target = {}
  for i_netS = 1, opt.netS_num do
    local feature_map = netSVGG[i_netS]:forward(BlockPixel_target):clone()
    table.insert(BlockVGG_target, feature_map)
  end

  print("forward through netSVGG")
  local BlockVGG_G = {}
  for i_netS = 1, opt.netS_num do
    local feature_map = netSVGG[i_netS]:forward(BlockPixel_G):clone()
    table.insert(BlockVGG_G, feature_map)
  end

  print("set learning rate")
  local optimStateG = {learningRate = opt.netG_lr, beta1 = opt.netG_beta1,}
  local optimStateS = {}
  for i_netS = 1, opt.netS_num do
    table.insert(optimStateS, {learningRate = opt.netD_lr, beta1 = opt.netD_beta1,})
  end
  optimStateG =  {learningRate = opt.netG_lr, beta1 = opt.netG_beta1,}

  local parametersG, gradparametersG = netG:getParameters()
  local errG = 0
  local errG_Pixel, errG_Style

  local parametersS = {}
  local gradParametersS = {}
  for i_netS = 1, opt.netS_num do
    local parametersS_, gradParametersS_ = netS[i_netS]:getParameters()
    table.insert(parametersS, parametersS_)
    table.insert(gradParametersS, gradParametersS_)
  end

  print("select " .. opt.batchSize .. " img indices at random ")
  local list_image_test = torch.Tensor(opt.batchSize)
  for i_img = 1, opt.batchSize do
    list_image_test[i_img] = torch.random(1, num_test_images)
  end   

  print('loading testsource patches from: ' .. opt.testsource_folder)
 
  for i_img = 1, opt.batchSize do
    local tmp   = image.load(opt.testsource_folder .. '/' .. list_image_test[i_img] .. '.png', 3)
    local noise = generate_noise(pixel_blockSize_source)
    noise = torch.repeatTensor(noise, 3,1,1)
    tmp:add(noise:mul(opt.training_noise_weight))
    BlockPixel_testsource[i_img] = tmp
  end
  BlockPixel_testsource:mul(2):add(-1)
  BlockPixel_testsource = BlockPixel_testsource:cuda()
  
  BlockInterfacetest = netEnco:forward(BlockPixel_testsource):clone()

  local BlockInterfacetest_cropped = BlockInterfacetest:clone()
  if opt.cropped_inputs == 1 then
    local block_h = BlockInterfacetest:size(3)
    local block_w = BlockInterfacetest:size(4)
    local range = 16+1
    local h_bound = math.ceil((block_h-range)/2)+1
    local w_bound = math.ceil((block_w-range)/2)+1
    BlockInterfacetest_cropped = BlockInterfacetest[{{}, {}, {h_bound, -h_bound-1}, {w_bound, -w_bound-1}}]:cuda()
  end

  if opt.num_noise_FM > 0 then
    local noise = generate_noise(opt.netEnco_vgg_Outputblocksize)
    noise = torch.repeatTensor(noise, opt.num_noise_FM,1,1):cuda()
    local tmp = torch.Tensor(BlockInterfacetest_cropped:size(1), BlockInterfacetest_cropped:size(2)+opt.num_noise_FM, BlockInterfacetest_cropped:size(3), BlockInterfacetest_cropped:size(4))
    for i = 1, BlockInterfacetest_cropped:size(1) do
      tmp[i] = BlockInterfacetest_cropped[i]:cat(noise, 1):double()
    end
    BlockInterfacetest_cropped = tmp:cuda()
  end

  print("importing target images from: " .. opt.testtarget_folder)
  for i_img = 1, opt.batchSize do
    BlockPixel_testtarget[i_img] = image.load(opt.testtarget_folder .. '/' .. list_image_test[i_img] .. '.png', 3)
  end

  print("orig size:")
  print(BlockPixel_testtarget:size())

  BlockPixel_testtarget_crop = BlockPixel_testtarget
  if opt.cropped_inputs == 1 then
    local block_h = BlockPixel_testtarget:size(3)
    local block_w = BlockPixel_testtarget:size(4)
    local range = pixel_blockSize_target-68+1
    local h_bound = math.ceil((block_h-range)/2)+1
    local w_bound = math.ceil((block_w-range)/2)+1
    -- take central 128x128 crop
    BlockPixel_testtarget_crop = BlockPixel_testtarget[{{}, {}, {h_bound, -h_bound}, {w_bound, -w_bound}}]:cuda()
    print("central TEST target crop size:")
    print(BlockPixel_testtarget_crop:size())
  end

  BlockPixel_testtarget_crop:mul(2):add(-1)
  BlockPixel_testtarget_crop = BlockPixel_testtarget_crop:cuda()

  local StyleScore_real = {}
  local StyleScore_G = {}
  local label = {}
  for i_netS = 1, opt.netS_num do
    table.insert(StyleScore_real, torch.Tensor(opt.batchSize * opt.netS_blocksize[i_netS] * opt.netS_blocksize[i_netS]))
    table.insert(StyleScore_G, torch.Tensor(opt.batchSize * opt.netS_blocksize[i_netS] * opt.netS_blocksize[i_netS]))
    table.insert(label, torch.Tensor(opt.batchSize * opt.netS_blocksize[i_netS] * opt.netS_blocksize[i_netS]))
    StyleScore_real[i_netS] = StyleScore_real[i_netS]:cuda()
    StyleScore_G[i_netS] = StyleScore_G[i_netS]:cuda()
    label[i_netS] = label[i_netS]:cuda()
  end


  local errS = {}
  for i_netS = 1, opt.netS_num do
    table.insert(errS, 0)
  end

  local netS_mask = {}
  for i_netS = 1, opt.netS_num do
    table.insert(netS_mask, torch.Tensor(opt.batchSize, 1, opt.netS_blocksize[i_netS], opt.netS_blocksize[i_netS]):fill(1))
    if opt.netS_flag_mask == 1 then
      netS_mask[i_netS][{{1, opt.batchSize}, {1, 1}, {opt.netS_border + 1, opt.netS_blocksize[i_netS] - opt.netS_border}, {opt.netS_border + 1, opt.netS_blocksize[i_netS] - opt.netS_border}}]:fill(0)
    end
    netS_mask[i_netS] = netS_mask[i_netS]:reshape(opt.batchSize * opt.netS_blocksize[i_netS] * opt.netS_blocksize[i_netS], 1)
  end
  for i_netS = 1, opt.netS_num do
    netS_mask[i_netS] = netS_mask[i_netS]:cuda()
  end

  local cur_net_id

  -- discriminator
  local fDx = function(x)
    netS[cur_net_id]:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)
    gradParametersS[cur_net_id]:zero()   

    errS[cur_net_id] = 0

    -- train with real images    
    label[cur_net_id]:fill(opt.real_label)
    
    StyleScore_real[cur_net_id] = netS[cur_net_id]:forward(BlockVGG_target[cur_net_id]):clone()
    local avg_netD_Score_real = torch.sum(StyleScore_real[cur_net_id]) / StyleScore_real[cur_net_id]:nElement()
    if opt.netS_flag_mask == 1 then
      StyleScore_real[cur_net_id][netS_mask[cur_net_id]] = opt.real_label
      avg_netD_Score_real = torch.sum(StyleScore_real[cur_net_id][netS_mask[cur_net_id]:clone():mul(-1):add(1)]) / (StyleScore_real[cur_net_id]:nElement() - torch.sum(netS_mask[cur_net_id]))
    end

    errS[cur_net_id] = errS[cur_net_id] + criterion_netS[cur_net_id]:forward(StyleScore_real[cur_net_id], label[cur_net_id])
    local gradInput_StyleScore_real = criterion_netS[cur_net_id]:backward(StyleScore_real[cur_net_id], label[cur_net_id]):clone()  
    netS[cur_net_id]:backward(BlockVGG_target[cur_net_id], gradInput_StyleScore_real)
    print("netS avg_netD_Score_real: " .. avg_netD_Score_real .. ", gradParametersS: " .. torch.norm(gradParametersS[cur_net_id]))

    label[cur_net_id]:fill(opt.fake_label)

    StyleScore_G[cur_net_id] = netS[cur_net_id]:forward(BlockVGG_G[cur_net_id]):clone()
    local avg_netD_Score_G = torch.sum(StyleScore_G[cur_net_id]) / StyleScore_G[cur_net_id]:nElement()
    if opt.netS_flag_mask == 1 then
      StyleScore_G[cur_net_id][netS_mask[cur_net_id]] = opt.fake_label
      avg_netD_Score_G = torch.sum(StyleScore_G[cur_net_id][netS_mask[cur_net_id]:clone():mul(-1):add(1)]) / (StyleScore_G[cur_net_id]:nElement() - torch.sum(netS_mask[cur_net_id]))
    end
    errS[cur_net_id] = errS[cur_net_id] + criterion_netS[cur_net_id]:forward(StyleScore_G[cur_net_id], label[cur_net_id])
    local gradInput_StyleScore_G = criterion_netS[cur_net_id]:backward(StyleScore_G[cur_net_id], label[cur_net_id]):clone()
    netS[cur_net_id]:backward(BlockVGG_G[cur_net_id], gradInput_StyleScore_G)
    print("netS avg_netD_Score_G: " .. avg_netD_Score_G .. ", gradParametersS: " .. torch.norm(gradParametersS[cur_net_id]))

    gradParametersS[cur_net_id] = gradParametersS[cur_net_id]:div(opt.batchSize)
    
    gradInput_StyleScore_real = nil
    gradInput_StyleScore_G = nil
    collectgarbage()
    collectgarbage()

    return errS[cur_net_id], gradParametersS[cur_net_id]
  end


  -- generator
  local fGx = function(x)

    if opt.ulyanov_loss == 0 then
      netG:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)
      gradparametersG:zero()  
      local gradOutput_G = BlockPixel_G:clone():fill(0)
      errG = 0
      errG_Pixel = 0
      errG_Style = 0

      -- pixel loss
      errG_Pixel = criterion_Pixel:forward(BlockPixel_G, BlockPixel_target_crop)
      local gradOutput_G_Pixel = criterion_Pixel:backward(BlockPixel_G, BlockPixel_target_crop)
      gradOutput_G = gradOutput_G + gradOutput_G_Pixel:mul(opt.pixel_weights)
      errG = errG + errG_Pixel

      -- style loss
      for i_netS = 1, opt.netS_num do
        label[i_netS]:fill(opt.real_label)
        StyleScore_G[i_netS] = netS[i_netS]:forward(BlockVGG_G[i_netS]):clone()
        local avg_netD_Score_G = torch.sum(StyleScore_G[i_netS]) / StyleScore_G[i_netS]:nElement()
        if opt.netS_flag_mask == 1 then
          StyleScore_G[i_netS][netS_mask[i_netS] ] = opt.real_label
          avg_netD_Score_G = torch.sum(StyleScore_G[cur_net_id][netS_mask[cur_net_id]:clone():mul(-1):add(1)]) / (StyleScore_G[cur_net_id]:nElement() - torch.sum(netS_mask[cur_net_id]))
        end
      
        errS[i_netS] = criterion_netS[i_netS]:forward(StyleScore_G[i_netS], label[i_netS])
        errG_Style = errG_Style + errS[i_netS]
        local gradInput_StyleScore_G = criterion_netS[i_netS]:backward(StyleScore_G[i_netS], label[i_netS]):clone()  
        local gradInput_netS = netS[i_netS]:updateGradInput(BlockVGG_G[i_netS], gradInput_StyleScore_G)
        netSVGG[i_netS]:forward(BlockPixel_G)
        local gradInput_Style = netSVGG[i_netS]:updateGradInput(BlockPixel_G, gradInput_netS)
        gradOutput_G = gradOutput_G + gradInput_Style:mul(opt.netS_weights[i_netS])

        gradInput_StyleScore_G = nil
        gradInput_netS = nil
        collectgarbage()
        collectgarbage()
      end

      netG:backward(BlockInterface_cropped, gradOutput_G)
      errG = errG + errG_Style

      gradOutput_G_Pixel = nil
      gradOutput_G = nil
      collectgarbage()
      collectgarbage()
      -- logging
      print(('Epoch: [%d][%8d / %8d]\t Time: %.3f '
             .. 'errG: %.4f'):format(
                 epoch, ((i_iter-1) / opt.batchSize),
                 math.floor(num_target_images / opt.batchSize),
                 tm:time().real, errG and errG or -1))
    else 
      -- Use Dmitry Ulyanov loss function --
      errG = errG + crit:forward({BlockPixel_G, BlockPixel_target_crop})
      -- Backward
      local grad = crit:backward({BlockPixel_G, BlockPixel_target_crop}, nil)
      netG:backward(BlockPixel_source, grad[1])
      errG = errG/params.batch_size
      print('#it: ', iteration, 'Ulyanov loss: ', errG)
    end
    return errG, gradParametersG
  end

  print('*****************************************************')
  print('Training Loop: ');
  print('*****************************************************') 
  local epoch_tm   = torch.Timer()
  local tm         = torch.Timer()
  local data_tm    = torch.Timer()
  local record_err = torch.Tensor(1)

  print('source size: ' .. pixel_blockSize_source .. ', target size: ' .. pixel_blockSize_target)
  local BlockPixel_testsource_crop = torch.Tensor(opt.batchSize, opt.nc, pixel_blockSize_target, pixel_blockSize_target)
  BlockPixel_testsource_crop = BlockPixel_testsource_crop:cuda()
  local BlockPixel_source_crop = torch.Tensor(opt.batchSize, opt.nc, pixel_blockSize_target, pixel_blockSize_target)
  BlockPixel_source_crop = BlockPixel_source_crop:cuda()

  -- take target crops
  if opt.cropped_inputs == 1 then
    local block_h = BlockPixel_target:size(3)
    local block_w = BlockPixel_target:size(4)
    local range = pixel_blockSize_target-68+1
    local h_bound = math.ceil((block_h-range)/2)+1
    local w_bound = math.ceil((block_w-range)/2)+1
    -- take central 128x128 crop
    BlockPixel_testsource_crop = BlockPixel_testsource[{{}, {}, {h_bound, -h_bound}, {w_bound, -w_bound}}]
  end

  print("BlockPixel_testsource_crop size:")
  print(BlockPixel_testsource_crop:size())
  print("BlockPixel_testtarget_crop size:")
  print(BlockPixel_testtarget:size())

  print('compute MSE between source and target')
  record_err:fill(criterion_Pixel:forward(BlockPixel_testsource_crop, BlockPixel_testtarget_crop))

  print("start training loop")
  for epoch = opt.start_epoch, opt.numEpoch do
    local source 
    local sourcetest
    local target
    local targettest
    local generated
    local generatedtest

   local counter = 0
   for i_iter = 1, num_source_images, opt.batchSize do
     tm:reset()
     counter = counter + 1
      
      -- randomly select images to train
      local list_image = torch.Tensor(opt.batchSize)
      print("randomly selecting " .. opt.batchSize .. "patches")
      for i_img = 1, opt.batchSize do
        list_image[i_img] = torch.random(1, num_source_images)
      end  
      
      -- read PatchPixel_real and PatchPixel_photo
      for i_img = 1, opt.batchSize do
        local tmp = image.load(opt.target_folder .. '/' .. list_image[i_img] .. '.png', 3)
        BlockPixel_target[i_img] = tmp
      end
      BlockPixel_target:mul(2):add(-1)
      BlockPixel_target = BlockPixel_target:cuda()
      
      -- take target crops
      if opt.cropped_inputs == 1 then
        local block_h = BlockPixel_target:size(3)
        local block_w = BlockPixel_target:size(4)
        local range = pixel_blockSize_target-68+1
        local h_bound = math.ceil((block_h-range)/2)+1
        local w_bound = math.ceil((block_w-range)/2)+1
        -- take central 128x128 crop
        BlockPixel_target_crop = BlockPixel_target[{{}, {}, {h_bound, -h_bound}, {w_bound, -w_bound}}]
      end

      for i_img = 1, opt.batchSize do
	-- add noise here
	local tmp   = image.load(opt.source_folder .. '/' .. list_image[i_img] ..  '.png', 3)
	local noise = generate_noise(pixel_blockSize_source)
	noise = torch.repeatTensor(noise, 3,1,1)
	tmp:add(noise:mul(opt.training_noise_weight))
	BlockPixel_source[i_img] = tmp
      end
      BlockPixel_source:mul(2):add(-1)
      BlockPixel_source = BlockPixel_source:clone()
      BlockInterface  = netEnco:forward(BlockPixel_source):clone()
      
      BlockInterface_cropped = BlockInterface:clone()

      if opt.cropped_inputs == 1 then
        local block_h = BlockInterface:size(3)
        local block_w = BlockInterface:size(4)
        local range   = 16+1
        local h_bound = math.ceil((block_h-range)/2)+1
        local w_bound = math.ceil((block_w-range)/2)+1
        BlockInterface_cropped = BlockInterface[{{}, {}, {h_bound, -h_bound-1}, {w_bound, -w_bound-1}}]
      end

      if opt.num_noise_FM > 0 then
        local noise = generate_noise(opt.netEnco_vgg_Outputblocksize)
        noise = torch.repeatTensor(noise, opt.num_noise_FM,1,1):cuda()
        local tmp=torch.Tensor(BlockInterface_cropped:size(1), BlockInterface_cropped:size(2)+opt.num_noise_FM, BlockInterface_cropped:size(3), BlockInterface_cropped:size(4))
        for i = 1, BlockInterface_cropped:size(1) do
          tmp[i] = BlockInterface_cropped[i]:cat(noise, 1):double()
        end
        BlockInterface_cropped = tmp:cuda()
      end

      BlockPixel_G = netG:forward(BlockInterface_cropped):clone()

      if epoch >= opt.multi_step_ON then
        local vgg_fm = netEnco:forward(BlockPixel_G)
        BlockPixel_G = netG:forward(vgg_fm)
      end

      for i_netS = 1, opt.netS_num do
        BlockVGG_target[i_netS] = netSVGG[i_netS]:forward(BlockPixel_target_crop:clone()):clone()
        BlockVGG_G[i_netS]      = netSVGG[i_netS]:forward(BlockPixel_G):clone()
      end
      
     -- train netS
      for i_netS = 1, opt.netS_num do
        cur_net_id = i_netS
        optim.adam(fDx, parametersS[cur_net_id], optimStateS[cur_net_id])
      end

      -- train netG
      optim.adam(fGx, parametersG, optimStateG)

      if opt.display then
        print('prep display')

        if opt.cropped_inputs == 1 then
          src      = torch.Tensor(opt.batchSize, opt.nc, pixel_blockSize_target-68, pixel_blockSize_target-68)
          src_test = torch.Tensor(opt.batchSize, opt.nc, pixel_blockSize_target-68, pixel_blockSize_target-68)

          -- use input images of smaller size for display
          for i_img = 1, opt.batchSize do
            local tmp = BlockPixel_source[i_img]:double()
            local noise = generate_noise(pixel_blockSize_source)
            noise = torch.repeatTensor(noise, 3,1,1)
            tmp:add(noise:mul(opt.training_noise_weight))
            local smaller = image.scale(tmp:double(), pixel_blockSize_target-68, pixel_blockSize_target-68, 'bilinear')
            src[i_img] = smaller

            local tmp_test = BlockPixel_testsource[i_img]:double()
            local noise_test = generate_noise(pixel_blockSize_source)
            noise_test = torch.repeatTensor(noise_test, 3,1,1)
            tmp_test:add(noise_test:mul(opt.training_noise_weight))
            local smaller_test = image.scale(tmp_test:double(), pixel_blockSize_target-68, pixel_blockSize_target-68, 'bilinear')
            src_test[i_img] = smaller_test
          end
          source = src
          sourcetest = src_test
        else
          source        = BlockPixel_source
          sourcetest    = BlockPixel_testsource
        end

        target        = BlockPixel_target_crop
        targettest    = BlockPixel_testtarget_crop
        generated     = netG:forward(BlockInterface_cropped):clone()
	generatedtest = netG:forward(BlockInterfacetest_cropped):clone()

        if epoch >= opt.multi_step_ON then
          local vgg_fm = netEnco:forward(generated)
          generated = netG:forward(vgg_fm):clone()
          vgg_fm = nil
          vgg_fm = netEnco:forward(generatedtest)
          generatedtest = netG:forward(vgg_fm):clone()
        end

	disp.image(source, {win=1, title='source'})
        disp.image(target, {win=2, title='target'})
        disp.image(generated, {win=3, title='generated'})
      end

      if counter == math.floor(num_target_images / opt.batchSize) or counter % opt.save_interval_image == 0 then
        print('display')
        local img_source = image.toDisplayTensor{input = source, nrow = math.ceil(math.sqrt(source:size(1)))}
        local img_sourcetest = image.toDisplayTensor{input = sourcetest, nrow = math.ceil(math.sqrt(sourcetest:size(1)))} 
        local img_target = image.toDisplayTensor{input = target, nrow = math.ceil(math.sqrt(target:size(1)))}
        local img_targettest = image.toDisplayTensor{input = targettest, nrow = math.ceil(math.sqrt(targettest:size(1)))}
        local img_generated = image.toDisplayTensor{input = generated, nrow = math.ceil(math.sqrt(generated:size(1)))}
        local img_generatedtest = image.toDisplayTensor{input = generatedtest, nrow = math.ceil(math.sqrt(generatedtest:size(1)))}
        local img_out = torch.Tensor(img_generated:size()[1], img_generated:size()[2] + img_generated:size()[2] + 16, img_generated:size()[3] + img_generated:size()[3] + img_generated:size()[3] + 32):fill(0):cuda()

        img_out[{{1,img_generated:size()[1]}, {1,img_generated:size()[2]}, {1,img_generated:size(3)}}] = img_source
        img_out[{{1,img_generated:size()[1]}, {1,img_generated:size()[2]}, {img_out:size()[3] - img_generated:size()[3] - 16 - img_generated:size()[3] + 1, img_out:size()[3] - img_generated:size()[3] - 16}}] = img_generated
        img_out[{{1,img_generated:size()[1]}, {1,img_generated:size()[2]}, {img_out:size()[3] - img_generated:size()[3] + 1, img_out:size()[3]}}] = img_target

        img_out[{{1,img_generated:size()[1]}, {img_out:size()[2] - img_generated:size()[2] + 1, img_out:size()[2]}, {1,img_generated:size(3)}}] = img_sourcetest
        img_out[{{1,img_generated:size()[1]}, {img_out:size()[2] - img_generated:size()[2] + 1, img_out:size()[2]}, {img_out:size()[3] - img_generated:size()[3] - 16 - img_generated:size()[3] + 1, img_out:size()[3] - img_generated:size()[3] - 16}}] = img_generatedtest
        img_out[{{1,img_generated:size()[1]}, {img_out:size()[2] - img_generated:size()[2] + 1, img_out:size()[2]}, {img_out:size()[3] - img_generated:size()[3] + 1, img_out:size()[3]}}] = img_targettest

        -- save image
        image.save(opt.dataset_name .. opt.experiment_name .. 'epoch_' .. epoch .. '_' .. counter .. '.png', img_out)
        -- save model
        util.save(opt.dataset_name .. opt.experiment_name .. 'epoch_' .. epoch .. '_' .. counter .. '_netG.t7', netG, opt.gpu)
        parametersG, gradparametersG = netG:getParameters()
     end 
   end

    parametersG = nil
    gradparametersG = nil
    util.save(opt.dataset_name .. opt.experiment_name .. 'epoch_' .. epoch .. '_netG.t7', netG, opt.gpu)
    parametersG, gradparametersG = netG:getParameters()

    for i_netS = 1, opt.netS_num do 
      print(opt.dataset_name .. opt.experiment_name .. 'epoch_' .. epoch .. '_netS_' .. i_netS .. '.t7')
      parametersS[i_netS] = nil
      gradParametersS[i_netS] = nil
      util.save(opt.dataset_name .. opt.experiment_name .. 'epoch_' .. epoch .. '_netS_' .. i_netS .. '.t7', netS[i_netS], opt.gpu)
      parametersS[i_netS], gradParametersS[i_netS] = netS[i_netS]:getParameters()
    end  

    record_err = torch.cat(record_err, torch.Tensor(1):fill(criterion_Pixel:forward(generatedtest, targettest)), 1)
    print(record_err)
    disp.plot(torch.cat(torch.linspace(0, record_err:nElement(), record_err:nElement()), record_err, 2), {win=7, title='energy'})
  end -- for epoch = opt.start_epoch, opt.numEpoch do

  netVGG = nil
  for i_netS = 1, opt.netS_num do 
    netS[i_netS] = nil
  end  
  for i_netS = 1, opt.netS_num do 
    netSVGG[i_netS] = nil
  end  
  netEnco = nil
  netG = nil

  collectgarbage()
  collectgarbage()
  
  return flag_state
end

return {
  state = run_MGAN
}

