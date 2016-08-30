require 'nn'
require 'loadcaffe'
require 'src/SpatialCircularPadding'

----------------------------------------------------------
-- Shortcuts 
----------------------------------------------------------
function conv(in_,out_, k, s, m)
    m = m or 1
    s = s or 1

    local to_pad = (k-1)/2*m

    if to_pad == 0 then
      return backend.SpatialConvolution(in_, out_, k, k, s, s, 0, 0)
    else

      local net = nn.Sequential()
      net:add(pad(to_pad,to_pad,to_pad,to_pad))
      net:add(backend.SpatialConvolution(in_, out_, k, k, s, s, 0, 0))

      return net
    end
end

---------------------------------------------------------
-- Helper function
---------------------------------------------------------

-- from fb.resnet.torch
function deepCopy(tbl)
   -- creates a copy of a network with new modules and the same tensors
   local copy = {}
   for k, v in pairs(tbl) do
      if type(v) == 'table' then
         copy[k] = deepCopy(v)
      else
         copy[k] = v
      end
   end
   if torch.typename(tbl) then
      torch.setmetatable(copy, torch.typename(tbl))
   end
   return copy
end

-- adds first dummy dimension
function torch.Tensor:add_dummy()
  local sz = self:size()
  local new_sz = torch.Tensor(sz:size()+1)
  new_sz[1] = 1
  new_sz:narrow(1,2,sz:size()):copy(torch.Tensor{sz:totable()})

  if self:isContiguous() then
    return self:view(new_sz:long():storage())
  else
    return self:reshape(new_sz:long():storage())
  end
end

if cutorch then 
  torch.CudaTensor.add_dummy = torch.Tensor.add_dummy
end

---------------------------------------------------------
-- DummyGradOutput
---------------------------------------------------------

-- Simpulates Identity operation with 0 gradOutput
local DummyGradOutput, parent = torch.class('nn.DummyGradOutput', 'nn.Module')

function DummyGradOutput:__init()
  parent.__init(self)
  self.gradInput = nil
end


function DummyGradOutput:updateOutput(input)
  self.output = input
  return self.output
end

function DummyGradOutput:updateGradInput(input, gradOutput)
  self.gradInput = self.gradInput or input.new():resizeAs(input):fill(0)
  if not input:isSameSizeAs(self.gradInput) then
    self.gradInput = self.gradInput:resizeAs(input):fill(0)
  end  
  return self.gradInput 
end

----------------------------------------------------------
-- NoiseFill 
----------------------------------------------------------
-- Fills last `num_noise_channels` channels of an existing `input` tensor with noise. 
local NoiseFill, parent = torch.class('nn.NoiseFill', 'nn.Module')

function NoiseFill:__init(num_noise_channels)
  parent.__init(self)

  -- last `num_noise_channels` maps will be filled with noise
  self.num_noise_channels = num_noise_channels
  self.mult = 1.0
end

function NoiseFill:updateOutput(input)
  self.output = self.output or input:new()
  self.output:resizeAs(input)

  -- copy non-noise part
  if self.num_noise_channels ~= input:size(2) then
    local ch_to_copy = input:size(2) - self.num_noise_channels
    self.output:narrow(2,1,ch_to_copy):copy(input:narrow(2,1,ch_to_copy))
  end

  -- fill noise
  if self.num_noise_channels > 0 then
    local num_channels = input:size(2)
    local first_noise_channel = num_channels - self.num_noise_channels + 1

    self.output:narrow(2,first_noise_channel, self.num_noise_channels):uniform():mul(self.mult)
  end
  return self.output
end

function NoiseFill:updateGradInput(input, gradOutput)
   self.gradInput = gradOutput
   return self.gradInput
end

----------------------------------------------------------
-- GenNoise 
----------------------------------------------------------
-- Generates a new tensor with noise of spatial size as `input`
-- Forgets about `input` returning 0 gradInput.

local GenNoise, parent = torch.class('nn.GenNoise', 'nn.Module')

function  GenNoise:__init(num_planes)
    self.num_planes = num_planes
    self.mult = 1.0
end
function GenNoise:updateOutput(input)
    self.sz = input:size()

    self.sz_ = input:size()
    self.sz_[2] = self.num_planes

    self.output = self.output or input.new()
    self.output:resize(self.sz_)
    
    -- It is concated with normed data, so gen from N(0,1)
    self.output:normal(0,1):mul(self.mult)

   return self.output
end

function GenNoise:updateGradInput(input, gradOutput)
   self.gradInput = self.gradInput or gradOutput.new()
   self.gradInput:resizeAs(input):zero()
   
   return self.gradInput
end

---------------------------------------------------------
-- Image processing
---------------------------------------------------------

-- Preprocess an image before passing it to a Caffe model.
-- We need to rescale from [0, 1] to [0, 255], convert from RGB to BGR,
-- and subtract the mean pixel.
function preprocess(img)
  local mean_pixel = torch.FloatTensor({103.939, 116.779, 123.68})
  local perm = torch.LongTensor{3, 2, 1}
  img = img:index(1, perm):mul(255.0)
  mean_pixel = mean_pixel:view(3, 1, 1):expandAs(img)
  img:add(-1, mean_pixel)
  return img
end

function preprocess1(images)
  local out = images:clone()
  for i=1, images:size(1) do
    out[i] = preprocess(images[i]:clone())
  end
  return out
end


-- Undo the above preprocessing.
function deprocess(img)
  local mean_pixel = torch.DoubleTensor({103.939, 116.779, 123.68})
  mean_pixel = mean_pixel:view(3, 1, 1):expandAs(img)
  img = img + mean_pixel
  local perm = torch.LongTensor{3, 2, 1}
  img = img:index(1, perm):div(255.0)
  return img
end

-----Almost copy paste of jcjohnson s code -----------
local TVLoss_ULYANOV, parent = torch.class('nn.TVLoss_ULYANOV', 'nn.Module')

function TVLoss_ULYANOV:__init(strength)
  parent.__init(self)
  print('Using TV loss with weight ', strength)
  self.strength = strength
  self.x_diff = torch.Tensor()
  self.y_diff = torch.Tensor()
end

function TVLoss_ULYANOV:updateOutput(input)
  self.output = input
  return self.output
end

-- TV loss backward pass inspired by kaishengtai/neuralart
function TVLoss_ULYANOV:updateGradInput(input, gradOutput)
  self.gradInput:resizeAs(input):zero()
  
  for obj = 1, input:size(1) do
    local input_= input[obj]
    local C, H, W = input_:size(1), input_:size(2), input_:size(3)
    self.x_diff:resize(3, H - 1, W - 1)
    self.y_diff:resize(3, H - 1, W - 1)
    self.x_diff:copy(input_[{{}, {1, -2}, {1, -2}}])
    self.x_diff:add(-1, input_[{{}, {1, -2}, {2, -1}}])
    self.y_diff:copy(input_[{{}, {1, -2}, {1, -2}}])
    self.y_diff:add(-1, input_[{{}, {2, -1}, {1, -2}}])
    self.gradInput[obj][{{}, {1, -2}, {1, -2}}]:add(self.x_diff):add(self.y_diff)
    self.gradInput[obj][{{}, {1, -2}, {2, -1}}]:add(-1, self.x_diff)
    self.gradInput[obj][{{}, {2, -1}, {1, -2}}]:add(-1, self.y_diff)
  end

  self.gradInput:mul(self.strength)
  self.gradInput:add(gradOutput)

  return self.gradInput
end
