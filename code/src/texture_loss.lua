-- Returns a network that computes batch of CxC Gram matrix from inputs
function GramMatrix()
  local net = nn.Sequential()
  local concat = nn.ConcatTable()
  concat:add(nn.Identity())
  concat:add(nn.Identity())
  net:add(concat)
  net:add(nn.MM(false, true))
  return net
end

local TextureLoss, parent = torch.class('nn.TextureLoss', 'nn.Module')

function TextureLoss:__init(strength, target, normalize)
  parent.__init(self)
  self.normalize = normalize or false
  self.strength = strength

  local tsz = target:size()
  self.target = target:add_dummy()
  
  self.loss = 0

  self.gram = GramMatrix()
  self.G = nil
  self.crit = nn.MSECriterion()
  self.active = true
end


function TextureLoss:updateOutput(input)
  if self.active then 
    -- input is 4d 
    local sz = input:size()

    -- now batch_size x C x WH
    local input3d = input:view(sz[1], sz[2], sz[3]*sz[4])

    self.G = self.gram:forward(input3d)
    self.G:div(input[1]:nElement())
    self.match_to = self.target:expandAs(self.G)
    
    self.loss = self.crit:forward(self.G, self.match_to)
    self.loss = self.loss * self.strength

  end
  self.output = input
  return self.output
end

function TextureLoss:updateGradInput(input, gradOutput)
  if self.active then
    local dG = self.crit:backward(self.G, self.match_to)
    dG:div(input[1]:nElement())

    local sz = input:size()
    local input3d = input:view(sz[1], sz[2], sz[3]*sz[4])
    self.gradInput = self.gram:backward(input3d, dG):viewAs(input)

    -- if self.normalize then
    --   self.gradInput:div(torch.norm(self.gradInput, 1) + 1e-8)
    -- end

    self.gradInput:mul(self.strength)
    self.gradInput:add(gradOutput)
  else
    self.gradInput  = gradOutput
  end
  return self.gradInput
end