local TVLoss2, parent = torch.class('nn.TVLoss2', 'nn.Module')

function TVLoss2:__init(strength)
  parent.__init(self)
  self.strength = strength
  self.x_diff = torch.Tensor()
  self.y_diff = torch.Tensor()
end

------------------------------------------------------------------------
-- TVLoss
------------------------------------------------------------------------
function TVLoss2:updateOutput(input)
  self.output = input
  return self.output
end

-- TV loss backward pass inspired by kaishengtai/neuralart
function TVLoss2:updateGradInput(input, gradOutput)
  self.gradInput:resizeAs(input):zero()
  local B, C, H, W = input:size(1), input:size(2), input:size(3), input:size(4)
  self.x_diff:resize(B, C, H - 1, W - 1)
  self.y_diff:resize(B, C, H - 1, W - 1)
  -- print(self.x_diff:size())
  -- print(input[{{}, {}, {1, -2}, {1, -2}}]:size())
  self.x_diff:copy(input[{{}, {}, {1, -2}, {1, -2}}])
  self.x_diff:add(-1, input[{{}, {}, {1, -2}, {2, -1}}])
  self.y_diff:copy(input[{{}, {}, {1, -2}, {1, -2}}])
  self.y_diff:add(-1, input[{{}, {}, {2, -1}, {1, -2}}])
  self.gradInput[{{}, {}, {1, -2}, {1, -2}}]:add(self.x_diff):add(self.y_diff)
  
  local epsilon = 1e-8
  self.gradInput[{{}, {}, {1, -2}, {1, -2}}]:cdiv(torch.sqrt(torch.add(torch.pow(self.x_diff,2), torch.pow(self.y_diff,2)):add(epsilon)))

  -- self.gradInput[{{}, {}, {1, -2}, {2, -1}}]:add(-1, self.x_diff)
  -- self.gradInput[{{}, {}, {2, -1}, {1, -2}}]:add(-1, self.y_diff)

  self.gradInput = torch.sign(self.gradInput)

  self.gradInput:mul(self.strength)
  self.gradInput:add(gradOutput)
  return self.gradInput
end
