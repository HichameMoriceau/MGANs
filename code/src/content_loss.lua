local ContentLoss, parent = torch.class('nn.ContentLoss', 'nn.Module')

function ContentLoss:__init(strength, normalize)
  parent.__init(self)
  self.strength = strength
  self.target = torch.Tensor()
  self.normalize = normalize or false
  self.loss = 0
  self.crit = nn.MSECriterion()
  self.active = true
end

function ContentLoss:updateOutput(input)
  if self.active then
    if input:nElement() == self.target:nElement() then
      self.loss = self.crit:forward(input, self.target) * self.strength
    end
  end
  
  self.output = input
  return self.output
end

function ContentLoss:updateGradInput(input, gradOutput)
  if self.active then
    if input:nElement() == self.target:nElement() then
      self.gradInput = self.crit:backward(input, self.target)
    end
    if self.normalize then
      self.gradInput:div(torch.norm(self.gradInput, 1) + 1e-8)
    end
    self.gradInput:mul(self.strength)
    self.gradInput:add(gradOutput)
  else
    self.gradInput = gradOutput
  end
  return self.gradInput
end