local M = {}

local nchannel = 3
local image_size = 32
local nclasses = 10
--local featureMapSize = nclasses * nchannel * image_size * image_size
local featureSize = 4096
local featureMapSize = nclasses * featureSize

local function featureVec(x, y)
	local feature = torch.Tensor(featureMapSize):zero()
	offset = y*featureSize
	feature[{{offset + 1, offset + featureSize}}] = x
	return feature

end

local function loss(y1, y2)
	if y1 ~= y2 then
		return 1
	else
		return 0
	end
end

local function sparseDot(w, x, y)
	local startId = y*featureSize + 1
	local endId = (y + 1)*featureSize
	local w_y = w[{{startId, endId}}]
	return w_y:dot(x)
end

local function maxOracle(x, y, w)
	local maxVal = - math.huge 
	local maxy = -1
	for i = 0, 9 do
		local a = - sparseDot(w, x, i)
		if loss(y, i) - a > maxVal then
			maxy = i
			maxVal = loss(y, i) - a 
		end
	end
	return maxy
end

local function getDual(w, l, lambda)
	if debug_flag == 1 then
		print("In objective function..\n")
		print("lambda/2 * torch.norm(w)^2: " .. lambda/2 * torch.norm(w)^2 .. "\n")
		print("l: " .. l .. "\n")
	end
	return - (lambda/2 * w:dot(w) - l)
end

local function duality_gap(w, lambda, l, n, minitrainset_data, minitrainset_label)
	local ws = torch.Tensor(featureMapSize):zero()
	local ls = 0
	for i = 1, n do
		local x = minitrainset_data[i]
		local y = minitrainset_label[i]
		local max_y = maxOracle(x, y, w)
		local psi_i = featureVec(x, y) - featureVec(x, max_y)
		ws = ws + 1/(n*lambda) * psi_i 
		ls = ls + 1/n * loss(y, max_y)
	end
	local dualGap = (lambda * w:dot(w - ws) - l + ls)
	return dualGap
end

local function getPrimal(w, lambda, dataset_data, dataset_label)
	local primalVal = 0
	primalVal = primalVal + (lambda/2) * w:dot(w)
	local n = trainset_size
	for i = 1, n do
		local x = dataset_data[i]
		local y = dataset_label[i]
		local max_y = maxOracle(x, y, w)
		local maxVal = loss(y, max_y) - (spareDot(w, x, y) - sparseDot(w, x, max_y))
		primalVal = primalVal + (1/n) * maxVal
	end
	return primalVal	
end	

local function predict_y(x, w)
	local y = -1
	local maxVal = -100
	for i = 0, 9 do
		local a = sparseDot(w, x, i) 
		if a > maxVal then
			y = i
			maxVal = a
		end
	end
	return y
end

local function average_loss(trainset_data, trainset_label, w)
	local lossVal = 0
	n = trainset_size
	for i = 1, n do
		local x = trainset_data[i]
		local y = trainset_label[i]
		local y_pred = predict_y(x, w)
		if y ~= y_pred then
			lossVal = lossVal + loss(y, y_pred) 
		end
	end
	return  lossVal/n
end

M.featureVec = featureVec
M.loss = loss
M.maxOracle = maxOracle
M.getDual = getDual
M.duality_gap = duality_gap
M.getPrimal = getPrimal
M.average_loss = average_loss
M.predict_y = predict_y

return M
